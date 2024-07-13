""" Author: Yifan Lu <yifan_lu@sjtu.edu.cn>

HEAL: An Extensible Framework for Open Heterogeneous Collaborative Perception 
"""

from collections import OrderedDict, Counter
from typing import Dict, Mapping

import torch
import torch.nn as nn
import torchvision

from opencood.logger import get_logger
from opencood.models.fuse_modules.pyramid_fuse import PyramidFusion
from opencood.models.heter_encoders import encoders
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.feature_alignnet import AlignNet
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.utils.model_utils import check_trainable_module
from opencood.utils.transformation_utils import normalize_pairwise_tfm

logger = get_logger()


class HeterPyramidCollab(nn.Module):
    def __init__(self, args: Dict):
        super().__init__()
        modality_name_list = list(args.keys())
        # 把 model-args-m${number}选出来
        modality_name_list = [x for x in modality_name_list if x.startswith("m") and x[1:].isdigit()]
        # self.modality_name_set = set(modality_name_list)
        self.modality_name_list = modality_name_list

        # TODO: 我总是认为配置文件里面参数的名字应该和程序里变量的名字一致才比较好.
        self.cav_range = args['lidar_range']
        self.sensor_type_dict = OrderedDict()

        self.cam_crop_info = {}

        # setup each modality model (设置每个模态模型) m1, m2, m3, m4
        # 每个模态有各自不同的 encoder, backbone, Aligner
        for modality_name in self.modality_name_list:
            model_setting = args[modality_name]
            sensor_name = model_setting['sensor_type']
            self.sensor_type_dict[modality_name] = sensor_name

            # import model
            # encoder_filename = "opencood.models.heter_encoders"
            # encoder_lib = importlib.import_module(encoder_filename)
            # encoder_class = None
            # target_model_name = model_setting['core_method'].replace('_', '')
            #
            # for name, cls in encoder_lib.__dict__.items():
            #     if name.lower() == target_model_name.lower():
            #         encoder_class = cls
            # TODO: 如果找到了直接 break 是否更好一些呢?
            try:
                encoder_class = encoders[model_setting['core_method']]
            except KeyError:
                available_encoders = ', '.join(encoders.keys())
                logger.error(
                    f'不受支持的 encoder. 选择的 encoder 为: {model_setting["core_method"]}.'
                    f'可用的 encoders: {available_encoders}')
                exit(-1)

            """Encoder building"""
            # setattr 是 Python 内置的一个函数，用于动态地为对象设置属性, python真的十分灵活......
            setattr(self, f"encoder_{modality_name}", encoder_class(model_setting['encoder_args']))
            # 判断是否启用了深度监督
            if model_setting['encoder_args'].get("depth_supervision", False):
                setattr(self, f"depth_supervision_{modality_name}", True)
            else:
                setattr(self, f"depth_supervision_{modality_name}", False)

            """Backbone building"""
            setattr(self, f"backbone_{modality_name}", ResNetBEVBackbone(model_setting['backbone_args']))

            """Aligner building"""
            setattr(self, f"aligner_{modality_name}", AlignNet(model_setting['aligner_args']))
            if sensor_name == "camera":
                camera_mask_args = model_setting['camera_mask_args']
                setattr(self, f"crop_ratio_W_{modality_name}",
                        (self.cav_range[3]) / (camera_mask_args['grid_conf']['xbound'][1]))
                setattr(self, f"crop_ratio_H_{modality_name}",
                        (self.cav_range[4]) / (camera_mask_args['grid_conf']['ybound'][1]))
                setattr(self, f"xdist_{modality_name}",
                        (camera_mask_args['grid_conf']['xbound'][1] - camera_mask_args['grid_conf']['xbound'][0]))
                setattr(self, f"ydist_{modality_name}",
                        (camera_mask_args['grid_conf']['ybound'][1] - camera_mask_args['grid_conf']['ybound'][0]))
                self.cam_crop_info[modality_name] = {
                    f"crop_ratio_W_{modality_name}": eval(f"self.crop_ratio_W_{modality_name}"),
                    f"crop_ratio_H_{modality_name}": eval(f"self.crop_ratio_H_{modality_name}"),
                }

        """
        For feature transformation
        例子 (具体计算方法与配置文件有关):
            H = 51.2 - (-51.2)
            W = 102.4 - (-102.4)
        """
        self.H = (self.cav_range[4] - self.cav_range[1])
        self.W = (self.cav_range[3] - self.cav_range[0])
        self.fake_voxel_size = 1  # TODO: 这是什么意思?

        """
        Fusion, by default multiscale fusion: 
        Note the input of PyramidFusion has downsampled 2x. (SECOND required)
        """
        self.pyramid_backbone = PyramidFusion(args['fusion_backbone'])

        """
        Shrink header
        """
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])

        """
        Shared Heads
        """
        self.cls_head = nn.Conv2d(args['in_head'], args['anchor_number'], kernel_size=1)
        self.reg_head = nn.Conv2d(args['in_head'], 7 * args['anchor_number'], kernel_size=1)

        # BIN_NUM = 2
        self.dir_head = nn.Conv2d(args['in_head'], args['dir_args']['num_bins'] * args['anchor_number'], kernel_size=1)

        # compressor will be only trainable
        self.compress = False
        if 'compressor' in args:
            self.compress = True
            self.compressor = NaiveCompressor(args['compressor']['input_dim'], args['compressor']['compress_ratio'])

        # self.model_train_init()  # TODO: 在主函数中已经调用, 这里有必要再调用一次吗?
        # check again which module is not fixed.
        # TODO: 为什么要进行这一步的检查呢? 这一步的检查有什么意义吗?
        check_trainable_module(self)

    def model_train_init(self):
        if not self.compress:
            return
        # if compress, only make compressor trainable
        # freeze all
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)
        # unfreeze compressor
        self.compressor.train()
        for p in self.compressor.parameters():
            p.requires_grad_(True)

    def forward(self, data_dict: Mapping):
        output_dict = {'pyramid': 'collab'}
        agent_modality_list = data_dict['agent_modality_list']
        affine_matrix = normalize_pairwise_tfm(data_dict['pairwise_t_matrix'], self.H, self.W, self.fake_voxel_size)
        record_len = data_dict['record_len']
        # print(agent_modality_list)
        modality_count_dict = Counter(agent_modality_list)
        modality_feature_dict = {}

        # TODO: 与其让其遍历倒不如, 将 `self.modality_name_list` 与 `modality_count_dict` 都转换成 set,
        #  然后两者取交集即可后转换成  list 再进行计算, 然后下面两个就能和在一起了

        # 遍历模型的每个模态, 如果有一致的就调用对应模态的层进行训练
        for modality_name in self.modality_name_list:
            if modality_name not in modality_count_dict:
                continue
            feature = eval(f"self.encoder_{modality_name}")(data_dict, modality_name)
            feature = eval(f"self.backbone_{modality_name}")({"spatial_features": feature})['spatial_features_2d']
            feature = eval(f"self.aligner_{modality_name}")(feature)
            # 提取到每个模态的 feature, 并将其放到 `modality_feature_dict` 供下一步操作
            modality_feature_dict[modality_name] = feature

        """
        Crop/Padding camera feature map.
        # Crop（裁剪）
        裁剪是指从原始图像或特征图中提取出一个特定的子区域。这种操作通常用于去除不需要的边缘区域或者提取感兴趣的区域。
        裁剪可以通过指定起始坐标和目标大小来完成。例如，如果有一张 256x256 像素的图像，
        裁剪一个从 (50, 50) 开始的 128x128 区域，就可以得到一个新的 128x128 像素的图像。
        
        # Padding（填充）
        填充是指在图像或特征图的边缘添加额外的像素，以增加图像的尺寸。填充操作通常用于确保图像尺寸符合模型的输入要求，
        或者在卷积操作时保持边缘信息。填充的像素值可以是零（称为零填充），也可以是其他值，例如镜像填充、常数填充等。
        """
        for modality_name in self.modality_name_list:
            if modality_name not in modality_count_dict:
                continue

            if self.sensor_type_dict[modality_name] == "camera":
                # should be padding. Instead of masking
                feature = modality_feature_dict[modality_name]
                _, _, H, W = feature.shape
                target_H = int(H * eval(f"self.crop_ratio_H_{modality_name}"))
                target_W = int(W * eval(f"self.crop_ratio_W_{modality_name}"))

                crop_func = torchvision.transforms.CenterCrop((target_H, target_W))
                modality_feature_dict[modality_name] = crop_func(feature)
                if eval(f"self.depth_supervision_{modality_name}"):
                    output_dict.update({
                        f"depth_items_{modality_name}": eval(f"self.encoder_{modality_name}").depth_items
                    })

        """
        Assemble heter features
        """
        counting_dict = {modality_name: 0 for modality_name in self.modality_name_list}
        heter_feature_2d_list = []
        for modality_name in agent_modality_list:
            feat_idx = counting_dict[modality_name]
            heter_feature_2d_list.append(modality_feature_dict[modality_name][feat_idx])
            counting_dict[modality_name] += 1

        heter_feature_2d = torch.stack(heter_feature_2d_list)

        if self.compress:
            heter_feature_2d = self.compressor(heter_feature_2d)

        # heter_feature_2d is downsampled 2x
        # add croping information to collaboration module

        fused_feature, occ_outputs = self.pyramid_backbone.forward_collab(
            heter_feature_2d, record_len, affine_matrix, agent_modality_list, self.cam_crop_info)

        if self.shrink_flag:
            fused_feature = self.shrink_conv(fused_feature)

        cls_preds = self.cls_head(fused_feature)
        reg_preds = self.reg_head(fused_feature)
        dir_preds = self.dir_head(fused_feature)

        output_dict.update({'cls_preds': cls_preds,
                            'reg_preds': reg_preds,
                            'dir_preds': dir_preds})

        output_dict.update({'occ_single_list': occ_outputs})

        return output_dict
