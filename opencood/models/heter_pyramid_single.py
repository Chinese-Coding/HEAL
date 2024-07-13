""" Author: Yifan Lu <yifan_lu@sjtu.edu.cn>

HEAL: An Extensible Framework for Open Heterogeneous Collaborative Perception 
"""

from collections import OrderedDict
from typing import Dict

import torch.nn as nn
import torchvision

from opencood.logger import get_logger
from opencood.models.fuse_modules.pyramid_fuse import PyramidFusion
from opencood.models.heter_encoders import encoders
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.feature_alignnet import AlignNet
from opencood.utils.model_utils import check_trainable_module, fix_bn

logger = get_logger()


class HeterPyramidSingle(nn.Module):
    def __init__(self, args: Dict):
        super().__init__()
        modality_name_list = list(args.keys())
        modality_name_list = [x for x in modality_name_list if x.startswith("m") and x[1:].isdigit()]
        self.modality_name_list = modality_name_list
        self.cav_range = args['lidar_range']
        self.sensor_type_dict = OrderedDict()
        self.fix_modules = ['pyramid_backbone', 'cls_head', 'reg_head', 'dir_head']

        # setup each modality model
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

            # TODO: 这样利用字典从 `opencood.models.heter_encoders` 中引入, 看起来有些死板, 但是 IDE 的代码提示变好了啊
            #        也许将 `encoder` 再单独从这个文件里面提取出来比较好. 配置文件中 `encoder` 的命名方式最好和代码中的一样
            try:
                encoder_class = encoders[model_setting['core_method']]
            except KeyError:
                available_encoders = ', '.join(encoders.keys())
                logger.error(
                    f'不受支持的 encoder. 选择的 encoder 为: {model_setting["core_method"]}.'
                    f'可用的 encoders: {available_encoders}')
                exit(-1)

            # build encoder
            setattr(self, f"encoder_{modality_name}", encoder_class(model_setting['encoder_args']))
            # depth supervision for camera
            if model_setting['encoder_args'].get("depth_supervision", False):
                setattr(self, f"depth_supervision_{modality_name}", True)
            else:
                setattr(self, f"depth_supervision_{modality_name}", False)

            # setup backbone (very light-weight)
            setattr(self, f"backbone_{modality_name}", ResNetBEVBackbone(model_setting['backbone_args']))
            if sensor_name == "camera":
                camera_mask_args = model_setting['camera_mask_args']
                setattr(self, f"crop_ratio_W_{modality_name}",
                        (self.cav_range[3]) / (camera_mask_args['grid_conf']['xbound'][1]))
                setattr(self, f"crop_ratio_H_{modality_name}",
                        (self.cav_range[4]) / (camera_mask_args['grid_conf']['ybound'][1]))

            setattr(self, f"aligner_{modality_name}", AlignNet(model_setting['aligner_args']))

            if args.get("fix_encoder", False):
                self.fix_modules += [f"encoder_{modality_name}", f"backbone_{modality_name}"]

        """
        Would load from pretrain base.
        """
        self.pyramid_backbone = PyramidFusion(args['fusion_backbone'])
        """
        Shrink header, Would load from pretrain base.
        """
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
            self.fix_modules.append('shrink_conv')

        """
        Shared Heads, Would load from pretrain base.
        """
        self.cls_head = nn.Conv2d(args['in_head'], args['anchor_number'], kernel_size=1)
        self.reg_head = nn.Conv2d(args['in_head'], 7 * args['anchor_number'], kernel_size=1)
        # BIN_NUM = 2
        self.dir_head = nn.Conv2d(args['in_head'], args['dir_args']['num_bins'] * args['anchor_number'], kernel_size=1)

        # self.model_train_init() # TODO: 在主函数中已经调用, 这里有必要再调用一次吗?
        # check again which module is not fixed.
        # TODO: 为什么要进行这一步的检查呢? 这一步的检查有什么意义吗?
        check_trainable_module(self)

    def model_train_init(self):
        for module in self.fix_modules:
            for p in eval(f"self.{module}").parameters():
                p.requires_grad_(False)
            eval(f"self.{module}").apply(fix_bn)

    def forward(self, data_dict):
        output_dict = {'pyramid': 'single'}
        modality_name = [x for x in list(data_dict.keys()) if x.startswith("inputs_")]
        assert len(modality_name) == 1
        modality_name = modality_name[0].lstrip('inputs_')

        feature = eval(f"self.encoder_{modality_name}")(data_dict, modality_name)
        feature = eval(f"self.backbone_{modality_name}")({"spatial_features": feature})['spatial_features_2d']
        feature = eval(f"self.aligner_{modality_name}")(feature)

        if self.sensor_type_dict[modality_name] == "camera":
            # should be padding. Instead of masking
            _, _, H, W = feature.shape
            feature = torchvision.transforms.CenterCrop(
                (int(H * eval(f"self.crop_ratio_H_{modality_name}")),
                 int(W * eval(f"self.crop_ratio_W_{modality_name}")))
            )(feature)

            if eval(f"self.depth_supervision_{modality_name}"):
                output_dict.update({
                    f"depth_items_{modality_name}": eval(f"self.encoder_{modality_name}").depth_items
                })

        # multiscale fusion. 
        feature, occ_map_list = self.pyramid_backbone.forward_single(feature)

        if self.shrink_flag:
            feature = self.shrink_conv(feature)

        cls_preds = self.cls_head(feature)
        reg_preds = self.reg_head(feature)
        dir_preds = self.dir_head(feature)

        output_dict.update({'cls_preds': cls_preds,
                            'reg_preds': reg_preds,
                            'dir_preds': dir_preds})
        output_dict.update({'occ_single_list':
                                occ_map_list})
        return output_dict
