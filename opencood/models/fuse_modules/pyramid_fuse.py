# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib
from typing import Mapping

import torch
import torch.nn as nn
from torch import Tensor

from opencood.models.fuse_modules.fusion_in_one import regroup
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.resblock import ResNetModified, Bottleneck
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple
from opencood.logger import get_logger

logger = get_logger()


def weighted_fuse(x, score, record_len, affine_matrix, align_corners):
    """
    weighted: 加权
    GPT: 该函数用于将多个特征图进行加权融合，主要应用于多视角特征对齐和融合。
    在计算机视觉任务中，尤其是多视角感知任务（如自动驾驶中的多摄像头融合），这种方法非常有用。
    这段代码通过仿射变换对齐多视角特征图，并利用归一化后的分数进行加权融合，输出融合后的特征图。
    这在多视角感知任务中可以显著提高特征的对齐精度和融合效果。
    Parameters
    ----------
    x : torch.Tensor
        input data, (sum(n_cav), C, H, W)
    
    score : torch.Tensor
        score, (sum(n_cav), 1, H, W)
        
    record_len : Tensor
        shape: (B)
        
    affine_matrix : torch.Tensor
        affine_matrix 怎么来的? 该矩阵通过 normalize_pairwise_tfm 函数得来, 而这个函数中又有着一个非常重要的参数: `pairwise_t_matrix`
        `pairwise_t_matrix` 矩阵是在 dataset 中定义和计算好了的. 可见, dataset 中并没有对齐坐标, 而是将它放到了融合阶段 (进行特征上的对齐)
        对齐这个概念在调用者 `HeterPyramidCollab` 中已经定义过一层 `Aligner building` (可以在存储这个类的代码中找到, 直接搜索)
        TODO: 那么这个 `Aligner` 和 这里的 `对齐` 又有什么区别呢?
        normalized affine matrix from 'normalize_pairwise_tfm'
        shape: (B, L, L, 2, 3)

    align_corners:
    """

    # 这里提取了输入数据x的形状信息，以及仿射矩阵的形状信息
    _, C, H, W = x.shape
    B, L = affine_matrix.shape[:2]

    # regroup函数将x和score根据record_len重新分组，使得每个批次中的特征和分数分别分开
    split_x = regroup(x, record_len)
    split_score = regroup(score, record_len)

    batch_node_features = split_x
    out = []

    # iterate each batch
    for b in range(B):
        """
        iterate each batch
        对于每个批次 b，首先获取该批次的车辆数量 N，对应的分数 score 以及仿射矩阵 t_matrix。
        然后，将每个车辆的特征和分数通过 warp_affine_simple 函数进行仿射变换，使其对齐到主车辆（ego vehicle）的坐标系。
        """
        N = record_len[b]
        t_matrix = affine_matrix[b][:N, :N, :, :]
        i = 0  # ego
        feature_in_ego = \
            warp_affine_simple(batch_node_features[b], t_matrix[i, :, :, :], (H, W), align_corners=align_corners)
        scores_in_ego = warp_affine_simple(split_score[b], t_matrix[i, :, :, :], (H, W), align_corners=align_corners)
        """
        对齐后的分数score_in_ego中，将值为0的部分填充为负无穷大，然后应用softmax函数进行归一化。
        接着，将所有NaN值替换为0，确保数值稳定性。
        """
        scores_in_ego.masked_fill_(scores_in_ego == 0, -float('inf'))
        scores_in_ego = torch.softmax(scores_in_ego, dim=0)
        scores_in_ego = torch.where(torch.isnan(scores_in_ego),
                                    torch.zeros_like(scores_in_ego, device=scores_in_ego.device), scores_in_ego)
        """将特征图和归一化后的分数相乘，并沿第0维求和，得到融合后的特征图，存入out列表"""
        out.append(torch.sum(feature_in_ego * scores_in_ego, dim=0))
    out = torch.stack(out)

    return out


class PyramidFusion(ResNetBEVBackbone):
    def __init__(self, model_cfg: Mapping):
        """
        Do not downsample in the first layer.
        """
        super().__init__(model_cfg)
        if model_cfg["resnext"]:
            Bottleneck.expansion = 1
            self.resnet = ResNetModified(Bottleneck, model_cfg['layer_nums'], model_cfg['layer_strides'],
                                         model_cfg['num_filters'], inplanes=model_cfg.get('inplanes', 64),
                                         groups=32, width_per_group=4)
        # TODO: 这个变量有什么深意吗? 为什么要专门打印出来
        self.align_corners = model_cfg.get('align_corners', False)
        # print('Align corners: ', self.align_corners)
        logger.success(f'Align corners: {self.align_corners}')

        # add single supervision head
        for i in range(self.num_levels):
            setattr(self, f"single_head_{i}", nn.Conv2d(model_cfg["num_filters"][i], 1, kernel_size=1))

    def forward_single(self, spatial_features):
        """
        This is used for single agent pass.
        """
        feature_list = self.get_multiscale_feature(spatial_features)
        occ_map_list = []
        for i in range(self.num_levels):
            occ_map = eval(f"self.single_head_{i}")(feature_list[i])
            occ_map_list.append(occ_map)
        final_feature = self.decode_multiscale_feature(feature_list)

        return final_feature, occ_map_list

    def forward_collab(self, spatial_features: Tensor, record_len: list, affine_matrix: Tensor,
                       agent_modality_list=None, cam_crop_info=None):
        """
        spatial_features : torch.tensor
            [sum(record_len), C, H, W]

        record_len : list
            cav num in each sample

        affine_matrix : torch.tensor
            [B, L, L, 2, 3]

        agent_modality_list : list
            len = sum(record_len), modality of each cav

        cam_crop_info : dict
            {'m2':
                {
                    'crop_ratio_W_m2': 0.5,
                    'crop_ratio_H_m2': 0.5,
                }
            }
        """
        # 这一部分应该和 camera 裁切有关, 如果只关注 lidar 的话, 我想这部分是暂时不需要看的
        crop_mask_flag = False
        if cam_crop_info is not None and len(cam_crop_info) > 0:
            crop_mask_flag = True
            cam_modality_set = set(cam_crop_info.keys())
            cam_agent_mask_dict = {}
            for cam_modality in cam_modality_set:
                mask_list = [1 if x == cam_modality else 0 for x in agent_modality_list]
                mask_tensor = torch.tensor(mask_list, dtype=torch.bool)
                cam_agent_mask_dict[cam_modality] = mask_tensor

                # e.g. {m2: [0,0,0,1], m4: [0,1,0,0]}
        # spatial_feature.shape: [12, 64, 128, 256]
        # len(feature_list) = 3, 也就是说, 一行代码, 走了三层 ResNeXt, 将三层特征提取出来了
        feature_list = self.get_multiscale_feature(spatial_features)
        fused_feature_list = []  # 存储着经 ResNeXt 提取出来的与 Foreground Estimator 提取出来特征的融合后特征 len(fused_feature_list) = 3
        occ_map_list = []
        """
        这部分应该就是 Foreground Estimator, 我觉得应该提出一个模块出来
        """
        for i in range(self.num_levels):
            occ_map = eval(f"self.single_head_{i}")(feature_list[i])  # [N, 1, H, W]
            occ_map_list.append(occ_map)
            score = torch.sigmoid(occ_map) + 1e-4

            if crop_mask_flag and not self.training:
                cam_crop_mask = torch.ones_like(occ_map, device=occ_map.device)
                _, _, H, W = cam_crop_mask.shape
                for cam_modality in cam_modality_set:
                    # There may be unstable response values at the edges.
                    crop_H = H / cam_crop_info[cam_modality][f"crop_ratio_H_{cam_modality}"] - 4

                    # There may be unstable response values at the edges.
                    crop_W = W / cam_crop_info[cam_modality][f"crop_ratio_W_{cam_modality}"] - 4

                    start_h = int(H // 2 - crop_H // 2)
                    end_h = int(H // 2 + crop_H // 2)
                    start_w = int(W // 2 - crop_W // 2)
                    end_w = int(W // 2 + crop_W // 2)

                    cam_crop_mask[cam_agent_mask_dict[cam_modality], :, start_h:end_h, start_w:end_w] = 0
                    cam_crop_mask[cam_agent_mask_dict[cam_modality]] = \
                        1 - cam_crop_mask[cam_agent_mask_dict[cam_modality]]

                score = score * cam_crop_mask
            # Foreground Map 融合的部分
            fused_feature_list.append(
                weighted_fuse(feature_list[i], score, record_len, affine_matrix, self.align_corners))
        fused_feature = self.decode_multiscale_feature(fused_feature_list) # fused_feature.shape = [4, 384, 128, 256] TODO: `4`? 三层 + 最后一层融合后的?

        return fused_feature, occ_map_list
