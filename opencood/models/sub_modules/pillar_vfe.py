"""
Pillar VFE, credits to OpenPCDet.
VFE: Voxel Feature Encoding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PFNLayer(nn.Module):
    """
    PFN: Pillar Feature Network，这是在点云处理领域中使用的一种网络结构，特别是在PointPillars模型中。

    在PointPillars模型中，点云数据首先被转换成柱状体素（pillar），然后通过PFN提取这些柱状体素的特征。
    具体来说，PFN负责将每个pillar中的点的特征进行编码和聚合，生成代表整个pillar的特征向量。
    """

    def __init__(self, in_channels: int, out_channels: int, use_norm=True, last_layer=False):
        """

        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            use_norm: 是否使用批量归一化
            last_layer: 是否是最后一层
        """
        super().__init__()

        self.last_vfe = last_layer
        self.use_norm = use_norm

        if not self.last_vfe:  # 如果是最后一层则输出通道减半
            out_channels = out_channels // 2

        self.linear = nn.Linear(in_channels, out_channels, bias=not self.use_norm)
        if self.use_norm:
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)

        self.part = 50000

    def forward(self, inputs: Tensor) -> Tensor:
        # 如果输入数据的第一个维度（通常是批量大小）超过 `self.part`，将数据分块处理以防止内存溢出。
        # 使用线性层 self.linear 对每个块进行处理，并将结果拼接起来
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(
                inputs[num_part * self.part:(num_part + 1) * self.part])
                for num_part in range(num_parts + 1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)

        # 禁用 cuDNN 加速以确保 BatchNorm1d 在高维数据上的正确性 TODO: 这是为什么?
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
        torch.backends.cudnn.enabled = True

        x = F.relu(x)

        # 通过取最大值进行特征聚合（沿第1维，即每个特征的最大值）。
        # 如果是最后一层，则返回聚合后的特征。
        # 否则，将聚合后的特征重复，并与原始特征拼接，形成新的输出特征
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class PillarVFE(nn.Module):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range):
        super().__init__()

        # 默认认为函数参数不可变, 所以要使用首先需要创建一个同名的变量 (加一个下划线如何?)
        _num_point_features = num_point_features

        self.use_absolute_xyz = model_cfg['use_absolute_xyz']
        self.with_distance = model_cfg['with_distance']

        _num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            _num_point_features += 1

        self.num_filters = model_cfg['num_filters']
        num_filters_len = len(self.num_filters)
        assert num_filters_len > 0
        self.num_filters = [_num_point_features] + list(self.num_filters)

        pfn_layers = []

        for i in range(num_filters_len - 1):
            in_filters = self.num_filters[i]
            out_filters = self.num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, model_cfg['use_norm'], last_layer=(i >= num_filters_len - 2)))
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    @staticmethod
    def get_paddings_indicator(actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict):
        """
        encoding voxel feature using point-pillar method
        TODO: 这代码是真看不懂......
        Args:
            batch_dict:
                voxel_features: [M, 32, 4]
                voxel_num_points: [M,]
                voxel_coords: [M, 4]
        Returns:
            features: [M,64], after PFN
        """
        voxel_features, voxel_num_points, coords = (
            batch_dict['voxel_features'], batch_dict['voxel_num_points'], batch_dict['voxel_coords'])

        points_mean = (voxel_features[:, :, :3].sum(dim=1, keepdim=True) /
                       voxel_num_points.type_as(voxel_features).view(-1, 1, 1))

        f_cluster = voxel_features[:, :, :3] - points_mean

        f_center = torch.zeros_like(voxel_features[:, :, :3])
        f_center[:, :, 0] = voxel_features[:, :, 0] - (
                coords[:, 3].to(voxel_features.dtype).unsqueeze(
                    1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (
                coords[:, 2].to(voxel_features.dtype).unsqueeze(
                    1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (
                coords[:, 1].to(voxel_features.dtype).unsqueeze(
                    1) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]

        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)

        voxel_count = features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        features *= mask
        for pfn in self.pfn_layers:
            features = pfn(features)
        features = features.squeeze()
        batch_dict['pillar_features'] = features

        return batch_dict
