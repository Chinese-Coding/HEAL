"""
Pillar VFE, credits to OpenPCDet.
VFE: Voxel Feature Encoding
"""
from typing import Mapping, List

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
            #  x 的维度由（M, 32, 10）升维成了（M, 32, 64）
            x = self.linear(inputs)

        # 禁用 cuDNN 加速以确保 BatchNorm1d 在高维数据上的正确性 TODO: 这是为什么?
        torch.backends.cudnn.enabled = False
        """
        BatchNorm1d: (M, 64, 32) -> (M, 32, 64)（pillars, num_point, channel）-> (pillars, channel, num_points)
        这里之所以变换维度，是因为 BatchNorm1d 在通道维度上进行, 对于图像来说默认模式为 [N, C, H * W], 通道在第二个维度上
        """
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
        torch.backends.cudnn.enabled = True

        x = F.relu(x)

        # 完成 pointnet的 最大池化操作，找出每个 pillar 中最能代表该 pillar 的点
        x_max = torch.max(x, dim=1, keepdim=True)[0]  # x_max.shape: (M, 1, 64)

        # 如果是最后一层，则返回聚合后的特征。
        # 否则，将聚合后的特征重复，并与原始特征拼接，形成新的输出特征
        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class PillarVFE(nn.Module):
    def __init__(self, model_cfg: Mapping, num_point_features: int, voxel_size: List, point_cloud_range: List):
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
        num_filters_len = len(self.num_filters)  # 更新 `self.num_filters` 后记得更新其长度, 一晚上都在搞这个问题, damn!
        pfn_layers = []

        for i in range(num_filters_len - 1):
            in_filters = self.num_filters[i]
            out_filters = self.num_filters[i + 1]
            pfn_layers.append(PFNLayer(in_filters, out_filters, model_cfg['use_norm'], (i >= num_filters_len - 2)))
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
    def get_paddings_indicator(actual_num: Tensor, max_num: int, axis: int = 0) -> Tensor:
        """
        计算 padding 的指示
        当 actual_num 的数量小于 max_num, 将 actual_num 填充为 32 (有数据的地方填 True, 没有的则填 False),
        如果大于或等于, 则截断 (全为 True, 但是大小为 32)

        :param actual_num: 每个voxel实际点的数量 (M,)
        :param max_num: voxel最大点的数量 (32,)
        :param axis:
        :return: 表明一个pillar中哪些是真实数据，哪些是填充的0数据
        """
        actual_num = torch.unsqueeze(actual_num, axis + 1)  # 扩展一个维度，使变为 (M, 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, voxel_features: Tensor, voxel_num_points: Tensor, coords: Tensor):
        # def forward(self, batch_dict: MutableMapping[AnyStr, Tensor]):
        """
        encoding voxel feature using point-pillar method
        TODO: 这代码是真看不懂......
        Args:
            batch_dict:
                voxel_features: [M, 32, 4]
                voxel_num_points: [M,]
                voxel_coords: [M, 4]
        Returns:
        features: [M, 64], after
        PFN
        """
        # voxel_features, voxel_num_points, coords = batch_dict['voxel_features'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']

        # 根据代码注释检查一下 Tensor 的形状
        M = voxel_features.shape[0]
        assert voxel_features.shape == (M, 32, 4), f"voxel_features shape is incorrect: {voxel_features.shape}"
        assert voxel_num_points.shape == (M,), f"voxel_num_points shape is incorrect: {voxel_num_points.shape}"
        assert coords.shape == (M, 4), f"coords shape is incorrect: {coords.shape}"

        # 计算每个体素中点的均值
        # TODO: `voxel_features` 也许早在 DataLoader 的时候就已经划分好了
        # 求每个 pillar 中所有点云的和 (M, 32, 3) -> (M, 1, 3) 设置 `keepdim=True` 的，则保留原来的维度信息
        # 然后在使用求和信息除以每个点云中有多少个点来求每个 pillar 中所有点云的平均值 points_mean shape：(M, 1, 3)
        points_mean = (voxel_features[:, :, :3].sum(dim=1, keepdim=True) /
                       voxel_num_points.type_as(voxel_features).view(-1, 1, 1))

        # 每个点云数据减去该点对应pillar的平均值得到差值 xc,yc,zc
        f_cluster = voxel_features[:, :, :3] - points_mean

        # 计算每个点相对于体素网格中心的相对位置 TODO: 和 `f_cluster` 又有什么区别吗?
        """
        `f_center` 和 `f_cluster` 的区别(GPT回答的)
        * `f_cluster` 表示每个点相对于该体素内所有点的均值的相对位置。
          它计算每个点与所在体素中点的质心（centroid）的差值。这可以帮助网络了解点在体素内的局部分布。
        * `f_center` 表示每个点相对于该体素网格中心的相对位置。
          它计算每个点与其所在体素的中心点的差值。这可以帮助网络理解点在体素网格中的空间分布。
        """
        # 创建每个点云到该pillar的坐标中心点偏移量空数据 xp, yp, zp
        f_center = torch.zeros_like(voxel_features[:, :, :3])

        """
        coords 是每个网格点的坐标，即[432, 496, 1]，需要乘以每个 pillar 的长宽得到点云数据中实际的长宽（单位米）
        同时为了获得每个 pillar 的中心点坐标，还需要加上每个 pillar 长宽的一半得到中心点坐标
        每个点的 x、y、z 减去对应pillar的坐标中心点，得到每个点到该点中心点的偏移量
        """
        f_center[:, :, 0] = voxel_features[:, :, 0] - (
                coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (
                coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (
                coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]

        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)

        # 将特征在最后一维度拼接 得到维度为（M，32, 10）的张量
        features = torch.cat(features, dim=-1)

        voxel_count = features.shape[1]
        """
        由于在生成每个pillar中，不满足最大32个点的pillar会存在由0填充的数据，而刚才上面的计算中，会导致这些
        由0填充的数据在计算出现xc,yc,zc和xp,yp,zp出现数值，所以需要将这个被填充的数据的这些数值清0,
        因此使用get_paddings_indicator计算features中哪些是需要被保留真实数据和需要被置0的填充数据
        """
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)  # mask.shape = [M, 32]
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)  # mask.shape = [M, 32, 1]
        features *= mask
        for pfn in self.pfn_layers:
            features = pfn(features)
        features = features.squeeze()  # (M, 64), 每个 pillar 抽象出一个 64 维特征
        return features

        # batch_dict['pillar_features'] = features
        # return batch_dict


if __name__ == '__main__':
    actual_num = torch.tensor([35])
    max_num = 32
    paddings_indicator = PillarVFE.get_paddings_indicator(actual_num, max_num)
    print(paddings_indicator)
