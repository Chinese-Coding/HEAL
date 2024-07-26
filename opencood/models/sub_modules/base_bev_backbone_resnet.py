# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib
from typing import Mapping

import numpy as np
import torch
import torch.nn as nn

from opencood.models.sub_modules.my_resblock import MyResNetModified, MyBasicBlock
from opencood.models.sub_modules.resblock import ResNetModified, BasicBlock

DEBUG = False


class ResNetBEVBackbone(nn.Module):
    """
    TODO: 这个类应该好好看看, 真的没看明白......
    """

    def __init__(self, model_cfg: Mapping):
        super().__init__()
        """
        一些简单的参数配置
        """
        # 如果在模型参数中定义了层数, 那么就应该校验一下层参数列表的长度是否等于层数
        if 'layer_nums' in model_cfg:
            assert len(model_cfg['layer_nums']) == len(model_cfg['layer_strides']) == len(model_cfg['num_filters'])

            layer_nums = model_cfg['layer_nums']
            layer_strides = model_cfg['layer_strides']
            num_filters = model_cfg['num_filters']
        else:
            layer_nums = layer_strides = num_filters = []

        if 'upsample_strides' in model_cfg:
            assert len(model_cfg['upsample_strides']) == len(model_cfg['num_upsample_filter'])

            num_upsample_filters = model_cfg['num_upsample_filter']
            upsample_strides = model_cfg['upsample_strides']
        else:
            upsample_strides = num_upsample_filters = []

        """
        开始定义网络
        """
        self.my_resnet = MyResNetModified(MyBasicBlock, model_cfg['layer_nums'], model_cfg['layer_strides'],
                                        model_cfg['num_filters'], inplanes=model_cfg.get('inplanes', 64))

        self.resnet = ResNetModified(BasicBlock, layer_nums, layer_strides, num_filters,
                                     inplanes=model_cfg.get('inplanes', 64))

        # 创建反卷积层 TODO: 什么是反卷积层...... 又是一个复杂的概念......
        # GPT: 反卷积层用于将低分辨率的特征图转换为高分辨率，这在目标检测、语义分割等任务中很常见
        # num_levels = len(layer_nums)
        self.num_levels = len(layer_nums)
        self.deblocks = nn.ModuleList()  # deblocks = deconvolution blocks (反卷积块或上采样块)

        for idx in range(self.num_levels):
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                # 将 if 语句里面的公共部分提取一下
                batch_norm2d, re_lu = nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01), nn.ReLU()

                if stride >= 1:  # 如果 stride 大于或等于 1，使用 nn.ConvTranspose2d（反卷积）来上采样
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(num_filters[idx], num_upsample_filters[idx], upsample_strides[idx],
                                           stride=upsample_strides[idx], bias=False),
                        batch_norm2d, re_lu))
                else:  # 如果 stride 小于 1，使用 nn.Conv2d 来进行下采样
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(num_filters[idx], num_upsample_filters[idx], stride, stride=stride, bias=False),
                        batch_norm2d, re_lu))

        # GPT: 计算上采样滤波器的总通道数 c_in，如果上采样步长的数量多于层数，则添加额外的反卷积层。
        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > self.num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01), nn.ReLU()))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        spatial_features = data_dict['spatial_features']

        # 将 spatial_features 输入到 resnet 中，得到一个特征图的元组 x。每个元素对应 ResNet 中不同层的输出特征图
        # TODO: 为什么这里是元组呢?
        # x = self.resnet(spatial_features)  # tuple of features
        x = self.my_resnet(spatial_features)
        ups = []  # 初始化上采样结果列表? TODO: ups 是什么的缩写呢?

        for i in range(self.num_levels):
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x[i]))
            else:
                ups.append(x[i])

        # 将 ups 中的特征图拼接起来，形成最终的特征图 x
        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > self.num_levels:
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x
        return data_dict

    # these two functions are seperated for multiscale intermediate fusion
    def get_multiscale_feature(self, spatial_features):
        """
        before multiscale intermediate fusion
        """
        x = self.resnet(spatial_features)  # tuple of features
        return x

    def decode_multiscale_feature(self, x):
        """
        after multiscale interemediate fusion
        """
        ups = []
        for i in range(self.num_levels):
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x[i]))
            else:
                ups.append(x[i])
        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > self.num_levels:
            x = self.deblocks[-1](x)
        return x

    def get_layer_i_feature(self, spatial_features, layer_i):
        """
        before multiscale intermediate fusion
        """
        return eval(f"self.resnet.layer{layer_i}")(spatial_features)  # tuple of features
