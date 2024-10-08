from typing import Type, Callable, Union, List, Optional

import torch
import torch.nn as nn
from torch import Tensor


def conv7x7(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """7x7 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=dilation, groups=groups, bias=False)


def conv5x5(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """5x5 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=dilation, groups=groups, bias=False)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    # TODO: 参数 `padding` 和 `dilation` 为什么设置的值是一样的呢?
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, )  # dilation=dilation


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None,
                 groups: int = 1, base_width: int = 64, dilation: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        """

        Args:
            inplanes:
            planes:
            stride:
            downsample:
            groups:
            base_width:
            dilation: 卷积扩张率, 默认为 1
                    在卷积神经网络中，dilation（扩张率或膨胀率）是一个卷积层的参数，
                    用于控制卷积核（滤波器）在输入特征图上应用时的扩展方式。
                    具体来说，dilation 指定了卷积核中元素之间的间距，从而影响卷积操作的感受野。
            norm_layer:
        """
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride  # TODO: `stride` 变量并没有在 forward 中用到, 为什么还要定义呢 (也许在类外能用到?)

    def forward(self, x: Tensor) -> Tensor:
        identity = x  # TODO: 后面直接加 x 就可以, 定义一个新变量 `identity` 是否冗余? 还是说普遍都是这么写的?

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # TODO: 残差网络的精髓: 残差连接.
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4  # original 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None,
                 groups: int = 1, base_width: int = 64, dilation: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetModified(nn.Module):

    def __init__(self, block: Type[Union[BasicBlock, Bottleneck]],
                 layers: List[int],  # number of block in one layer
                 layer_strides: List[int],  # stride after one layer
                 num_filters: List[int],  # feature dim
                 zero_init_residual: bool = False, groups: int = 1, width_per_group: int = 64,
                 replace_stride_with_dilation: Optional[List[bool]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 inplanes=64) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = inplanes
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        # 创建每一层
        self.layernum = len(num_filters)
        for i in range(self.layernum):
            self.__setattr__(f"layer{i}", self._make_layer(block, num_filters[i], layers[i], stride=layer_strides[i]))

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        # TODO: 这部分已经看不懂了
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        # if stride != 1, the first block will downsample the feature map
        # plane is the feature dim
        # if Bottleneck, then the output dim is planes * block.expansion(4)
        layers = [block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation,
                        norm_layer)]
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation,
                      norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor, return_interm=True) -> Union[List[Tensor], Tensor]:
        # See note [TorchScript super()]
        interm_features = []
        for i in range(self.layernum):
            # 高级, `__init__` 函数中通过 `__setattr__` 函数定义了一系列层这些层的名字是 `layer0`, `layer1`, ...
            # 如果使用变量来定义的话, 会写很多重复的代码. 这里使用使用 eval 将字符串转换成可执行的 python 代码
            x = eval(f"self.layer{i}")(x)
            interm_features.append(x)
        # 选择性地返回每层的中间特征
        if return_interm:
            return interm_features
        return x

    def forward(self, x: Tensor) -> Union[List[Tensor], Tensor]:
        return self._forward_impl(x)


if __name__ == "__main__":
    Bottleneck.expansion = 1
    model = ResNetModified(Bottleneck, layers=[3, 4, 5], layer_strides=[1, 2, 2],
                           num_filters=[64, 128, 256], groups=32, width_per_group=4)
    input = torch.randn(4, 64, 200, 704)
    output = model(input)
    from icecream import ic

    for out in output:
        ic(out.shape)
    ic(model)
