from typing import Callable, Union, List, Optional, Type

import torch
import torch.nn as nn
from torch import Tensor


def conv7x7(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """7x7 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=3 * dilation, groups=groups, bias=False)


def conv5x5(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """5x5 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=2 * dilation, groups=groups, bias=False)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    # TODO: 参数 `padding` 和 `dilation` 为什么设置的值是一样的呢?
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, )  # dilation=dilation


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class MyBasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None,
                 groups: int = 1, base_width: int = 64, dilation: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None, kernel_size: int = 3) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            pass
            # raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        if kernel_size == 3:
            self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
        elif kernel_size == 5:
            self.conv1 = conv5x5(inplanes, planes, stride, dilation=dilation)
        elif kernel_size == 7:
            self.conv1 = conv7x7(inplanes, planes, stride, dilation=dilation)

        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

        if kernel_size == 3:
            self.conv2 = conv3x3(planes, planes, dilation=dilation)  # 修改此处
        elif kernel_size == 5:
            self.conv2 = conv5x5(planes, planes, dilation=dilation)  # 修改此处
        elif kernel_size == 7:
            self.conv2 = conv7x7(planes, planes, dilation=dilation)  # 修改此处

        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class MyResNetModified(nn.Module):
    def __init__(self, block: Type[MyBasicBlock],
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
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.layernum = len(num_filters)
        for i in range(self.layernum):
            self.__setattr__(f"layer{i}",
                             self._make_layer(block, num_filters[i], layers[i], stride=layer_strides[i],
                                              dilate=replace_stride_with_dilation[i],
                                              kernel_size=2 * i + 3))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, MyBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: MyBasicBlock, planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False, kernel_size=3) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, dilation=self.dilation,
                        norm_layer=norm_layer, kernel_size=kernel_size)]
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, groups=self.groups, base_width=self.base_width,
                      dilation=self.dilation, norm_layer=norm_layer, kernel_size=kernel_size))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor, return_interm: bool = True) -> Union[List[Tensor], Tensor]:
        interm_features = []
        for i in range(self.layernum):
            x = eval(f"self.layer{i}")(x)
            interm_features.append(x)
        if return_interm:
            return interm_features
        return x

    def forward(self, x: Tensor) -> Union[List[Tensor], Tensor]:
        return self._forward_impl(x)


if __name__ == "__main__":
    model = MyResNetModified(MyBasicBlock, layers=[1, 1, 1], layer_strides=[1, 2, 3],
                             num_filters=[64, 64, 64], groups=1, width_per_group=64)
    input = torch.randn(4, 64, 200, 704)

    output = model(input)
    from icecream import ic

    for out in output:
        ic(out.shape)
    ic(model)
