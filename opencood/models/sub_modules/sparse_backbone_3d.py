from functools import partial
from termcolor import colored
import torch.nn as nn

try:  # spconv1
    from spconv import SparseSequential, SubMConv3d, SparseConv3d, SparseInverseConv3d, SparseConvTensor
except ImportError:  # spconv2
    from spconv.pytorch import SparseSequential, SubMConv3d, SparseConv3d, SparseInverseConv3d, SparseConvTensor


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):
    if conv_type == 'subm':
        conv = SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                            bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = SparseSequential(conv, norm_fn(out_channels), nn.ReLU())

    return m


class VoxelBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size):
        super().__init__()
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        if input_channels == 64:
            print(f"{colored('[Warning]', 'red', attrs=['bold'])}", 
                  f"{colored('In this checkpoint and configuration yaml (typically provided by the author), SECOND model has wrong `encoder_args`-`spconv`-`num_features_in`.', 'red')}\n",
                  f"{colored('It is supposed to be 4, but is provided with 64.', 'red')}\n",
                  f"{colored('Though you can still run the model due to no sanity check in spconv 1.2.1 and get reasonable performance,', 'red')}",
                  f"{colored('it is not a correct convolution. See discussion in HEAL issue 20. ', 'red')}")

        self.conv_input = SparseSequential(
            SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16), nn.ReLU())
        block = post_act_block

        self.conv1 = SparseSequential(block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'))

        self.conv2 = SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        if 'num_features_out' in model_cfg:
            self.num_point_features = model_cfg['num_features_out']
        else:
            self.num_point_features = 128
        self.conv_out = SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            SparseConv3d(64, self.num_point_features, (3, 1, 1), stride=(2, 1, 1),
                         padding=last_pad, bias=False, indice_key='spconv_down2'),
            norm_fn(self.num_point_features),
            nn.ReLU(),
        )

        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64
        }

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C) # TODO: C 的含义是什么?
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = SparseConvTensor(features=voxel_features, indices=voxel_coords.int(),
                                           spatial_shape=self.sparse_shape, batch_size=batch_size)
        # TODO: 这里有 bug
        # print("Input shape:", input_sp_tensor.features.shape)
        # print(self.conv_input)
        x = self.conv_input(input_sp_tensor)
        # print("After conv_input shape:", x.features.shape)
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict
