import numpy as np
from torch import nn

from opencood.models.sub_modules.cia_ssd_utils import SSFA, Head
from opencood.models.sub_modules.height_compression import HeightCompression
from opencood.models.sub_modules.mean_vfe import MeanVFE
from opencood.models.sub_modules.sparse_backbone_3d import VoxelBackBone8x


class CIASSD(nn.Module):
    def __init__(self, args):
        super(CIASSD, self).__init__()
        lidar_range = np.array(args['lidar_range'])
        grid_size = np.round((lidar_range[3:6] - lidar_range[:3]) /
                             np.array(args['voxel_size'])).astype(np.int64)
        self.vfe = MeanVFE(args['mean_vfe'], args['mean_vfe']['num_point_features'])
        self.spconv_block = VoxelBackBone8x(args['spconv'],
                                            input_channels=args['spconv']['num_features_in'],
                                            grid_size=grid_size)
        self.map_to_bev = HeightCompression(args['map2bev'])
        self.ssfa = SSFA(args['ssfa'])
        self.head = Head(**args['head'])

    def forward(self, batch_dict):
        voxel_features = batch_dict['processed_lidar']['voxel_features']
        voxel_coords = batch_dict['processed_lidar']['voxel_coords']
        voxel_num_points = batch_dict['processed_lidar']['voxel_num_points']

        # save memory
        batch_dict.pop('processed_lidar')
        batch_dict.update({'voxel_features': voxel_features,
                           'voxel_coords': voxel_coords,
                           'voxel_num_points': voxel_num_points})

        batch_dict['batch_size'] = batch_dict['object_bbx_center'].shape[0]

        batch_dict = self.vfe(batch_dict)
        batch_dict = self.spconv_block(batch_dict)
        batch_dict = self.map_to_bev(batch_dict)
        out = self.ssfa(batch_dict['spatial_features'])
        out = self.head(out)
        batch_dict['preds_dict_stage1'] = out

        return batch_dict



if __name__=="__main__":
    model = SSFA(None)
    print(model)