# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, OpenPCDet
# License: TDG-Attribution-NonCommercial-NoDistrib

"""
Transform points to voxels using sparse conv library
"""
import sys

import numpy as np
import torch

from opencood.data_utils.pre_processor.base_preprocessor import BasePreprocessor


class SpVoxelPreprocessor(BasePreprocessor):
    def __init__(self, preprocess_params, train):
        super().__init__(preprocess_params, train)
        self.spconv = 1
        try:
            # spconv v1.x
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
        except ImportError:
            # spconv v2.x
            from cumm import tensorview as tv
            from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
            self.tv = tv
            self.spconv = 2

        self.lidar_range = self.params['cav_lidar_range']
        self.voxel_size = self.params['args']['voxel_size']
        self.max_points_per_voxel = self.params['args']['max_points_per_voxel']

        if train:
            self.max_voxels = self.params['args']['max_voxel_train']
        else:
            self.max_voxels = self.params['args']['max_voxel_test']
        # 调试：[512. 2561 1.], m2: [256. 256.  1.]
        grid_size = (np.array(self.lidar_range[3:6]) - np.array(self.lidar_range[0:3])) / np.array(self.voxel_size)
        self.grid_size = np.round(grid_size).astype(np.int64)

        # use sparse conv library to generate voxel
        if self.spconv == 1:
            self.voxel_generator = VoxelGenerator(voxel_size=self.voxel_size, point_cloud_range=self.lidar_range,
                                                  max_num_points=self.max_points_per_voxel, max_voxels=self.max_voxels)
        else:
            self.voxel_generator = VoxelGenerator(vsize_xyz=self.voxel_size, coors_range_xyz=self.lidar_range,
                                                  max_num_points_per_voxel=self.max_points_per_voxel,
                                                  num_point_features=4, max_num_voxels=self.max_voxels)

    def preprocess(self, pcd_np):
        """

        :param pcd_np: 包含点云数据的 numpy 数组
        :return:
        """
        data_dict = {}
        if self.spconv == 1:
            voxel_output = self.voxel_generator.generate(pcd_np)
        else:
            pcd_tv = self.tv.from_numpy(pcd_np)  # 将点云数据转换为 tensorview
            voxel_output = self.voxel_generator.point_to_voxel(pcd_tv)  # 生成体素

        if isinstance(voxel_output, dict):
            voxels, coordinates, num_points = \
                voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
        else:  # spconv v2.x 版本
            voxels, coordinates, num_points = voxel_output

        if self.spconv == 2:
            voxels = voxels.numpy()
            coordinates = coordinates.numpy()
            num_points = num_points.numpy()

        data_dict['voxel_features'] = voxels
        data_dict['voxel_coords'] = coordinates
        data_dict['voxel_num_points'] = num_points

        return data_dict

    def collate_batch(self, batch):
        """
        Customized pytorch data loader collate function.

        Parameters
        ----------
        batch : list or dict
            List or dictionary.

        Returns
        -------
        processed_batch : dict
            Updated lidar batch.
        """

        if isinstance(batch, list):
            return self.collate_batch_list(batch)
        elif isinstance(batch, dict):
            return self.collate_batch_dict(batch)
        else:
            sys.exit('Batch has too be a list or a dictionarn')

    @staticmethod
    def collate_batch_list(batch):
        """
        Customized pytorch data loader collate function.

        Parameters
        ----------
        batch : list
            List of dictionary. Each dictionary represent a single frame.

        Returns
        -------
        processed_batch : dict
            Updated lidar batch.
        """
        voxel_features = []
        voxel_num_points = []
        voxel_coords = []

        for i in range(len(batch)):
            voxel_features.append(batch[i]['voxel_features'])
            voxel_num_points.append(batch[i]['voxel_num_points'])
            coords = batch[i]['voxel_coords']
            voxel_coords.append(np.pad(coords, ((0, 0), (1, 0)), mode='constant', constant_values=i))

        voxel_num_points = torch.from_numpy(np.concatenate(voxel_num_points))
        voxel_features = torch.from_numpy(np.concatenate(voxel_features))
        voxel_coords = torch.from_numpy(np.concatenate(voxel_coords))

        return {'voxel_features': voxel_features, 'voxel_coords': voxel_coords, 'voxel_num_points': voxel_num_points}

    @staticmethod
    def collate_batch_dict(batch: dict):
        """
        Collate batch if the batch is a dictionary,
        eg: {'voxel_features': [feature1, feature2...., feature n]}
        GPT: 将一个包含点云数据的批次（batch）字典转换为适合 PyTorch 模型输入的格式。
        具体来说，该函数处理的输入是一个字典，每个键都对应一个列表，这些列表包含不同点云帧（frame）的特征、坐标和点的数量。
        该函数将这些列表中的数据合并，并转换为 PyTorch 张量（tensor），从而使数据能够被 PyTorch 模型有效地处理。

        Parameters
        ----------
        batch : dict

        Returns
        -------
        processed_batch : dict
            Updated lidar batch.
        """
        voxel_features = torch.from_numpy(np.concatenate(batch['voxel_features']))
        voxel_num_points = torch.from_numpy(np.concatenate(batch['voxel_num_points']))
        coords = batch['voxel_coords']
        voxel_coords = []

        for i in range(len(coords)):
            """
            对于每一帧的坐标，在前面添加一列常数值 i，用以标识该坐标属于批次中的第 i 帧。
            然后将所有帧的坐标合并成一个 numpy 数组，并将其转换为 PyTorch 张量。
            coords[i] 是一个二维数组，形状为 (num_voxels, 3)，表示该帧的体素坐标，其中 num_voxels 是体素的数量，3 是每个体素坐标的维度（通常是 x, y, z）
            ((0, 0), (1, 0)) 是填充宽度参数: (0, 0): 表示对第一个维度 (行) 不进行填充; (1, 0):表示对第二个维度（列）在前面填充 1 列，在后面不填充
            mode='constant'：填充的模式为常数填充。
            constant_values=i：填充值为常数 i，表示该帧在批次中的索引
            填充后: (num_voxels, 4)
            """
            voxel_coords.append(np.pad(coords[i], ((0, 0), (1, 0)), mode='constant', constant_values=i))
        voxel_coords = torch.from_numpy(np.concatenate(voxel_coords))

        return {'voxel_features': voxel_features, 'voxel_coords': voxel_coords, 'voxel_num_points': voxel_num_points}
