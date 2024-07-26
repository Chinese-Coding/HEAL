# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from opencood.utils import pcd_utils


# @dataclass
# class PreprocessorParams:
#     """
#     专门用于存储, preprocessor 各个类所需的参数
#     TODO: 目前仅支持 SpVoxelPreprocessor
#     """
#     lidar_range: List[float]
#     voxel_size: List[float]
#     max_points_per_voxel: int
#     max_voxels: int
#     sample_num: int
#     cav_lidar_range: List[float]
#
#     def __init__(self, preprocessor_params: Mapping, train=True):
#         self.lidar_range = preprocessor_params['cav_lidar_range']
#         self.voxel_size = preprocessor_params['args']['voxel_size']
#         self.max_points_per_voxel = preprocessor_params['args']['max_points_per_voxel']
#         temp = f'max_voxels_train' if train else 'max_voxels_test'
#         self.max_voxels = preprocessor_params['args'][temp]
#         self.sample_num = preprocessor_params['args']['sample_num']
#         self.cav_lidar_range= preprocessor_params['cav_lidar_range']
#

class BasePreprocessor:
    """
    Basic Lidar pre-processor.

    Parameters
    ----------
    preprocess_params : dict
        The dictionary containing all parameters of the preprocessing.

    train : bool
        Train or test mode.
    """

    def __init__(self, params, train):
        self.params = params
        self.train = train

    def preprocess(self, pcd_np):
        """
        Preprocess the lidar points by simple sampling.

        Parameters
        ----------
        pcd_np : np.ndarray
            The raw lidar.

        Returns
        -------
        data_dict : the output dictionary.
        """
        data_dict = {}
        sample_num = self.params['args']['sample_num']

        pcd_np = pcd_utils.downsample_lidar(pcd_np, sample_num)
        data_dict['downsample_lidar'] = pcd_np

        return data_dict

    def project_points_to_bev_map(self, points, ratio=0.1):
        """
        Project points to BEV occupancy map with default ratio=0.1.

        Parameters
        ----------
        points : np.ndarray
            (N, 3) / (N, 4)

        ratio : float
            Discretization parameters. Default is 0.1.

        Returns
        -------
        bev_map : np.ndarray
            BEV occupancy map including projected points with shape
            (img_row, img_col).

        """
        L1, W1, H1, L2, W2, H2 = self.params["cav_lidar_range"]
        img_row = int((L2 - L1) / ratio)
        img_col = int((W2 - W1) / ratio)
        bev_map = np.zeros((img_row, img_col))
        bev_origin = np.array([L1, W1, H1]).reshape(1, -1)
        # (N, 3)
        indices = ((points[:, :3] - bev_origin) / ratio).astype(int)
        mask = np.logical_and(indices[:, 0] > 0, indices[:, 0] < img_row)
        mask = np.logical_and(mask, np.logical_and(indices[:, 1] > 0, indices[:, 1] < img_col))
        indices = indices[mask, :]
        bev_map[indices[:, 0], indices[:, 1]] = 1
        return bev_map
