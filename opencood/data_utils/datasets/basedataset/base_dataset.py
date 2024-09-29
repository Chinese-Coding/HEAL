from abc import abstractmethod
from collections import OrderedDict
from typing import Mapping

import numpy as np
from torch.utils.data import Dataset
import abc

from opencood.data_utils.augmentor.data_augmentor import DataAugmentor
from opencood.data_utils.post_processor import build_postprocessor
from opencood.data_utils.pre_processor import build_preprocessor


class BaseDataset(Dataset):
    __metaclass__ = abc.ABCMeta

    def __init__(self, params: Mapping, visualize: bool, train=True):
        self.params = params
        self.visualize = visualize
        self.train = train

        self.pre_processor = build_preprocessor(params["preprocess"], train)
        self.post_processor = build_postprocessor(params["postprocess"], train)

        if 'data_augment' in params:  # late and early
            self.data_augmentor = DataAugmentor(params['data_augment'], train)
        else:  # intermediate
            self.data_augmentor = None

        self.max_cav: int

        self.load_lidar_file = True if 'lidar' in params['input_source'] or self.visualize else False
        self.load_camera_file = True if 'camera' in params['input_source'] else False
        self.load_depth_file = True if 'depth' in params['input_source'] else False

        self.label_type = params['label_type']
        self.generate_object_center = self.generate_object_center_lidar if self.label_type == "lidar" \
            else self.generate_object_center_camera

        # TODO: 如果配置文件里没有则直接在这里加, 这可不是一个好习惯
        if "noise_setting" not in self.params:
            self.params['noise_setting'] = OrderedDict()
            self.params['noise_setting']['add_noise'] = False

        if self.load_camera_file:
            self.data_aug_conf = params["fusion"]["args"]["data_aug_conf"]

    @abstractmethod
    def retrieve_base_data(self, idx):
        raise NotImplementedError

    def augment(self, lidar_np: np.ndarray, object_bbx_center: np.ndarray, object_bbx_mask: np.ndarray):
        """
        Given the raw point cloud, augment by flipping and rotation.

        Parameters
        ----------
        lidar_np : np.ndarray
            (n, 4) shape

        object_bbx_center : np.ndarray
            (n, 7) shape to represent bbx's x, y, z, h, w, l, yaw
        object_bbx_mask : np.ndarray
            Indicate which elements in object_bbx_center are padded.
        """
        tmp_dict = {'lidar_np': lidar_np, 'object_bbx_center': object_bbx_center, 'object_bbx_mask': object_bbx_mask}
        tmp_dict = self.data_augmentor.forward(tmp_dict)

        lidar_np = tmp_dict['lidar_np']
        object_bbx_center = tmp_dict['object_bbx_center']
        object_bbx_mask = tmp_dict['object_bbx_mask']

        return lidar_np, object_bbx_center, object_bbx_mask

    @abstractmethod
    def reinitialize(self):
        raise NotImplementedError

    @abstractmethod
    def generate_object_center_lidar(self, cav_contents, reference_lidar_pose):
        raise NotImplementedError

    @abstractmethod
    def generate_object_center_camera(self, cav_contents, reference_lidar_pose):
        raise NotImplementedError

    @abstractmethod
    def get_ext_int(self, params, camera_id):
        raise NotImplementedError
