# -*- coding: utf-8 -*-
"""
Class for data augmentation
"""
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

from functools import partial

from opencood.data_utils.augmentor import augment_utils


class DataAugmentor:
    """
    Augmentor: 增强器
    Data Augmentor.
    这段代码定义了一个用于数据增强（data augmentation）的类 DataAugmentor。数据增强是机器学习和计算机视觉中常用的技术，
    通过对训练数据进行各种变换来增加数据集的多样性，从而提高模型的泛化能力

    Parameters
    ----------
    augment_config : list
        A list of augmentation (增强) configuration.

    Attributes
    ----------
    data_augmentor_queue : list
        The list of data augmented functions.
    """

    def __init__(self, augment_config, train=True):
        self.data_augmentor_queue = []
        self.train = train

        for cur_cfg in augment_config:
            cur_augmentor = getattr(self, cur_cfg['NAME'])(config=cur_cfg)
            self.data_augmentor_queue.append(cur_augmentor)

    def random_world_flip(self, data_dict=None, config=None):
        """
        随机翻转数据，支持沿 x 轴或 y 轴翻转。
        具体操作包括从 data_dict 中提取 gt_boxes (目标框), gt_mask (目标掩码)
        和 points (点云数据), 并对有效的目标框和点云数据进行翻转。

        :param data_dict:
        :param config:
        :return:
        """
        if data_dict is None:
            return partial(self.random_world_flip, config=config)

        gt_boxes, gt_mask, points = data_dict['object_bbx_center'], data_dict['object_bbx_mask'], data_dict['lidar_np']
        gt_boxes_valid = gt_boxes[gt_mask == 1]

        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['x', 'y']
            gt_boxes_valid, points = getattr(augment_utils, 'random_flip_along_%s' % cur_axis)(gt_boxes_valid, points, )

        gt_boxes[:gt_boxes_valid.shape[0], :] = gt_boxes_valid

        data_dict['object_bbx_center'] = gt_boxes
        data_dict['object_bbx_mask'] = gt_mask
        data_dict['lidar_np'] = points

        return data_dict

    def random_world_rotation(self, data_dict=None, config=None):
        """
        随机旋转数据。
        :param data_dict:
        :param config:
        :return:
        """
        if data_dict is None:
            """
            partial 函数来自于 Python 的 functools 模块，它用于创建一个新的函数，该函数是原函数的一部分参数被固定的结果。
            简单来说，partial 函数可以用来预先填充原函数的一些参数，并返回一个新的函数，该函数只需要提供剩下的参数即可
            python 灵活性的又一强大体现
            """
            return partial(self.random_world_rotation, config=config)

        rot_range = config['WORLD_ROT_ANGLE']
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]

        gt_boxes, gt_mask, points = data_dict['object_bbx_center'], data_dict['object_bbx_mask'], data_dict['lidar_np']
        gt_boxes_valid = gt_boxes[gt_mask == 1]
        gt_boxes_valid, points = augment_utils.global_rotation(gt_boxes_valid, points, rot_range=rot_range)
        gt_boxes[:gt_boxes_valid.shape[0], :] = gt_boxes_valid

        data_dict['object_bbx_center'] = gt_boxes
        data_dict['object_bbx_mask'] = gt_mask
        data_dict['lidar_np'] = points

        return data_dict

    def random_world_scaling(self, data_dict=None, config=None):
        """
        随机缩放数据。
        具体操作包括从 data_dict 中提取 gt_boxes、gt_mask 和 points，并对有效的目标框和点云数据进行缩放。
        :param data_dict:
        :param config:
        :return:
        """
        if data_dict is None:
            return partial(self.random_world_scaling, config=config)

        gt_boxes, gt_mask, points = data_dict['object_bbx_center'], data_dict['object_bbx_mask'], data_dict['lidar_np']
        gt_boxes_valid = gt_boxes[gt_mask == 1]

        gt_boxes_valid, points = augment_utils.global_scaling(gt_boxes_valid, points, config['WORLD_SCALE_RANGE'])
        gt_boxes[:gt_boxes_valid.shape[0], :] = gt_boxes_valid

        data_dict['object_bbx_center'] = gt_boxes
        data_dict['object_bbx_mask'] = gt_mask
        data_dict['lidar_np'] = points

        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7) [x, y, z, dx, dy, dz, heading]
                gt_names: optional, (N), string
                ...

        Returns:
        """
        if self.train:
            for cur_augmentor in self.data_augmentor_queue:
                data_dict = cur_augmentor(data_dict=data_dict)

        return data_dict
