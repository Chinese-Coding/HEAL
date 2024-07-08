# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import json
import logging
import os
import random
from collections import OrderedDict

import cv2
import h5py
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from opencood.logger import get_logger
import opencood.utils.pcd_utils as pcd_utils
from opencood.data_utils.augmentor.data_augmentor import DataAugmentor
from opencood.data_utils.post_processor import build_postprocessor
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils.camera_utils import load_camera_data
from opencood.utils.transformation_utils import x1_to_x2

logger = get_logger()


class OPV2VBaseDataset(Dataset):
    def __init__(self, params, visualize, train=True):
        self.params = params
        self.visualize = visualize
        self.train = train
        self.use_hdf5 = True

        self.pre_processor = build_preprocessor(params["preprocess"], train)
        self.post_processor = build_postprocessor(params["postprocess"], train)
        # late and early
        self.data_augmentor = DataAugmentor(params['data_augment'], train) if 'data_augment' in params else None

        root_dir = params['root_dir'] if self.train else params['validate_dir']
        # print("Dataset dir:", root_dir)
        logger.success(f'Dataset dir: {root_dir}')

        self.max_cav = 5 if 'train_params' not in params or 'max_cav' not in params['train_params'] \
            else params['train_params']['max_cav']

        self.load_lidar_file = True if 'lidar' in params['input_source'] or self.visualize else False
        self.load_camera_file = True if 'camera' in params['input_source'] else False
        self.load_depth_file = True if 'depth' in params['input_source'] else False

        # 'lidar' or 'camera'
        self.generate_object_center = self.generate_object_center_lidar if params['label_type'] == "lidar" \
            else self.generate_object_center_camera
        # will it follows 'self.generate_object_center' when 'self.generate_object_center' change?
        self.generate_object_center_single = self.generate_object_center

        if self.load_camera_file:
            self.data_aug_conf = params["fusion"]["args"]["data_aug_conf"]

        # by default, we load lidar, camera and metadata. But users may define additional inputs/tasks
        self.add_data_extension = params['add_data_extension'] if 'add_data_extension' in params else []

        if "noise_setting" not in self.params:
            self.params['noise_setting'] = OrderedDict()
            self.params['noise_setting']['add_noise'] = False

        # first load all paths of different scenarios
        self.scenario_folders = sorted([os.path.join(root_dir, x)
                                        for x in os.listdir(root_dir) if
                                        os.path.isdir(os.path.join(root_dir, x))])

        self.logger = logging.getLogger(__name__)
        # 注释掉该函数, 改为由子类调用.
        # self.reinitialize()

    def reinitialize(self):
        # Structure: {scenario_id : {cav_1 : {timestamp1 : {yaml: path,
        # lidar: path, cameras:list of path}}}}
        self.scenario_database = OrderedDict()
        self.len_record = []

        # loop over all scenarios
        for (i, scenario_folder) in enumerate(self.scenario_folders):
            self.scenario_database.update({i: OrderedDict()})

            # at least 1 cav should show up
            if self.train:
                # 读入某一时间戳下的所有文件夹的文件, 并将其命名为 cav_list.
                # 例如: 读入 `OPV2V/train/2021_08_16_22_26_54` 里面的文件, 得到的列表为: `['641', '650', '659']`
                cav_list = [x for x in os.listdir(scenario_folder)
                            if os.path.isdir(
                        os.path.join(scenario_folder, x))]
                # cav_list = sorted(cav_list)
                random.shuffle(cav_list)  # 随机打乱车的 id 号
            else:
                cav_list = sorted([x for x in os.listdir(scenario_folder)
                                   if os.path.isdir(
                        os.path.join(scenario_folder, x))])
            assert len(cav_list) > 0

            """
            roadside unit data's id is always negative, 
            so here we want tomake sure they will be in the end of the list as they shouldn'tbe ego vehicle.
            """
            if int(cav_list[0]) < 0:
                cav_list = cav_list[1:] + [cav_list[0]]

            """
            make the first cav to be ego modality
            """
            if getattr(self, "heterogeneous", False):
                scenario_name = scenario_folder.split("/")[-1]
                cav_list = self.adaptor.reorder_cav_list(cav_list, scenario_name)

            # loop over all CAV data
            for (j, cav_id) in enumerate(cav_list):
                if j > self.max_cav - 1:
                    # print(f'\033[93m[WARNING]\033[0m In {scenario_folder}, there are too many cavs reinitialize.')
                    logger.warning(f'In {scenario_folder}, there are too many cavs reinitialize.')
                    break
                self.scenario_database[i][cav_id] = OrderedDict()

                # save all yaml files to the dictionary
                cav_path = os.path.join(scenario_folder, cav_id)

                yaml_files = sorted([os.path.join(cav_path, x) for x in os.listdir(cav_path) if
                                     x.endswith('.yaml') and 'additional' not in x])

                # this timestamp is not ready
                # TODO: 直接固定写死在这里是否合适？
                yaml_files = [x for x in yaml_files if not ("2021_08_20_21_10_24" in x and "000265" in x)]

                timestamps = self.extract_timestamps(yaml_files)

                for timestamp in timestamps:
                    self.scenario_database[i][cav_id][timestamp] = OrderedDict()
                    yaml_file = os.path.join(cav_path, timestamp + '.yaml')
                    lidar_file = os.path.join(cav_path, timestamp + '.pcd')
                    camera_files = self.find_camera_files(cav_path, timestamp)
                    # 加载深度信息时, 默认采用的是 OPV2V_Hetero 数据集, 这里只是将对应时间戳的深度信息的文件名加载进来, 并没有真正读取文件
                    # 所以在第一阶段训练时, 并没有报错, 当进行第二阶段的训练时才会报错.
                    depth_files = self.find_camera_files(cav_path, timestamp, sensor="depth")
                    # TODO： 从这里可以看出， OPV2V-H 数据集采集的场景和 OPV2V 的相同, 只不过是传感器多了一些罢了
                    depth_files = [depth_file.replace("OPV2V", "OPV2V_Hetero") for depth_file in depth_files]

                    self.scenario_database[i][cav_id][timestamp]['yaml'] = yaml_file
                    self.scenario_database[i][cav_id][timestamp]['lidar'] = lidar_file
                    self.scenario_database[i][cav_id][timestamp]['cameras'] = camera_files
                    self.scenario_database[i][cav_id][timestamp]['depths'] = depth_files

                    if getattr(self, "heterogeneous", False):
                        scenario_name = scenario_folder.split("/")[-1]

                        cav_modality = self.adaptor.reassign_cav_modality(
                            self.modality_assignment[scenario_name][cav_id], j)

                        self.scenario_database[i][cav_id][timestamp]['modality_name'] = cav_modality

                        self.scenario_database[i][cav_id][timestamp]['lidar'] = (
                            self.adaptor.switch_lidar_channels(cav_modality, lidar_file))

                    # load extra data
                    for file_extension in self.add_data_extension:
                        file_name = os.path.join(cav_path, timestamp + '_' + file_extension)
                        self.scenario_database[i][cav_id][timestamp][file_extension] = file_name

                # Assume all cavs will have the same timestamps length. Thus
                # we only need to calculate for the first vehicle in the
                # scene.
                if j == 0:
                    # we regard the agent with the minimum id as the ego
                    self.scenario_database[i][cav_id]['ego'] = True
                    if not self.len_record:
                        self.len_record.append(len(timestamps))
                    else:
                        prev_last = self.len_record[-1]
                        self.len_record.append(prev_last + len(timestamps))
                else:
                    self.scenario_database[i][cav_id]['ego'] = False
        # print(f"len: {self.len_record[-1]}\n")
        logger.success(f'len: {self.len_record[-1]}')

    def retrieve_base_data(self, idx):
        """
        Given the index, return the corresponding data.

        Parameters
        ----------
        idx : int
            Index given by dataloader.

        Returns
        -------
        data : dict
            The dictionary contains loaded yaml params and lidar data for
            each cav.
        """
        # we loop the accumulated length list to see get the scenario index
        scenario_index = 0
        for i, ele in enumerate(self.len_record):
            if idx < ele:
                scenario_index = i
                break
        scenario_database = self.scenario_database[scenario_index]

        # check the timestamp index
        timestamp_index = idx if scenario_index == 0 else \
            idx - self.len_record[scenario_index - 1]
        # retrieve the corresponding timestamp key
        timestamp_key = self.return_timestamp_key(scenario_database,timestamp_index)
        data = OrderedDict()
        # load files for all CAVs
        for cav_id, cav_content in scenario_database.items():
            data[cav_id] = OrderedDict()
            data[cav_id]['ego'] = cav_content['ego']

            # load param file: json is faster than yaml
            json_file = cav_content[timestamp_key]['yaml'].replace("yaml", "json")
            if os.path.exists(json_file):
                with open(json_file, "r") as f:
                    data[cav_id]['params'] = json.load(f)
            else:
                data[cav_id]['params'] = load_yaml(cav_content[timestamp_key]['yaml'])

            # load camera file: hdf5 is faster than png
            hdf5_file = cav_content[timestamp_key]['cameras'][0].replace("camera0.png", "imgs.hdf5")

            if self.use_hdf5 and os.path.exists(hdf5_file):
                with h5py.File(hdf5_file, "r") as f:
                    data[cav_id]['camera_data'] = []
                    data[cav_id]['depth_data'] = []
                    for i in range(4):
                        if self.load_camera_file:
                            data[cav_id]['camera_data'].append(Image.fromarray(f[f'camera{i}'][()]))
                        if self.load_depth_file:
                            data[cav_id]['depth_data'].append(Image.fromarray(f[f'depth{i}'][()]))
            else:
                if self.load_camera_file:
                    data[cav_id]['camera_data'] = load_camera_data(cav_content[timestamp_key]['cameras'])
                if self.load_depth_file:
                    data[cav_id]['depth_data'] = load_camera_data(cav_content[timestamp_key]['depths'])

                    # load lidar file
            if self.load_lidar_file or self.visualize:
                data[cav_id]['lidar_np'] = pcd_utils.pcd_to_np(cav_content[timestamp_key]['lidar'])

            if getattr(self, "heterogeneous", False):
                data[cav_id]['modality_name'] = cav_content[timestamp_key]['modality_name']

            for file_extension in self.add_data_extension:
                # if not find in the current directory
                # go to additional folder
                if not os.path.exists(cav_content[timestamp_key][file_extension]):
                    cav_content[timestamp_key][file_extension] = cav_content[timestamp_key][file_extension].replace(
                        "train", "additional/train")
                    cav_content[timestamp_key][file_extension] = cav_content[timestamp_key][file_extension].replace(
                        "validate", "additional/validate")
                    cav_content[timestamp_key][file_extension] = cav_content[timestamp_key][file_extension].replace(
                        "test", "additional/test")

                if '.yaml' in file_extension:
                    data[cav_id][file_extension] = load_yaml(cav_content[timestamp_key][file_extension])
                else:
                    data[cav_id][file_extension] = cv2.imread(cav_content[timestamp_key][file_extension])
                    if data[cav_id][file_extension] is None:
                        logger.error(f'Failed to read the image for {cav_id} with extension {file_extension} '
                                     f'from {cav_content[timestamp_key][file_extension]}')

        return data

    def __len__(self):
        return self.len_record[-1]

    def __getitem__(self, idx):
        """
        Abstract method, needs to be define by the children class.
        """
        pass

    @staticmethod
    def extract_timestamps(yaml_files):
        """
        Given the list of the yaml files, extract the mocked timestamps.

        Parameters
        ----------
        yaml_files : list
            The full path of all yaml files of ego vehicle

        Returns
        -------
        timestamps : list
            The list containing timestamps only.
        """
        timestamps = []

        for file in yaml_files:
            res = file.split('/')[-1]

            timestamp = res.replace('.yaml', '')
            timestamps.append(timestamp)

        return timestamps

    @staticmethod
    def return_timestamp_key(scenario_database, timestamp_index):
        """
        Given the timestamp index, return the correct timestamp key, e.g.
        2 --> '000078'.

        Parameters
        ----------
        scenario_database : OrderedDict
            The dictionary contains all contents in the current scenario.

        timestamp_index : int
            The index for timestamp.

        Returns
        -------
        timestamp_key : str
            The timestamp key saved in the cav dictionary.
        """
        # get all timestamp keys
        timestamp_keys = list(scenario_database.items())[0][1]
        # retrieve the correct index
        timestamp_key = list(timestamp_keys.items())[timestamp_index][0]

        return timestamp_key

    @staticmethod
    def find_camera_files(cav_path, timestamp, sensor="camera"):
        """
        Retrieve the paths to all camera files.

        Parameters
        ----------
        cav_path : str
            The full file path of current cav.

        timestamp : str
            Current timestamp

        sensor : str
            "camera" or "depth"

        Returns
        -------
        camera_files : list
            The list containing all camera png file paths.
        """
        camera0_file = os.path.join(cav_path, timestamp + f'_{sensor}0.png')
        camera1_file = os.path.join(cav_path, timestamp + f'_{sensor}1.png')
        camera2_file = os.path.join(cav_path, timestamp + f'_{sensor}2.png')
        camera3_file = os.path.join(cav_path, timestamp + f'_{sensor}3.png')
        return [camera0_file, camera1_file, camera2_file, camera3_file]

    def augment(self, lidar_np, object_bbx_center, object_bbx_mask):
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
        tmp_dict = {'lidar_np': lidar_np,
                    'object_bbx_center': object_bbx_center,
                    'object_bbx_mask': object_bbx_mask}
        tmp_dict = self.data_augmentor.forward(tmp_dict)

        lidar_np = tmp_dict['lidar_np']
        object_bbx_center = tmp_dict['object_bbx_center']
        object_bbx_mask = tmp_dict['object_bbx_mask']

        return lidar_np, object_bbx_center, object_bbx_mask

    def generate_object_center_lidar(self, cav_contents, reference_lidar_pose):
        """
        Retrieve all objects in a format of (n, 7), where 7 represents
        x, y, z, l, w, h, yaw or x, y, z, h, w, l, yaw.
        The object_bbx_center is in ego coordinate.

        Notice: it is a wrap of postprocessor

        Parameters
        ----------
        cav_contents : list
            List of dictionary, save all cavs' information.
            in fact it is used in get_item_single_car, so the list length is 1

        reference_lidar_pose : list
            The final target lidar pose with length 6.

        Returns
        -------
        object_np : np.ndarray
            Shape is (max_num, 7).
        mask : np.ndarray
            Shape is (max_num,).
        object_ids : list
            Length is number of bbx in current sample.
        """
        return self.post_processor.generate_object_center(cav_contents,
                                                          reference_lidar_pose)

    def generate_object_center_camera(self, cav_contents, reference_lidar_pose):
        """
        Retrieve all objects in a format of (n, 7), where 7 represents
        x, y, z, l, w, h, yaw or x, y, z, h, w, l, yaw.
        The object_bbx_center is in ego coordinate.

        Notice: it is a wrap of postprocessor

        Parameters
        ----------
        cav_contents : list
            List of dictionary, save all cavs' information.
            in fact it is used in get_item_single_car, so the list length is 1

        reference_lidar_pose : list
            The final target lidar pose with length 6.

        visibility_map : np.ndarray
            for OPV2V, its 256*256 resolution. 0.39m per pixel. heading up.

        Returns
        -------
        object_np : np.ndarray
            Shape is (max_num, 7).
        mask : np.ndarray
            Shape is (max_num,).
        object_ids : list
            Length is number of bbx in current sample.
        """
        return self.post_processor.generate_visible_object_center(cav_contents, reference_lidar_pose)

    def get_ext_int(self, params, camera_id):
        camera_coords = np.array(params["camera%d" % camera_id]["cords"]).astype(
            np.float32)
        camera_to_lidar = x1_to_x2(
            camera_coords, params["lidar_pose_clean"]
        ).astype(np.float32)  # T_LiDAR_camera
        camera_to_lidar = camera_to_lidar @ np.array(
            [[0, 0, 1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
            dtype=np.float32)  # UE4 coord to opencv coord
        camera_intrinsic = np.array(params["camera%d" % camera_id]["intrinsic"]).astype(
            np.float32
        )
        return camera_to_lidar, camera_intrinsic
