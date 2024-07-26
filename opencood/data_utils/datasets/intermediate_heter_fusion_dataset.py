"""
-*- coding: utf-8 -*-
Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
License: TDG-Attribution-NonCommercial-NoDistrib

intermediate heter fusion dataset

Note that for DAIR-V2X dataset,
Each agent should retrieve the objects itself, and merge them by iou, instead of using the cooperative label.
"""

import copy
import math
from collections import OrderedDict
from typing import Mapping

import numpy as np
import torch

from opencood.data_utils.pre_processor import build_preprocessor
from opencood.utils import box_utils as box_utils
from opencood.utils.camera_utils import (
    sample_augmentation,
    img_transform,
    normalize_img,
    img_to_tensor,
)
from opencood.utils.common_utils import merge_features_to_dict, compute_iou, convert_format
from opencood.utils.common_utils import read_json
from opencood.utils.heter_utils import Adaptor
from opencood.utils.pcd_utils import (
    mask_ego_points,
    shuffle_points,
    downsample_lidar_minimum,
)
from opencood.utils.pose_utils import add_noise_data_dict
from opencood.utils.transformation_utils import x1_to_x2, get_pairwise_transformation


def getIntermediateheterFusionDataset(cls):
    """
    cls: the Basedataset.
    """

    def get_ego_info(base_data_dict: Mapping):
        """
        从 base_data_dict 里面找到对应的 ego 车辆, 并对变量进行一些赋值, 同时检查值的合法性
        """
        ego_id = -1
        ego_lidar_pose = []
        ego_cav_base = None
        for cav_id, cav_content in base_data_dict.items():
            if cav_content['ego']:
                ego_id = cav_id
                ego_lidar_pose = cav_content['params']['lidar_pose']
                ego_cav_base = cav_content
                break
        assert cav_id == list(base_data_dict.keys())[0], "The first element in the OrderedDict must be ego"
        assert ego_id != -1
        assert len(ego_lidar_pose) > 0
        return ego_id, ego_lidar_pose, ego_cav_base

    class IntermediateheterFusionDataset(cls):
        def __init__(self, params: Mapping, visualize: bool, train=True):
            super().__init__(params, visualize, train)
            # intermediate and supervise single
            self.supervise_single = True if \
                'supervise_single' in params['model']['args'] and params['model']['args']['supervise_single'] else False
            self.proj_first = False if 'proj_first' not in params['fusion']['args'] \
                else params['fusion']['args']['proj_first']

            self.anchor_box = self.post_processor.generate_anchor_box()
            self.anchor_box_torch = torch.from_numpy(self.anchor_box)

            self.heterogeneous = True
            # 某一场景下车辆 id 与 模态的对应关系
            self.modality_assignment = None if (
                    'assignment_path' not in params['heter'] or params['heter']['assignment_path'] is None) \
                else read_json(params['heter']['assignment_path'])

            # 自车的模态
            self.ego_modality = params['heter']['ego_modality']  # "m1" or "m1&m2" or "m3"

            self.modality_name_list = list(params['heter']['modality_setting'].keys())
            self.sensor_type_dict = OrderedDict()

            lidar_channels_dict = params['heter'].get('lidar_channels_dict', OrderedDict())
            mapping_dict = params['heter']['mapping_dict']
            cav_preference = params['heter'].get("cav_preference", None)

            self.adaptor = Adaptor(self.ego_modality, self.modality_name_list, self.modality_assignment,
                                   lidar_channels_dict, mapping_dict, cav_preference, train)

            for modality_name, modal_setting in params['heter']['modality_setting'].items():
                self.sensor_type_dict[modality_name] = modal_setting['sensor_type']
                if modal_setting['sensor_type'] == 'lidar':
                    setattr(self, f"pre_processor_{modality_name}",
                            build_preprocessor(modal_setting['preprocess'], train))
                elif modal_setting['sensor_type'] == 'camera':
                    setattr(self, f"data_aug_conf_{modality_name}", modal_setting['data_aug_conf'])

                else:
                    raise "Not support this type of sensor"

            self.reinitialize()  # 该函数由父类 (cls) 实现

            self.kd_flag = params.get('kd_flag', False)  # TODO：kd_flag 是什么意思？

            self.box_align = False
            if "box_align" in params:
                self.box_align = True
                self.stage1_result_path = \
                    params['box_align']['train_result'] if train else params['box_align']['val_result']
                self.stage1_result = read_json(self.stage1_result_path)
                self.box_align_args = params['box_align']['args']

        def get_item_single_car(self, selected_cav_base: Mapping, ego_cav_base):
            """
            Process a single CAV's information for the train/test pipeline.

            Parameters
            ----------
            selected_cav_base : dict
                The dictionary contains a single CAV's raw information including 'params', 'camera_data'
            ego_pose : list, length 6
                The ego vehicle lidar pose under world coordinate.
            ego_pose_clean : list, length 6
                only used for gt box generation

            Returns
            -------
            selected_cav_processed : dict
                The dictionary contains the cav's processed information.
            """
            selected_cav_processed = {}
            ego_pose, ego_pose_clean = ego_cav_base['params']['lidar_pose'], ego_cav_base['params']['lidar_pose_clean']

            # calculate the transformation matrix
            transformation_matrix = x1_to_x2(selected_cav_base['params']['lidar_pose'], ego_pose)  # T_ego_cav
            transformation_matrix_clean = x1_to_x2(selected_cav_base['params']['lidar_pose_clean'], ego_pose_clean)

            modality_name = selected_cav_base['modality_name']
            sensor_type = self.sensor_type_dict[modality_name]

            # lidar
            if sensor_type == "lidar" or self.visualize:
                # process lidar
                lidar_np = selected_cav_base['lidar_np']
                lidar_np = shuffle_points(lidar_np)
                # remove points that hit itself
                lidar_np = mask_ego_points(lidar_np)
                # project the lidar to ego space
                # x,y,z in ego space
                projected_lidar = box_utils.project_points_by_matrix_torch(lidar_np[:, :3], transformation_matrix)
                if self.proj_first:
                    lidar_np[:, :3] = projected_lidar

                if self.visualize:  # filter lidar
                    selected_cav_processed.update({'projected_lidar': projected_lidar})

                if self.kd_flag:
                    lidar_proj_np = copy.deepcopy(lidar_np)
                    lidar_proj_np[:, :3] = projected_lidar

                    selected_cav_processed.update({'projected_lidar': lidar_proj_np})

                    # 2023.8.31, to correct discretization errors. Just replace one point to avoid empty voxels. need fix later.
                    lidar_proj_np[np.random.randint(0, lidar_proj_np.shape[0]), :3] = np.array([0, 0, 0])
                    processed_lidar_proj = eval(f"self.pre_processor_{modality_name}").preprocess(lidar_proj_np)
                    selected_cav_processed.update({f'processed_features_{modality_name}_proj': processed_lidar_proj})

                if sensor_type == "lidar":
                    processed_lidar = eval(f"self.pre_processor_{modality_name}").preprocess(lidar_np)
                    # TODO: `processed_features_` ? 通过上面那个函数, 简单地调几个包就把特征提取出来了?
                    selected_cav_processed.update({f'processed_features_{modality_name}': processed_lidar})

            # generate targets label single GT, note the reference pose is itself.
            """
            根据 selected_cav_base 中 ['params']['vehicles'] 生成.  ['params']['vehicles'] 这个里面应该就是人工标注的
            车辆信息, 根据这个进行训练. 为什么要在两段 if 长代码判断中加这么一段呢 TODO: post_processor 中好像并没有用到上面提取出来的 feature
            """
            object_bbx_center, object_bbx_mask, object_ids = \
                self.generate_object_center([selected_cav_base], selected_cav_base['params']['lidar_pose'])
            label_dict = self.post_processor.generate_label(
                gt_box_center=object_bbx_center, anchors=self.anchor_box, mask=object_bbx_mask
            )
            selected_cav_processed.update({
                "single_label_dict": label_dict,
                "single_object_bbx_center": object_bbx_center,
                "single_object_bbx_mask": object_bbx_mask
            })

            # camera
            if sensor_type == "camera":
                camera_data_list = selected_cav_base["camera_data"]
                params = selected_cav_base["params"]
                imgs, rots, trans, intrins, extrinsics, post_rots, post_trans = [], [], [], [], [], [], []

                for idx, img in enumerate(camera_data_list):
                    camera_to_lidar, camera_intrinsic = self.get_ext_int(params, idx)

                    intrin = torch.from_numpy(camera_intrinsic)
                    rot = torch.from_numpy(
                        camera_to_lidar[:3, :3]
                    )  # R_wc, we consider world-coord is the lidar-coord
                    tran = torch.from_numpy(camera_to_lidar[:3, 3])  # T_wc

                    post_rot = torch.eye(2)
                    post_tran = torch.zeros(2)

                    img_src = [img]

                    # depth
                    if self.load_depth_file:
                        depth_img = selected_cav_base["depth_data"][idx]
                        img_src.append(depth_img)
                    else:
                        depth_img = None

                    # data augmentation
                    resize, resize_dims, crop, flip, rotate = sample_augmentation(
                        eval(f"self.data_aug_conf_{modality_name}"), self.train
                    )
                    img_src, post_rot2, post_tran2 = img_transform(
                        img_src, post_rot, post_tran, resize=resize, resize_dims=resize_dims,
                        crop=crop, flip=flip, rotate=rotate)
                    # for convenience, make augmentation matrices 3x3
                    post_tran = torch.zeros(3)
                    post_rot = torch.eye(3)
                    post_tran[:2] = post_tran2
                    post_rot[:2, :2] = post_rot2

                    # decouple RGB and Depth

                    img_src[0] = normalize_img(img_src[0])
                    if self.load_depth_file:
                        img_src[1] = img_to_tensor(img_src[1]) * 255

                    imgs.append(torch.cat(img_src, dim=0))
                    intrins.append(intrin)
                    extrinsics.append(torch.from_numpy(camera_to_lidar))
                    rots.append(rot)
                    trans.append(tran)
                    post_rots.append(post_rot)
                    post_trans.append(post_tran)

                selected_cav_processed.update({
                    f"image_inputs_{modality_name}": {
                        "imgs": torch.stack(imgs),  # [Ncam, 3or4, H, W]
                        "intrins": torch.stack(intrins),
                        "extrinsics": torch.stack(extrinsics),
                        "rots": torch.stack(rots),
                        "trans": torch.stack(trans),
                        "post_rots": torch.stack(post_rots),
                        "post_trans": torch.stack(post_trans),
                    }
                })
            """
            感觉这部分代码完全可以和它上面这段 if 语句的上面一段放在一起
            这一段是根据真是位置生成框, 上一段是根据有噪声的位置 (如果没有设置添加噪声, 这两部分生成的框应该是相同的)
            """
            # anchor box
            selected_cav_processed.update({"anchor_box": self.anchor_box})

            # note the reference pose ego
            object_bbx_center, object_bbx_mask, object_ids = \
                self.generate_object_center([selected_cav_base], ego_pose_clean)

            selected_cav_processed.update({
                "object_bbx_center": object_bbx_center[object_bbx_mask == 1],
                "object_bbx_mask": object_bbx_mask,
                "object_ids": object_ids,
                'transformation_matrix': transformation_matrix,
                'transformation_matrix_clean': transformation_matrix_clean
            })

            return selected_cav_processed

        def __getitem__(self, idx):
            """
            根据 idx 获取到某一个场景下全部的 cav, 同时添加噪声, 这一步就模拟了通信
            2024.07.19 补充: 好像在这里说完成了通信是不太适合的, 考虑到这是中期融合,
            而中期融合融合的是 encoder 后的特征. 很明显不符合中期融合的概念.
            同时, 论文给出的图片上指出, 在 Message Transmission 阶段不止传递 Feature, 也传递 Pose, 也许这里传递的是 Pose?
            """
            base_data_dict = self.retrieve_base_data(idx)
            base_data_dict = add_noise_data_dict(base_data_dict, self.params['noise_setting'])

            processed_data_dict = OrderedDict()
            processed_data_dict['ego'] = {}

            ego_id, ego_lidar_pose, ego_cav_base = get_ego_info(base_data_dict)
            # can contain lidar or camera
            input_list_dict = {
                'input_list_m1': [],
                'input_list_m2': [],
                'input_list_m3': [],
                'input_list_m4': []
            }

            agent_modality_list = []
            object_stack = []
            object_id_stack = []  # 存储着某一场景下全部 cav 标记的全部物体的 id (不再按照场景下 cav 的 id 来进行分类)
            single_label_list = []
            single_object_bbx_center_list = []
            single_object_bbx_mask_list = []
            projected_lidar_clean_list = []  # disconet

            # TODO：和可视化有关的部分, 也许可以先不用看?
            if self.visualize or self.kd_flag:
                projected_lidar_stack = []
                # 2023.8.31 to correct discretization errors with kd flag
                input_list_proj_dict = {
                    'input_list_m1_proj': [],
                    'input_list_m2_proj': [],
                    'input_list_m3_proj': [],
                    'input_list_m4_proj': [],
                }

            cav_id_list, exclude_agent, lidar_pose_clean_list, lidar_pose_list = \
                self._get_legal_cav_ids(base_data_dict, ego_lidar_pose)
            if len(cav_id_list) == 0:
                return None
            for cav_id in exclude_agent:  # 弹出非法的 id, TODO: 不弹出会怎么样呢?
                base_data_dict.pop(cav_id)

            """和对齐有关的部分? 也许可以先不用看?"""
            ########## Updated by Yifan Lu 2022.1.26 ############
            # box align to correct pose.
            # stage1_content contains all agent. Even out of comm range.
            if self.box_align and str(idx) in self.stage1_result.keys():
                from opencood.models.sub_modules.box_align_v2 import box_alignment_relative_sample_np
                stage1_content = self.stage1_result[str(idx)]
                if stage1_content is not None:
                    all_agent_id_list = stage1_content['cav_id_list']  # include those out of range
                    all_agent_corners_list = stage1_content['pred_corner3d_np_list']
                    all_agent_uncertainty_list = stage1_content['uncertainty_np_list']

                    cur_agent_id_list = cav_id_list
                    cur_agent_pose = [base_data_dict[cav_id]['params']['lidar_pose'] for cav_id in cav_id_list]
                    cur_agnet_pose = np.array(cur_agent_pose)
                    # indexing current agent in `all_agent_id_list`
                    cur_agent_in_all_agent = [all_agent_id_list.index(cur_agent) for cur_agent in cur_agent_id_list]

                    pred_corners_list = [np.array(all_agent_corners_list[cur_in_all_ind], dtype=np.float64)
                                         for cur_in_all_ind in cur_agent_in_all_agent]
                    uncertainty_list = [np.array(all_agent_uncertainty_list[cur_in_all_ind], dtype=np.float64)
                                        for cur_in_all_ind in cur_agent_in_all_agent]

                    if sum([len(pred_corners) for pred_corners in pred_corners_list]) != 0:
                        refined_pose = box_alignment_relative_sample_np(pred_corners_list, cur_agnet_pose,
                                                                        uncertainty_list=uncertainty_list,
                                                                        **self.box_align_args)
                        cur_agnet_pose[:, [0, 1, 4]] = refined_pose

                        for i, cav_id in enumerate(cav_id_list):
                            lidar_pose_list[i] = cur_agnet_pose[i].tolist()
                            base_data_dict[cav_id]['params']['lidar_pose'] = cur_agnet_pose[i].tolist()

            # 坐标转换矩阵, 两两之间各有一个
            pairwise_t_matrix = get_pairwise_transformation(base_data_dict, self.max_cav, self.proj_first)

            # 提取出含有噪声和无噪声的数据 (如果没有设置噪声, 这两个应该是一样的)
            lidar_poses = np.array(lidar_pose_list).reshape(-1, 6)  # [N_cav, 6]
            lidar_poses_clean = np.array(lidar_pose_clean_list).reshape(-1, 6)  # [N_cav, 6]

            # 判断一下, 如果不添加噪声, 两者应该是相同的
            assert not (self.params['noise_setting']['add_noise'] and lidar_poses != lidar_poses_clean)

            """
            merge preprocessed features from different cavs into the same dict
            将来自不同cav的预处理特征合并到同一字典中
            """
            cav_num = len(cav_id_list)
            # agent_modality_list, object_stack, object_id_stack, single_label_list, \
            #     single_object_bbx_center_list, single_object_bbx_mask_list = \
            #     self._merge_cav_features(base_data_dict, cav_id_list, input_list_dict, projected_lidar_stack,
            #                              input_list_proj_dict)
            for _i, cav_id in enumerate(cav_id_list):
                selected_cav_base = base_data_dict[cav_id]
                modality_name = selected_cav_base['modality_name']
                sensor_type = self.sensor_type_dict[selected_cav_base['modality_name']]

                # dynamic object center generator! for heterogeneous input
                if not self.visualize:
                    self.generate_object_center = eval(f"self.generate_object_center_{sensor_type}")
                # need discussion. In test phase, use lidar label.
                else:
                    self.generate_object_center = self.generate_object_center_lidar

                selected_cav_processed = self.get_item_single_car(selected_cav_base, ego_cav_base)

                object_stack.append(selected_cav_processed['object_bbx_center'])
                object_id_stack += selected_cav_processed['object_ids']

                if sensor_type == "lidar":
                    input_list_dict[f'input_list_{modality_name}'].append(
                        selected_cav_processed[f"processed_features_{modality_name}"])
                    # eval(f"input_list_{modality_name}").append(
                    #     selected_cav_processed[f"processed_features_{modality_name}"])
                elif sensor_type == "camera":
                    input_list_dict[f"input_list_{modality_name}"].append(
                        selected_cav_processed[f"image_inputs_{modality_name}"])
                    # eval(f"input_list_{modality_name}").append(selected_cav_processed[f"image_inputs_{modality_name}"])
                else:
                    raise

                agent_modality_list.append(modality_name)
                """
                这部分应该还是和可视化有关的, 所以还是不用看
                """
                if self.visualize or self.kd_flag:
                    # heterogeneous setting do not support disconet' kd
                    projected_lidar_stack.append(selected_cav_processed['projected_lidar'])
                    if sensor_type == "lidar" and self.kd_flag:
                        input_list_proj_dict[f"input_list_{modality_name}_proj"].append(
                            selected_cav_processed[f"processed_features_{modality_name}_proj"])
                        # eval(f"input_list_{modality_name}_proj").append(
                        #     selected_cav_processed[f"processed_features_{modality_name}_proj"])

                if self.supervise_single or self.heterogeneous:
                    single_label_list.append(selected_cav_processed['single_label_dict'])
                    single_object_bbx_center_list.append(selected_cav_processed['single_object_bbx_center'])
                    single_object_bbx_mask_list.append(selected_cav_processed['single_object_bbx_mask'])

            # generate single view GT label
            """
            将之前获得的数据从 numpy 类型转换为 torch 类型, 然后放到字典里面
            """
            if self.supervise_single or self.heterogeneous:
                single_label_dicts = self.post_processor.collate_batch(single_label_list)
                single_object_bbx_center = torch.from_numpy(np.array(single_object_bbx_center_list))
                single_object_bbx_mask = torch.from_numpy(np.array(single_object_bbx_mask_list))
                processed_data_dict['ego'].update({
                    "single_label_dict_torch": single_label_dicts,
                    "single_object_bbx_center_torch": single_object_bbx_center,
                    "single_object_bbx_mask_torch": single_object_bbx_mask,
                })

            # exculude all repetitve objects, DAIR-V2X
            """
            对 dairv2x 的特殊处理, 我感觉也应该封装到一个函数里面
            """
            if self.params['fusion']['dataset'] == 'dairv2x':
                if len(object_stack) == 1:
                    object_stack = object_stack[0]
                else:
                    ego_boxes_np = object_stack[0]
                    cav_boxes_np = object_stack[1]
                    order = self.params['postprocess']['order']
                    ego_corners_np = box_utils.boxes_to_corners_3d(ego_boxes_np, order)
                    cav_corners_np = box_utils.boxes_to_corners_3d(cav_boxes_np, order)
                    ego_polygon_list = list(convert_format(ego_corners_np))
                    cav_polygon_list = list(convert_format(cav_corners_np))
                    iou_thresh = 0.05

                    gt_boxes_from_cav = []
                    for i in range(len(cav_polygon_list)):
                        cav_polygon = cav_polygon_list[i]
                        ious = compute_iou(cav_polygon, ego_polygon_list)
                        if (ious > iou_thresh).any():
                            continue
                        gt_boxes_from_cav.append(cav_boxes_np[i])

                    if len(gt_boxes_from_cav):
                        object_stack_from_cav = np.stack(gt_boxes_from_cav)
                        object_stack = np.vstack([ego_boxes_np, object_stack_from_cav])
                    else:
                        object_stack = ego_boxes_np

                unique_indices = np.arange(object_stack.shape[0])
                object_id_stack = np.arange(object_stack.shape[0])
            else:
                # exclude all repetitive objects, OPV2V-H
                unique_indices = [object_id_stack.index(x) for x in set(object_id_stack)]  # id 去重, 提取出索引
                # `np.vstack` 纵向堆叠, 一开始 object_stack 是一个 list, 所以用 `np.vstack` 进行将为, object_stack 里面的内容提取出来
                object_stack = np.vstack(object_stack)
                object_stack = object_stack[unique_indices]

            # make sure bounding boxes across all frames have the same number # TODO: 应该就是一个对齐操作
            object_bbx_center = np.zeros((self.params['postprocess']['max_num'], 7))
            mask = np.zeros(self.params['postprocess']['max_num'])
            object_bbx_center[:object_stack.shape[0], :] = object_stack
            mask[:object_stack.shape[0]] = 1

            for modality_name in self.modality_name_list:
                if self.sensor_type_dict[modality_name] == "lidar":
                    # merged_feature_dict = merge_features_to_dict(eval(f"input_list_{modality_name}"))
                    """
                    一开始 input_list_dict 按照 cav_id 的数量分别存储着 'voxel_features', 'voxel_coords', 'voxel_num_points'
                    经过这个函数后, 上述三个键分别存储着 cav_id 中的对应数据
                    """
                    merged_feature_dict = merge_features_to_dict(input_list_dict[f"input_list_{modality_name}"])
                    processed_data_dict['ego'].update({f'input_{modality_name}': merged_feature_dict})  # maybe None
                elif self.sensor_type_dict[modality_name] == "camera":
                    merged_image_inputs_dict = merge_features_to_dict(input_list_dict[f"input_list_{modality_name}"],
                                                                      merge='stack')
                    # merged_image_inputs_dict = merge_features_to_dict(eval(f"input_list_{modality_name}"),
                    #                                                   merge='stack')
                    processed_data_dict['ego'].update(
                        {f'input_{modality_name}': merged_image_inputs_dict})  # maybe None
            """
            是不是有 kd_flag 的地方就是和可视化有关的呢? 与可视化有关的东西都不看???
            """
            if self.kd_flag:
                # heterogenous setting do not support DiscoNet's kd
                # stack_lidar_np = np.vstack(projected_lidar_stack)
                # stack_lidar_np = mask_points_by_range(stack_lidar_np,
                #                             self.params['preprocess'][
                #                                 'cav_lidar_range'])
                # stack_feature_processed = self.pre_processor.preprocess(stack_lidar_np)
                for modality_name in self.modality_name_list:
                    # processed_data_dict['ego'].update({
                    #     f'input_{modality_name}_proj': merge_features_to_dict(eval(f"input_list_{modality_name}_proj"))
                    processed_data_dict['ego'].update({
                        f'input_{modality_name}_proj':
                            merge_features_to_dict(input_list_proj_dict[f"input_list_{modality_name}_proj"])
                        # maybe None
                    })

            processed_data_dict['ego'].update({'agent_modality_list': agent_modality_list})

            # generate targets label
            label_dict = self.post_processor.generate_label(gt_box_center=object_bbx_center, anchors=self.anchor_box,
                                                            mask=mask)
            """
            TODO: 最后返回的给模型的数据, 包含以下内容
            里面的许多参数我都见过许多类似的, 内容经过更新
            是否里面的每个参数都用到了呢?
            """
            """
            在 m1_base 部分, 注释掉 `'anchor_box'` 后仍可正常运行 (可以训练和验证, 但是最后的推理情不得而知). 看代码的话 m1_base 只使用了 voxel (体素) 有关的部分, 是否有这个影响应该不大,
            如果这样说的话, 那么这些框是否都和 m1_base 的训练无关, 也都可以去掉? 但是其他的呢? 比如说 m2_base 的呢?
            去掉 `'anchor_box'`后, 推理不可以正常跑,
            """
            processed_data_dict['ego'].update({
                'object_bbx_center': object_bbx_center,
                'object_bbx_mask': mask,
                'object_ids': [object_id_stack[i] for i in unique_indices],
                'anchor_box': self.anchor_box,
                'label_dict': label_dict,
                'cav_num': cav_num,
                'pairwise_t_matrix': pairwise_t_matrix,
                'lidar_poses_clean': lidar_poses_clean,
                'lidar_poses': lidar_poses
            })

            if self.visualize:
                processed_data_dict['ego'].update({'origin_lidar': np.vstack(projected_lidar_stack)})

            processed_data_dict['ego'].update({'sample_idx': idx, 'cav_id_list': cav_id_list})

            return processed_data_dict

        def collate_batch_train(self, batch):
            """
            GPT: 用于在训练过程中将单个样本数据整合成批次数据的函数。
            它的目的是将不同车辆的传感器数据和标签整合到一个批次中，以便于模型的批量处理。
            :param batch:
            :return:
            """
            # Intermediate fusion is different the other two
            output_dict = {'ego': {}}

            inputs_list_dict = {'inputs_list_m1': [], 'inputs_list_m2': [], 'inputs_list_m3': [], 'inputs_list_m4': []}

            inputs_list_proj_dict = {'inputs_list_m1_proj': [], 'inputs_list_m2_proj': [],
                                     'inputs_list_m3_proj': [], 'inputs_list_m4_proj': [], }

            object_bbx_center = []
            object_bbx_mask = []
            object_ids = []

            agent_modality_list = []
            # used to record different scenario
            record_len = []
            label_dict_list = []
            lidar_pose_list = []
            origin_lidar = []
            lidar_pose_clean_list = []

            # pairwise transformation matrix
            pairwise_t_matrix_list = []

            # disconet
            teacher_processed_lidar_list = []

            ### 2022.10.10 single gt #### TODO: 这个 if 语句及其内容有什么作用?
            if self.supervise_single or self.heterogeneous:
                pos_equal_one_single = []
                neg_equal_one_single = []
                targets_single = []
                object_bbx_center_single = []
                object_bbx_mask_single = []

            # 对于批次中的每一个样本，提取其 ego 字典，
            # 并将相关的数据（如 object_bbx_center、object_bbx_mask 等）添加到相应的列表中。
            for i in range(len(batch)):
                ego_dict = batch[i]['ego']
                object_bbx_center.append(ego_dict['object_bbx_center'])
                object_bbx_mask.append(ego_dict['object_bbx_mask'])
                object_ids.append(ego_dict['object_ids'])
                lidar_pose_list.append(ego_dict['lidar_poses'])  # ego_dict['lidar_pose'] is np.ndarray [N,6]
                lidar_pose_clean_list.append(ego_dict['lidar_poses_clean'])

                for modality_name in self.modality_name_list:
                    if ego_dict[f'input_{modality_name}'] is not None:
                        # OrderedDict() if empty?
                        inputs_list_dict[f'inputs_list_{modality_name}'].append(ego_dict[f'input_{modality_name}'])
                        # eval(f"inputs_list_{modality_name}").append(ego_dict[f'input_{modality_name}'])

                agent_modality_list.extend(ego_dict['agent_modality_list'])

                record_len.append(ego_dict['cav_num'])
                label_dict_list.append(ego_dict['label_dict'])
                pairwise_t_matrix_list.append(ego_dict['pairwise_t_matrix'])

                if self.visualize:
                    origin_lidar.append(ego_dict['origin_lidar'])

                if self.kd_flag:
                    # hetero setting do not support disconet' kd
                    # teacher_processed_lidar_list.append(ego_dict['teacher_processed_lidar'])
                    for modality_name in self.modality_name_list:
                        if ego_dict[f'input_{modality_name}_proj'] is not None:
                            inputs_list_proj_dict[f'inputs_list_{modality_name}_proj'].append(
                                ego_dict[f'input_{modality_name}_proj'])
                            # eval(f"inputs_list_{modality_name}_proj").append(ego_dict[f"input_{modality_name}_proj"])

                ### 2022.10.10 single gt #### TODO: 这段 if 语句又有什么作用? GPT: 单一 GT 数据 (这个数据有什么用吗) 我好像并没有在其他类中看到这个东西
                if self.supervise_single or self.heterogeneous:
                    pos_equal_one_single.append(ego_dict['single_label_dict_torch']['pos_equal_one'])
                    neg_equal_one_single.append(ego_dict['single_label_dict_torch']['neg_equal_one'])
                    targets_single.append(ego_dict['single_label_dict_torch']['targets'])
                    object_bbx_center_single.append(ego_dict['single_object_bbx_center_torch'])
                    object_bbx_mask_single.append(ego_dict['single_object_bbx_mask_torch'])

            # convert to numpy, (B, max_num, 7) 将收集到的数据列表转换为张量
            object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
            object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))

            # 2023.2.5
            # 对于每一个模态，合并其输入数据（如 Lidar 数据或 Camera 数据），并将其添加到 output_dict 中。
            for modality_name in self.modality_name_list:
                # if len(eval(f"inputs_list_{modality_name}")) != 0:
                if len(inputs_list_dict[f'inputs_list_{modality_name}']) != 0:
                    if self.sensor_type_dict[modality_name] == "lidar":
                        # merged_feature_dict = merge_features_to_dict(eval(f"inputs_list_{modality_name}"))
                        merged_feature_dict = merge_features_to_dict(inputs_list_dict[f'inputs_list_{modality_name}'])
                        processed_lidar_torch_dict = \
                            eval(f"self.pre_processor_{modality_name}").collate_batch(merged_feature_dict)

                        output_dict['ego'].update({f'inputs_{modality_name}': processed_lidar_torch_dict})
                    elif self.sensor_type_dict[modality_name] == "camera":
                        merged_image_inputs_dict = \
                            merge_features_to_dict(inputs_list_dict[f'inputs_list_{modality_name}'], merge='cat')

                        output_dict['ego'].update({f'inputs_{modality_name}': merged_image_inputs_dict})

            output_dict['ego'].update({"agent_modality_list": agent_modality_list})

            record_len = torch.from_numpy(np.array(record_len, dtype=int))
            lidar_pose = torch.from_numpy(np.concatenate(lidar_pose_list, axis=0))
            lidar_pose_clean = torch.from_numpy(np.concatenate(lidar_pose_clean_list, axis=0))
            label_torch_dict = self.post_processor.collate_batch(label_dict_list)

            # for centerpoint
            label_torch_dict.update({'object_bbx_center': object_bbx_center, 'object_bbx_mask': object_bbx_mask})

            # (B, max_cav)
            pairwise_t_matrix = torch.from_numpy(np.array(pairwise_t_matrix_list))

            # add pairwise_t_matrix to label dict
            label_torch_dict['pairwise_t_matrix'] = pairwise_t_matrix
            label_torch_dict['record_len'] = record_len

            # object id is only used during inference, where batch size is 1. so here we only get the first element.
            """
            在这里去掉 `'object_bbx_center'` 和 `'object_ids'` 以及 `'anchor_box'` 后确实还可以训练, 但是训练速度仿佛变慢了
            epoch 0: 3.s/it, 现在 (epoch 1)是 1.s/it 
            全部加上以后速度好像也没差多少......
            """
            output_dict['ego'].update({
                'object_bbx_center': object_bbx_center,
                'object_bbx_mask': object_bbx_mask,
                'record_len': record_len,
                'label_dict': label_torch_dict,
                'object_ids': object_ids[0],
                'pairwise_t_matrix': pairwise_t_matrix,
                'lidar_pose_clean': lidar_pose_clean,
                'lidar_pose': lidar_pose,
                'anchor_box': self.anchor_box_torch
            })

            if self.visualize:
                origin_lidar = np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
                origin_lidar = torch.from_numpy(origin_lidar)
                output_dict['ego'].update({'origin_lidar': origin_lidar})

            if self.kd_flag:
                # teacher_processed_lidar_torch_dict = \
                #     self.pre_processor.collate_batch(teacher_processed_lidar_list)
                # output_dict['ego'].update({'teacher_processed_lidar':teacher_processed_lidar_torch_dict})
                for modality_name in self.modality_name_list:
                    if len(eval(f"inputs_list_{modality_name}_proj")) != 0 and self.sensor_type_dict[
                        modality_name] == "lidar":
                        merged_feature_proj_dict = merge_features_to_dict(eval(f"inputs_list_{modality_name}_proj"))
                        processed_lidar_torch_proj_dict = \
                            eval(f"self.pre_processor_{modality_name}").collate_batch(merged_feature_proj_dict)
                        output_dict['ego'].update({f'inputs_{modality_name}_proj': processed_lidar_torch_proj_dict})

            if self.supervise_single or self.heterogeneous:
                output_dict['ego'].update({
                    "label_dict_single": {
                        "pos_equal_one": torch.cat(pos_equal_one_single, dim=0),
                        "neg_equal_one": torch.cat(neg_equal_one_single, dim=0),
                        "targets": torch.cat(targets_single, dim=0),
                        # for centerpoint
                        "object_bbx_center_single": torch.cat(object_bbx_center_single, dim=0),
                        "object_bbx_mask_single": torch.cat(object_bbx_mask_single, dim=0)
                    },
                    "object_bbx_center_single": torch.cat(object_bbx_center_single, dim=0),
                    "object_bbx_mask_single": torch.cat(object_bbx_mask_single, dim=0)
                })

            return output_dict

        def collate_batch_test(self, batch):
            assert len(batch) <= 1, "Batch size 1 is required during testing!"
            if batch[0] is None:
                return None
            output_dict = self.collate_batch_train(batch)
            if output_dict is None:
                return None

            # check if anchor box in the batch
            if batch[0]['ego']['anchor_box'] is not None:
                output_dict['ego'].update({'anchor_box': self.anchor_box_torch})

            # save the transformation matrix (4, 4) to ego vehicle
            # transformation is only used in post process (no use.)
            # we all predict boxes in ego coord.
            transformation_matrix_torch = torch.from_numpy(np.identity(4)).float()
            transformation_matrix_clean_torch = torch.from_numpy(np.identity(4)).float()

            output_dict['ego'].update({'transformation_matrix': transformation_matrix_torch,
                                       'transformation_matrix_clean': transformation_matrix_clean_torch})

            output_dict['ego'].update({
                "sample_idx": batch[0]['ego']['sample_idx'],
                "cav_id_list": batch[0]['ego']['cav_id_list'],
                "agent_modality_list": batch[0]['ego']['agent_modality_list']
            })

            return output_dict

        def post_process(self, data_dict, output_dict):
            """
            Process the outputs of the model to 2D/3D bounding box.

            Parameters
            ----------
            data_dict : dict
                The dictionary containing the origin input data of model.

            output_dict :dict
                The dictionary containing the output of the model.

            Returns
            -------
            pred_box_tensor : torch.Tensor
                The tensor of prediction bounding box after NMS.
            gt_box_tensor : torch.Tensor
                The tensor of gt bounding box.
            """
            pred_box_tensor, pred_score = self.post_processor.post_process(data_dict, output_dict)
            gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)

            return pred_box_tensor, pred_score, gt_box_tensor

        def _get_legal_cav_ids(self, base_data_dict: Mapping, ego_lidar_pose: list):
            """
            获取合法的 cav id. 合法的含义: 在 ego 通信范围内, 被分配过模态
            """
            # 分别存储合法的和不合法的 cav id
            cav_id_list, exclude_agent = [], []
            # 分别存储合法的 cav 对应的 无噪声 lidar pose 和 有噪声 lidar pose
            lidar_pose_clean_list, lidar_pose_list = [], []

            for cav_id, selected_cav_base in base_data_dict.items():
                # 使用勾股定理来判断是否在通信范围内check if the cav is within the communication range with ego
                distance = math.sqrt((selected_cav_base['params']['lidar_pose'][0] - ego_lidar_pose[0]) ** 2 +
                                     (selected_cav_base['params']['lidar_pose'][1] - ego_lidar_pose[1]) ** 2)
                # if distance is too far, we will just skip this agent
                if distance > self.params['comm_range']:
                    exclude_agent.append(cav_id)
                    continue
                # if modality not match
                if self.adaptor.unmatched_modality(selected_cav_base['modality_name']):
                    exclude_agent.append(cav_id)
                    continue
                lidar_pose_clean_list.append(selected_cav_base['params']['lidar_pose_clean'])
                lidar_pose_list.append(selected_cav_base['params']['lidar_pose'])  # 6dof pose
                cav_id_list.append(cav_id)
            return cav_id_list, exclude_agent, lidar_pose_clean_list, lidar_pose_list

        def _merge_cav_features(self, base_data_dict, cav_id_list, ego_cav_base, input_list_dict,
                                projected_lidar_stack=None, input_list_proj_dict=None):
            agent_modality_list = []
            object_stack, object_id_stack = [], []
            single_label_list, single_object_bbx_center_list, single_object_bbx_mask_list = [], [], []
            for _i, cav_id in enumerate(cav_id_list):
                selected_cav_base = base_data_dict[cav_id]
                modality_name = selected_cav_base['modality_name']
                sensor_type = self.sensor_type_dict[selected_cav_base['modality_name']]

                # dynamic object center generator! for heterogeneous input
                if not self.visualize:
                    self.generate_object_center = eval(f"self.generate_object_center_{sensor_type}")
                # need discussion. In test phase, use lidar label.
                else:
                    self.generate_object_center = self.generate_object_center_lidar

                selected_cav_processed = self.get_item_single_car(selected_cav_base, ego_cav_base)

                object_stack.append(selected_cav_processed['object_bbx_center'])
                object_id_stack += selected_cav_processed['object_ids']

                if sensor_type == "lidar":
                    input_list_dict[f'input_list_{modality_name}'].append(
                        selected_cav_processed[f"processed_features_{modality_name}"])

                elif sensor_type == "camera":
                    input_list_dict[f"input_list_{modality_name}"].append(
                        selected_cav_processed[f"image_inputs_{modality_name}"])
                else:
                    raise

                agent_modality_list.append(modality_name)
                """
                这部分应该还是和可视化有关的, 所以还是不用看
                """
                if self.visualize or self.kd_flag:
                    # heterogeneous setting do not support disconet' kd
                    projected_lidar_stack.append(selected_cav_processed['projected_lidar'])
                    if sensor_type == "lidar" and self.kd_flag:
                        input_list_proj_dict[f"input_list_{modality_name}_proj"].append(
                            selected_cav_processed[f"processed_features_{modality_name}_proj"])

                if self.supervise_single or self.heterogeneous:
                    single_label_list.append(selected_cav_processed['single_label_dict'])
                    single_object_bbx_center_list.append(selected_cav_processed['single_object_bbx_center'])
                    single_object_bbx_mask_list.append(selected_cav_processed['single_object_bbx_mask'])

            return agent_modality_list, object_stack, object_id_stack, \
                single_label_list, single_object_bbx_center_list, single_object_bbx_mask_list

    return IntermediateheterFusionDataset
