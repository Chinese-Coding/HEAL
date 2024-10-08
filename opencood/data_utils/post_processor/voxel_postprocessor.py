# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, OpenPCDet
# License: TDG-Attribution-NonCommercial-NoDistrib


"""
3D Anchor Generator for Voxel
"""
import math
import sys

import numpy as np
import torch
import torch.nn.functional as F
from opencood.utils.box_overlaps import bbox_overlaps

from opencood.data_utils.post_processor.base_postprocessor import BasePostprocessor
from opencood.utils import box_utils
from opencood.utils.common_utils import limit_period
from opencood.visualization import vis_utils


class VoxelPostprocessor(BasePostprocessor):
    def __init__(self, anchor_params, train):
        super().__init__(anchor_params, train)
        self.anchor_num = self.params['anchor_args']['num']  # 调试值：2

    def generate_anchor_box(self):
        """
        TODO: 这个函数应该好好调试看一下
        """
        # load_voxel_params and load_point_pillar_params leads to the same anchor
        # if voxel_size * feature stride is the same.
        W = self.params['anchor_args']['W']
        H = self.params['anchor_args']['H']

        l = self.params['anchor_args']['l']
        w = self.params['anchor_args']['w']
        h = self.params['anchor_args']['h']
        r = self.params['anchor_args']['r']  # 调试值：[0, 90], 指的是生成框的框的角度吗？

        assert self.anchor_num == len(r)
        r = [math.radians(ele) for ele in r]  # radians 弧度，将 r 从角度转换为弧度表示

        # voxel_size
        vh = self.params['anchor_args']['vh']  # voxel_height
        vw = self.params['anchor_args']['vw']  # voxel_width

        # `cav_lidar_range` 调试得来：[-102.4, -51.2, -3, 102.4, 51.2, 1]
        xrange = [self.params['anchor_args']['cav_lidar_range'][0],
                  self.params['anchor_args']['cav_lidar_range'][3]]
        yrange = [self.params['anchor_args']['cav_lidar_range'][1],
                  self.params['anchor_args']['cav_lidar_range'][4]]

        if 'feature_stride' in self.params['anchor_args']:
            feature_stride = self.params['anchor_args']['feature_stride']
        else:
            feature_stride = 2

        # vw is not precise, vw * feature_stride / 2 should be better?
        # np.linspace 生成均匀分布的数值序列 TODO： 如果说保持配置文件一致，每次生成的 x, y 是否都是一样的呢
        x = np.linspace(xrange[0] + vw, xrange[1] - vw, W // feature_stride)
        y = np.linspace(yrange[0] + vh, yrange[1] - vh, H // feature_stride)

        """
        `np.meshgrid` 生成网格, 可用于生成坐标
        cx, cy 的长度为 (W // feature_stride) * (H // feature_stride).
        实际上就是 x, y 里面的东西做一个笛卡尔积, 然后返回成两个变量， 分别存储着生成点的的横纵坐标
        np.tile 在原有维度上新扩展一个维度，扩展的维度中包含着 `self.anchor_num` 个元素, 也许意味着在同一个中心中生成对应数量的框
        
        cz 的含义又是什么呢? 如果是在 BEV 图中, 不应该存在 z 轴啊
        """
        cx, cy = np.meshgrid(x, y)
        cx = np.tile(cx[..., np.newaxis], self.anchor_num)  # center
        cy = np.tile(cy[..., np.newaxis], self.anchor_num)
        cz = np.ones_like(cx) * -1.0

        # 维度匹配, 生成匹配框中心点个数的 ndarray
        w = np.ones_like(cx) * w
        l = np.ones_like(cx) * l
        h = np.ones_like(cx) * h

        r_ = np.ones_like(cx)
        for i in range(self.anchor_num):
            r_[..., i] = r[i]

        # 按照传入的配置文件的规则, 生成对应的 anchor box
        if self.params['order'] == 'hwl':  # pointpillar
            anchors = np.stack([cx, cy, cz, h, w, l, r_], axis=-1)  # (50, 176, 2, 7)
        elif self.params['order'] == 'lhw':
            anchors = np.stack([cx, cy, cz, l, h, w, r_], axis=-1)
        else:
            sys.exit('Unknown bbx order.')

        """
        调试：anchors.shape = [128, 256, 2, 7], 具体来看就是, 总共生成了 125 * 256 个点, 
        每个点上有两个框, 每个框又由 7 个元素组成. 而且, 只要配置文件相同, 每次生成的框应该是一模一样的
        """
        return anchors

    def generate_label(self, **kwargs):
        """
        Generate targets for training.
        函数的目的是生成训练过程中所需的正样本标签、负样本标签和目标回归值

        Parameters
        ----------
        argv : list
            gt_box_center:(max_num, 7), anchor:(H, W, anchor_num, 7)

        Returns
        -------
        label_dict : dict
            Dictionary that contains all target related info.
        """
        assert self.params['order'] == 'hwl', 'Currently Voxel only support hwl bbx order.'

        # (max_num, 7)
        gt_box_center = kwargs['gt_box_center']
        # (H, W, anchor_num, 7)
        anchors = kwargs['anchors']
        # (max_num)
        masks = kwargs['mask']

        # (H, W)
        feature_map_shape = anchors.shape[:2]
        """
        这一步应该算是将 anchors 转换成与 gt_box_center (或者说 masks, gt_box_center 和 masks 是同维的) 同样维度的一步
        shape : (H * W * anchor_num, 7)
        """
        anchors = anchors.reshape(-1, 7)
        # normalization factor, (H * W * anchor_num) TODO: 这一步是干什么, GPT 说是计算 anchor 的对角线的长度?
        anchors_d = np.sqrt(anchors[:, 4] ** 2 + anchors[:, 5] ** 2)

        """
        GPT: 初始化正样本标签、负样本标签和目标回归值的数组. 为什么要有这三个数组呢
        """
        pos_equal_one = np.zeros((*feature_map_shape, self.anchor_num))  # (H, W, 2)
        neg_equal_one = np.zeros((*feature_map_shape, self.anchor_num))  # (H, W, 2)
        targets = np.zeros((*feature_map_shape, self.anchor_num * 7))  # (H, W, self.anchor_num * 7)

        """
        这一部分应该是为了计算 IoU 做准备, 计算方法已经固定了, 没有看的具体必要 
        (如果真的想搞懂, 还是要下一番功夫的, 纯纯数学推理)
        但是根据该类中上面的哪个生成 anchor 的函数, anchors 的 z 轴坐标的没有意义的, 这是用于充数的 (详见该文件的第 76 行附近)
        全部默认为 -1 (调试后发现也是如此)
        """
        gt_box_center_valid = gt_box_center[masks == 1]  # (n, 7) # 提取有效的 gt_box
        gt_box_corner_valid = box_utils.boxes_to_corners_3d(gt_box_center_valid, self.params['order'])  # (n, 8, 3)
        anchors_corner = box_utils.boxes_to_corners_3d(anchors, self.params['order'])  # (H * W * anchor_num, 8, 3)
        # TODO: 根据 corner 坐标, 生成对应的二维坐标, 感觉这里应该是直接把 z 轴的数据给刨除了
        anchors_standup_2d = box_utils.corner2d_to_standup_box(anchors_corner)  # (H * W * anchor_num, 4)
        gt_standup_2d = box_utils.corner2d_to_standup_box(gt_box_corner_valid)  # (n, 4)

        # (H * W * anchor_n)
        iou = bbox_overlaps(
            np.ascontiguousarray(anchors_standup_2d).astype(np.float32),
            np.ascontiguousarray(gt_standup_2d).astype(np.float32),
        )

        """
        GPT: 筛选正负样本
        """
        # the anchor boxes has the largest iou across TODO: 找到每个 ground truth box 对应的最大 IoU 的 anchor
        # shape: (n), # 这里的 n 应该指的是 gt_box 的个数
        # TODO: 这里可能有 bug
        iouT = iou.T
        id_highest = np.argmax(iouT, axis=1)  # `np.argmax()` 求 axis 轴上最大数的索引
        # [0, 1, 2, ..., n-1] # TODO: iouT 第一维度
        id_highest_gt = np.arange(iouT.shape[0])
        # make sure all highest iou is larger than 0
        mask = iouT[id_highest_gt, id_highest] > 0
        id_highest, id_highest_gt = id_highest[mask], id_highest_gt[mask]

        """
        筛选 IoU 大于正样本阈值的 anchors, 以及 IoU小 于负样本阈值的 anchors
        TODO: 以下这部分没有看 (感觉不用看, 都是复杂的数学推导, 如果有公式最好理解, 反之, 如果是从代码反推则难度较大)
        """
        # find anchors iou > params['pos_iou']
        id_pos, id_pos_gt = np.where(iou > self.params['target_args']['pos_threshold'])  # 调试得到: 0.6
        id_neg = np.where(np.sum(iou < self.params['target_args']['neg_threshold'], axis=1) == iou.shape[1])[0]
        id_pos, id_pos_gt = np.concatenate([id_pos, id_highest]), np.concatenate([id_pos_gt, id_highest_gt])
        id_pos, index = np.unique(id_pos, return_index=True)
        id_pos_gt = id_pos_gt[index]
        id_neg.sort()

        # cal the target and set the equal one
        index_x, index_y, index_z = np.unravel_index(id_pos, (*feature_map_shape, self.anchor_num))
        pos_equal_one[index_x, index_y, index_z] = 1

        # calculate the targets
        targets[index_x, index_y, np.array(index_z) * 7] = \
            (gt_box_center[id_pos_gt, 0] - anchors[id_pos, 0]) / anchors_d[id_pos]
        targets[index_x, index_y, np.array(index_z) * 7 + 1] = \
            (gt_box_center[id_pos_gt, 1] - anchors[id_pos, 1]) / anchors_d[id_pos]
        targets[index_x, index_y, np.array(index_z) * 7 + 2] = \
            (gt_box_center[id_pos_gt, 2] - anchors[id_pos, 2]) / anchors[id_pos, 3]

        targets[index_x, index_y, np.array(index_z) * 7 + 3] = np.log(gt_box_center[id_pos_gt, 3] / anchors[id_pos, 3])
        targets[index_x, index_y, np.array(index_z) * 7 + 4] = np.log(gt_box_center[id_pos_gt, 4] / anchors[id_pos, 4])
        targets[index_x, index_y, np.array(index_z) * 7 + 5] = np.log(gt_box_center[id_pos_gt, 5] / anchors[id_pos, 5])
        targets[index_x, index_y, np.array(index_z) * 7 + 6] = (gt_box_center[id_pos_gt, 6] - anchors[id_pos, 6])

        index_x, index_y, index_z = np.unravel_index(id_neg, (*feature_map_shape, self.anchor_num))
        neg_equal_one[index_x, index_y, index_z] = 1

        # to avoid a box be pos/neg in the same time
        index_x, index_y, index_z = np.unravel_index(id_highest, (*feature_map_shape, self.anchor_num))
        neg_equal_one[index_x, index_y, index_z] = 0

        label_dict = {'pos_equal_one': pos_equal_one, 'neg_equal_one': neg_equal_one, 'targets': targets}

        return label_dict

    @staticmethod
    def collate_batch(label_batch_list: list) -> dict:
        """
        collate: 整理
        Customized collate function for target label generation.

        Parameters
        ----------
        label_batch_list : list
            The list of dictionary  that contains all labels for several frames.

        Returns
        -------
        target_batch : dict
            Reformatted labels in torch tensor.
        """
        pos_equal_one = []
        neg_equal_one = []
        targets = []

        for i in range(len(label_batch_list)):
            pos_equal_one.append(label_batch_list[i]['pos_equal_one'])
            neg_equal_one.append(label_batch_list[i]['neg_equal_one'])
            targets.append(label_batch_list[i]['targets'])

        pos_equal_one = torch.from_numpy(np.array(pos_equal_one))
        neg_equal_one = torch.from_numpy(np.array(neg_equal_one))
        targets = torch.from_numpy(np.array(targets))

        return {'targets': targets, 'pos_equal_one': pos_equal_one, 'neg_equal_one': neg_equal_one}

    def post_process(self, data_dict, output_dict):
        """
        Process the outputs of the model to 2D/3D bounding box.
        Step1: convert each cav's output to bounding box format
        Step2: project the bounding boxes to ego space.
        Step:3 NMS

        For early and intermediate fusion,
            data_dict only contains ego.

        For late fusion,
            data_dcit contains all cavs, so we need transformation matrix.


        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.

        output_dict :dict
            The dictionary containing the output of the model.

        Returns
        -------
        pred_box3d_tensor : torch.Tensor
            The prediction bounding box tensor after NMS.
        gt_box3d_tensor : torch.Tensor
            The groundtruth bounding box tensor.
        """
        # the final bounding box list
        pred_box3d_list = []
        pred_box2d_list = []
        for cav_id in output_dict.keys():
            assert cav_id in data_dict
            cav_content = data_dict[cav_id]
            # the transformation matrix to ego space
            transformation_matrix = cav_content['transformation_matrix']  # no clean

            # rename variable
            if 'psm' in output_dict[cav_id]:
                output_dict[cav_id]['cls_preds'] = output_dict[cav_id]['psm']
            if 'rm' in output_dict:
                output_dict[cav_id]['reg_preds'] = output_dict[cav_id]['rm']
            if 'dm' in output_dict:
                output_dict[cav_id]['dir_preds'] = output_dict[cav_id]['dm']

            # (H, W, anchor_num, 7)
            anchor_box = cav_content['anchor_box']

            # classification probability
            prob = output_dict[cav_id]['cls_preds']
            prob = F.sigmoid(prob.permute(0, 2, 3, 1))
            prob = prob.reshape(1, -1)

            # regression map
            reg = output_dict[cav_id]['reg_preds']

            # convert regression map back to bounding box
            if len(reg.shape) == 4:  # anchor-based. PointPillars, SECOND
                batch_box3d = self.delta_to_boxes3d(reg, anchor_box)
            else:  # anchor-free. CenterPoint
                batch_box3d = reg.view(1, -1, 7)

            mask = torch.gt(prob, self.params['target_args']['score_threshold'])
            mask = mask.view(1, -1)
            mask_reg = mask.unsqueeze(2).repeat(1, 1, 7)

            # during validation/testing, the batch size should be 1
            assert batch_box3d.shape[0] == 1
            boxes3d = torch.masked_select(batch_box3d[0], mask_reg[0]).view(-1, 7)
            scores = torch.masked_select(prob[0], mask[0])

            # adding dir classifier
            if 'dir_preds' in output_dict[cav_id].keys() and len(boxes3d) != 0:
                dir_offset = self.params['dir_args']['dir_offset']
                num_bins = self.params['dir_args']['num_bins']

                dm = output_dict[cav_id]['dir_preds']  # [N, H, W, 4]
                dir_cls_preds = dm.permute(0, 2, 3, 1).contiguous().reshape(1, -1, num_bins)  # [1, N*H*W*2, 2]
                dir_cls_preds = dir_cls_preds[mask]
                # if rot_gt > 0, then the label is 1, then the regression target is [0, 1]
                # indices. shape [1, N*H*W*2].  value 0 or 1. If value is 1, then rot_gt > 0
                dir_labels = torch.max(dir_cls_preds, dim=-1)[1]

                period = (2 * np.pi / num_bins)  # pi
                dir_rot = limit_period(boxes3d[..., 6] - dir_offset, 0, period)  # 限制在0到pi之间
                boxes3d[..., 6] = dir_rot + dir_offset + period * dir_labels.to(dir_cls_preds.dtype)  # 转化0.25pi到2.5pi
                boxes3d[..., 6] = limit_period(boxes3d[..., 6], 0.5, 2 * np.pi)  # limit to [-pi, pi]

            if 'iou_preds' in output_dict[cav_id].keys() and len(boxes3d) != 0:
                iou = torch.sigmoid(output_dict[cav_id]['iou_preds'].permute(0, 2, 3, 1).contiguous()).reshape(1, -1)
                iou = torch.clamp(iou, min=0.0, max=1.0)
                iou = (iou + 1) * 0.5
                scores = scores * torch.pow(iou.masked_select(mask), 4)

            # convert output to bounding box
            if len(boxes3d) != 0:
                # (N, 8, 3)
                boxes3d_corner = box_utils.boxes_to_corners_3d(boxes3d, order=self.params['order'])

                # STEP 2
                # (N, 8, 3)
                projected_boxes3d = box_utils.project_box3d(boxes3d_corner, transformation_matrix)
                # convert 3d bbx to 2d, (N,4)
                projected_boxes2d = box_utils.corner_to_standup_box_torch(projected_boxes3d)
                # (N, 5)
                boxes2d_score = torch.cat((projected_boxes2d, scores.unsqueeze(1)), dim=1)

                pred_box2d_list.append(boxes2d_score)
                pred_box3d_list.append(projected_boxes3d)

        if len(pred_box2d_list) == 0 or len(pred_box3d_list) == 0:
            return None, None
        # shape: (N, 5)
        pred_box2d_list = torch.vstack(pred_box2d_list)
        # scores
        scores = pred_box2d_list[:, -1]
        # predicted 3d bbx
        pred_box3d_tensor = torch.vstack(pred_box3d_list)
        # remove large bbx
        keep_index_1 = box_utils.remove_large_pred_bbx(pred_box3d_tensor)
        keep_index_2 = box_utils.remove_bbx_abnormal_z(pred_box3d_tensor)
        keep_index = torch.logical_and(keep_index_1, keep_index_2)

        pred_box3d_tensor = pred_box3d_tensor[keep_index]
        scores = scores[keep_index]

        # STEP3
        # nms
        keep_index = box_utils.nms_rotated(pred_box3d_tensor, scores, self.params['nms_thresh'])

        pred_box3d_tensor = pred_box3d_tensor[keep_index]

        # select cooresponding score
        scores = scores[keep_index]

        # filter out the prediction out of the range. with z-dim
        pred_box3d_np = pred_box3d_tensor.cpu().numpy()
        pred_box3d_np, mask = \
            box_utils.mask_boxes_outside_range_numpy(pred_box3d_np, self.params['gt_range'],
                                                     order=None, return_mask=True)
        pred_box3d_tensor = torch.from_numpy(pred_box3d_np).to(device=pred_box3d_tensor.device)
        scores = scores[mask]

        assert scores.shape[0] == pred_box3d_tensor.shape[0]

        return pred_box3d_tensor, scores

    @staticmethod
    def delta_to_boxes3d(deltas, anchors):
        """
        Convert the output delta to 3d bbx.

        Parameters
        ----------
        deltas : torch.Tensor
            (N, 14, H, W)
        anchors : torch.Tensor
            (W, L, 2, 7) -> xyzhwlr

        Returns
        -------
        box3d : torch.Tensor
            (N, W*L*2, 7)
        """
        # batch size
        N = deltas.shape[0]
        deltas = deltas.permute(0, 2, 3, 1).contiguous().view(N, -1, 7)
        boxes3d = torch.zeros_like(deltas)

        if deltas.is_cuda:
            anchors = anchors.cuda()
            boxes3d = boxes3d.cuda()

        # (W*L*2, 7)
        anchors_reshaped = anchors.view(-1, 7).float()
        # the diagonal of the anchor 2d box, (W*L*2)
        anchors_d = torch.sqrt(anchors_reshaped[:, 4] ** 2 + anchors_reshaped[:, 5] ** 2)
        anchors_d = anchors_d.repeat(N, 2, 1).transpose(1, 2)
        anchors_reshaped = anchors_reshaped.repeat(N, 1, 1)

        # Inv-normalize to get xyz
        boxes3d[..., [0, 1]] = torch.mul(deltas[..., [0, 1]], anchors_d) + anchors_reshaped[..., [0, 1]]
        boxes3d[..., [2]] = torch.mul(deltas[..., [2]], anchors_reshaped[..., [3]]) + anchors_reshaped[..., [2]]
        # hwl
        boxes3d[..., [3, 4, 5]] = torch.exp(deltas[..., [3, 4, 5]]) * anchors_reshaped[..., [3, 4, 5]]
        # yaw angle
        boxes3d[..., 6] = deltas[..., 6] + anchors_reshaped[..., 6]

        return boxes3d

    @staticmethod
    def visualize(pred_box_tensor, gt_tensor, pcd, show_vis, save_path, dataset=None):
        """
        Visualize the prediction, ground truth with point cloud together.

        Parameters
        ----------
        pred_box_tensor : torch.Tensor
            (N, 8, 3) prediction.

        gt_tensor : torch.Tensor
            (N, 8, 3) groundtruth bbx

        pcd : torch.Tensor
            PointCloud, (N, 4).

        show_vis : bool
            Whether to show visualization.

        save_path : str
            Save the visualization results to given path.

        dataset : BaseDataset
            opencood dataset object.

        """
        vis_utils.visualize_single_sample_output_gt(pred_box_tensor, gt_tensor, pcd, show_vis, save_path)
