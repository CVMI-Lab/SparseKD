from numpy import dtype
import torch
import torch.nn as nn
import numpy as np

from pcdet.ops.iou3d_nms import iou3d_nms_utils
from pcdet.config import cfg
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils


class KDHeadTemplate(object):
    def __init__(self, model_cfg, dense_head):
        self.model_cfg = model_cfg
        self.teacher_topk_channel_idx = None
        self.dense_head = dense_head
        self.point_cloud_range = dense_head.point_cloud_range
        self.voxel_size = dense_head.voxel_size
        self.feature_map_stride = dense_head.feature_map_stride

        self.need_nms = cfg.KD.NMS_CONFIG.ENABLED
        self.nms_config = cfg.KD.NMS_CONFIG
        self.roi_head = None

    def build_loss(self, dense_head):
        if not dense_head.is_teacher and self.model_cfg.get('KD_LOSS', None):
            if self.model_cfg.get('LOGIT_KD', None) and self.model_cfg.LOGIT_KD.ENABLED:
                self.build_logit_kd_loss()

            if self.model_cfg.get('FEATURE_KD', None) and self.model_cfg.FEATURE_KD.ENABLED:
                self.build_feature_kd_loss()

            if self.model_cfg.get('VFE_KD', None):
                self.build_vfe_kd_loss()

            if self.model_cfg.get('ROI_KD', None):
                self.build_roi_kd_loss()

    @staticmethod
    def interpolate_feature_map(feature_map, output_shape, interp_channel, interp_mode='bilinear', **kwargs):
        """
        Args:
            feature_map: [B, C, H, W]
            output_shape: [B, C1, H1, W1]
            interp_channel:
            interp_mode:

        Returns:

        """
        if feature_map.shape != output_shape:
            new_feature_map = feature_map.unsqueeze(1) if interp_channel else feature_map
            output_size = output_shape[1:] if interp_channel else output_shape[2:]

            new_feature_map = nn.functional.interpolate(
                new_feature_map, output_size, mode=interp_mode
            )
            # if align channel
            new_feature_map = new_feature_map.squeeze(1)
        else:
            new_feature_map = feature_map

        return new_feature_map

    @staticmethod
    def filter_boxes_by_iou(boxes_stu, boxes_tea, loss_cfg):
        """
        Args:
            boxes_stu: [N, 7]
            boxes_tea: [M, 8]
            loss_cfg

        Returns:
            stu_mask
        """
        if boxes_stu.shape[0] == 0 or boxes_tea.shape[0] == 0:
            return [], []

        iou_matrix = iou3d_nms_utils.boxes_iou3d_gpu(boxes_stu[:, :7], boxes_tea[:, :7]).cpu()

        ious, match_idx = torch.max(iou_matrix, dim=1)
        iou_mask = (ious >= loss_cfg.PRED_FILTER.iou_thresh)

        return iou_mask, match_idx[iou_mask]

    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        assert dim != -1
        rad_pred_encoding = torch.sin(boxes1[..., dim:dim + 1]) * torch.cos(boxes2[..., dim:dim + 1])
        rad_tg_encoding = torch.cos(boxes1[..., dim:dim + 1]) * torch.sin(boxes2[..., dim:dim + 1])
        boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding, boxes1[..., dim + 1:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding, boxes2[..., dim + 1:]], dim=-1)
        return boxes1, boxes2

    @staticmethod
    def bn_channel_align(feature_map, **kwargs):
        """
        select important channels in teacher feature map
        Args:
            feature_map: [B, C, H, W]
            output_shape: [B, C1, H, W]
            align_cfg:
            **kwargs:

        Returns:

        """
        channel_idx = kwargs['channel_idx']

        return feature_map[:, channel_idx, ...]

    @staticmethod
    def conv_channel_align(feature_map, output_shape, dense_head, align_cfg, **kwargs):
        """

        Args:
            feature_map:
            output_shape:
            dense_head:
            align_cfg:
            **kwargs:

        Returns:

        """
        assert align_cfg.target == 'teacher'
        assert feature_map.shape[1] < output_shape[1]
        assert feature_map.shape[2:] == output_shape[2:]

        new_feature_map = dense_head.align_block(feature_map)

        return new_feature_map

    def align_feature_map(self, feature_tea, feature_stu, align_cfg):
        """
        Args:
            feature_tea: [B, C1, H1, W1]
            feature_stu: [B, C2, H2, W2]
            align_cfg:

        Returns:

        """
        if feature_tea.shape == feature_stu.shape:
            return feature_tea, feature_stu

        target = align_cfg.target
        if target == 'teacher':
            output_shape = feature_tea.shape
        elif target == 'student':
            output_shape = feature_stu.shape
        else:
            raise NotImplementedError

        if align_cfg.MODE == 'interpolate':
            align_func = self.interpolate_feature_map
        elif align_cfg.MODE == 'bn':
            align_func = self.bn_channel_align
        elif align_cfg.MODE == 'conv':
            align_func = self.conv_channel_align
        else:
            raise NotImplementedError

        if feature_tea.shape != output_shape:
            new_feature_tea = align_func(
                feature_tea, output_shape,
                align_cfg=align_cfg,
                interp_channel=align_cfg.align_channel,
                interp_mode=align_cfg.mode,
                dense_head=self.dense_head
            )
        else:
            new_feature_tea = feature_tea

        if feature_stu.shape != output_shape:
            new_feature_stu = align_func(
                feature_stu, output_shape,
                align_cfg=align_cfg,
                interp_channel=align_cfg.align_channel,
                interp_mode=align_cfg.mode,
                dense_head=self.dense_head,
                channel_idx=self.teacher_topk_channel_idx
            )
        else:
            new_feature_stu = feature_stu

        return new_feature_tea, new_feature_stu

    def cal_fg_mask_from_gt_boxes_and_spatial_mask(self, gt_boxes, spatial_mask):
        """
        Args:
            gt_boxes: [B, N, 7]
            spatial_mask: [B, height, width]; feature mask after map to bev

        Returns:

        """
        bs, height, width = spatial_mask.shape
        fg_mask = torch.zeros([bs, height, width], dtype=torch.float32).cuda()

        # change to lidar coordinate
        gt_boxes_z0 = gt_boxes.clone()
        gt_boxes_z0[..., 2] = 0

        voxel_size_xy = torch.from_numpy(np.array(self.voxel_size_tea[:2])).float().cuda()
        min_point_cloud_range_xy = torch.from_numpy(self.point_cloud_range[0:2]).float().cuda()
        for b_idx in range(bs):
            # [N, 2], x, y
            valid_coord = spatial_mask[b_idx].nonzero()[:, [1, 0]]
            point_coord_xy = valid_coord.float() * voxel_size_xy * 8 + min_point_cloud_range_xy + \
                          0.5 * voxel_size_xy

            # import ipdb; ipdb.set_trace(context=20)
            # assert point_coord_xy[:, 0].max() < self.point_cloud_range[3] and point_coord_xy[:, 1].min() > self.point_cloud_range[0]
            # assert point_coord_xy.max() < self.point_cloud_range[3] and point_coord_xy.min() > self.point_cloud_range[0]
            point_coord_xyz = torch.cat([point_coord_xy, torch.zeros((point_coord_xy.shape[0], 1)).cuda()], dim=-1)

            # valid gt_boxes mask
            cur_gt_boxes = gt_boxes_z0[b_idx]
            valid_gt_boxes_mask = cur_gt_boxes[:, 3] > 0
            cur_gt_boxes = cur_gt_boxes[valid_gt_boxes_mask]

            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                point_coord_xyz.unsqueeze(0),
                cur_gt_boxes[:, :7].unsqueeze(0)
            ).long().squeeze(dim=0)

            valid_voxel_mask = box_idxs_of_pts != -1
            fg_voxel_coord = valid_coord[valid_voxel_mask]

            fg_mask[b_idx, fg_voxel_coord[:, 1], fg_voxel_coord[:, 0]] = 1

        return fg_mask

    def parse_teacher_pred_to_targets(self, kd_cfg, pred_boxes_tea, gt_boxes):
        """

        Args:
            kd_cfg:
            pred_boxes_tea: len batch_size
                pred_scores: [M]
                pred_boxes: [M, 7]
                pred_labels: [M]
            gt_boxes: [B, N, 8]. (x, y, z, dx, dy, dz, angle, label)

        Returns:

        """
        max_obj = 0
        target_boxes_list = []
        num_target_boxes_list = []
        batch_size = gt_boxes.shape[0]
        for bs_idx in range(batch_size):
            cur_pred_tea = pred_boxes_tea[bs_idx]
            score_mask = self.cal_mask_for_teacher_pred_boxes(
                kd_cfg, cur_pred_tea, gt_boxes[bs_idx]
            )

            target_boxes = torch.cat(
                [cur_pred_tea['pred_boxes'][score_mask],
                 cur_pred_tea['pred_labels'][score_mask, None].float()],
                dim=-1
            )
            num_target_boxes_list.append(target_boxes.shape[0])

            if kd_cfg.USE_GT:
                cur_gt_boxes = gt_boxes[bs_idx]
                valid_mask = cur_gt_boxes[:, 3] > 0
                if kd_cfg.get('GT_FIRST', None):
                    target_boxes = torch.cat([cur_gt_boxes[valid_mask], target_boxes], dim=0)
                else:
                    target_boxes = torch.cat([target_boxes, cur_gt_boxes[valid_mask]], dim=0)

            max_obj = max(max_obj, target_boxes.shape[0])
            target_boxes_list.append(target_boxes)

        final_boxes = torch.zeros((batch_size, max_obj, 8), dtype=torch.float32).cuda()
        for idx, target_boxes in enumerate(target_boxes_list):
            final_boxes[idx, :target_boxes.shape[0]] = target_boxes

        return final_boxes, num_target_boxes_list

    @staticmethod
    def cal_mask_for_teacher_pred_boxes(kd_cfg, cur_pred_tea, gt_boxes):
        """
        Mask part of teacher predicted boxes

        Returns:

        """
        score_type = kd_cfg.get('SCORE_TYPE', 'cls')
        pred_labels = cur_pred_tea['pred_labels']
        labels_remove_thresh = torch.from_numpy(np.array(kd_cfg.SCORE_THRESH)[pred_labels.cpu().numpy() - 1]).cuda().float()
        if score_type == 'cls':
            pred_scores = cur_pred_tea['pred_scores']
            score_mask = pred_scores > labels_remove_thresh
        elif score_type == 'iou':
            valid_mask = gt_boxes[:, 3] > 0
            valid_gt_boxes = gt_boxes[valid_mask]
            iou_matrix = iou3d_nms_utils.boxes_iou3d_gpu(
                cur_pred_tea['pred_boxes'][:, :7], valid_gt_boxes[:, :7]
            )
            ious, _ = torch.max(iou_matrix, dim=1)
            score_mask = ious > labels_remove_thresh
        elif score_type == 'mixed': #  iou for vehicle and cls for ped and cyclist
            score_mask = labels_remove_thresh.new_zeros(labels_remove_thresh.shape, dtype=torch.float32).cuda()
            # for vehicle, use iou as criterion
            gt_car_mask = gt_boxes[:, -1] == 1
            gt_boxes_car = gt_boxes[gt_car_mask]
            pred_tea_car_mask = pred_labels == 1
            iou_matrix = iou3d_nms_utils.boxes_iou3d_gpu(
                cur_pred_tea['pred_boxes'][pred_tea_car_mask, :7], gt_boxes_car[:, :7]
            )
            ious, _ = torch.max(iou_matrix, dim=1)
            car_score_mask = (ious > kd_cfg.SCORE_THRESH[0]).float()
            score_mask[pred_tea_car_mask] = car_score_mask

            # for ped and cyclist, use cls score
            pred_scores = cur_pred_tea['pred_scores']
            pred_tea_ped_cyc_mask = pred_labels != 1
            ped_cyc_score_mask = (pred_scores[pred_tea_ped_cyc_mask] > labels_remove_thresh[pred_tea_ped_cyc_mask]).float()
            score_mask[pred_tea_ped_cyc_mask] = ped_cyc_score_mask
            score_mask = score_mask.byte()
        else:
            raise NotImplementedError

        return score_mask

    def sigmoid(self, x):
        y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
        return y

    @staticmethod
    def cal_fg_mask_from_gt_anchors(box_cls_labels, anchor_shape, num_anchor, count_ignore=True):
        bs, height, width, _ = anchor_shape

        box_cls_labels_hw = box_cls_labels.view(bs, height, width, num_anchor)
        if count_ignore:
            box_cls_labels_hw_pos = box_cls_labels_hw != 0
            fg_mask = box_cls_labels_hw_pos.sum(dim=-1) > 0
        else:
            box_cls_labels_hw_pos = box_cls_labels_hw > 0
            fg_mask = box_cls_labels_hw_pos.sum(dim=-1) > 0

        return fg_mask
