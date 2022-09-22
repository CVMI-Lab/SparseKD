import torch
import numpy as np

from pcdet.ops.iou3d_nms import iou3d_nms_utils
from pcdet.models.kd_heads.kd_head import KDHeadTemplate


class CenterLabelAssignKDHead(KDHeadTemplate):
    def __init__(self, model_cfg, dense_head):
        super().__init__(model_cfg, dense_head)

    @staticmethod
    def filter_inds_for_regression_tea(target_dict, num_target_boxes_list, kd_cfg):
        """
        Filter same center voxel targets in teacher predicted assigned labels.
        This is mainly used to analysis the effect of label assign kd in regression loss.
        
        Three keys need to be filtered: target_boxes, inds, masks
        
        Args:
            target_dict:
                target_boxes:
                inds:
                masks:
            num_target_boxes_list: number of teacher assgined boxes in each scene
            
        Returns:

        
        """
        h_idx = 0
        new_num_tea_boxes_list = []
        for b_idx, num_tea_boxes in enumerate(num_target_boxes_list):
            inds = target_dict['inds'][h_idx][b_idx]
            masks = target_dict['masks'][h_idx][b_idx]

            num_boxes = masks.sum()

            # find the index of teacher boxes
            if kd_cfg.GT_FIRST:
                start_idx = num_boxes - num_tea_boxes
                boxes_idx_tea = torch.arange(start_idx, num_boxes).cuda()
            else:
                boxes_idx_tea = torch.arange(0, num_tea_boxes).cuda()
            
            unique_inds, unq_inv, unq_cnt = torch.unique(
                inds[:num_boxes], return_inverse=True, return_counts=True
            )

            all_cnts = unq_cnt[unq_inv]
            tea_cnts = all_cnts[boxes_idx_tea]

            # find the intra duplicate between teacher boxes
            tea_inds = inds[boxes_idx_tea]
            intra_unq_inds, intra_unq_inv, intra_unq_cnts = torch.unique(
                tea_inds, return_inverse=True, return_counts=True
            )

            intra_tea_cnts = intra_unq_cnts[intra_unq_inv]

            # find the duplicated boxes inside 
            invalid_tea_mask = (tea_cnts - intra_tea_cnts) > 0 
            new_num_tea = invalid_tea_mask.shape[0] - invalid_tea_mask.sum()

            invalid_tea_idx = boxes_idx_tea[invalid_tea_mask]

            target_dict['masks'][h_idx][b_idx, invalid_tea_idx] = False

            new_num_tea_boxes_list.append(new_num_tea)

        return target_dict, new_num_tea_boxes_list

    @staticmethod
    def remove_tea_boxes_in_target_dicts(target_dict, num_target_boxes_list, kd_cfg):
        h_idx = 0
        for b_idx, num_tea_boxes in enumerate(num_target_boxes_list):
            masks = target_dict['masks'][h_idx][b_idx]

            num_boxes = masks.sum()

            if kd_cfg.GT_FIRST:
                start_idx = num_boxes - num_tea_boxes
                boxes_idx_tea = torch.arange(start_idx, num_boxes).cuda()
            else:
                boxes_idx_tea = torch.arange(0, num_tea_boxes).cuda()

            target_dict['masks'][h_idx][b_idx, boxes_idx_tea] = False

        return target_dict

    @staticmethod
    def filter_inds_for_regression_gt(target_dict, num_target_boxes_list, kd_cfg, replace_only=False):
        """
        Filter same center voxel targets in teacher predicted assigned labels.
        This is mainly used to analysis the effect of label assign kd in regression loss.
        
        Three keys need to be filtered: target_boxes, inds, masks
        
        Args:
            target_dict:
                target_boxes:
                inds:
                masks:
            num_target_boxes_list: number of teacher assgined boxes in each scene
            
        Returns:

        
        """
        h_idx = 0
        for b_idx, num_tea_boxes in enumerate(num_target_boxes_list):
            inds = target_dict['inds'][h_idx][b_idx]
            masks = target_dict['masks'][h_idx][b_idx]

            num_boxes = masks.sum()

            # find the index of teacher boxes
            if kd_cfg.GT_FIRST:
                start_idx = num_boxes - num_tea_boxes
                gt_boxes_idx = torch.arange(0, start_idx).cuda()
                boxes_idx_tea = torch.arange(start_idx, num_boxes).cuda()
            else:
                boxes_idx_tea = torch.arange(0, num_tea_boxes).cuda()
                gt_boxes_idx = torch.arange(num_tea_boxes, num_boxes).cuda()
            
            _, unq_inv, unq_cnt = torch.unique(
                inds[:num_boxes], return_inverse=True, return_counts=True
            )

            all_cnts = unq_cnt[unq_inv]
            gt_cnts = all_cnts[gt_boxes_idx]

            invalid_gt_mask = gt_cnts > 1 
            invalid_gt_idx = gt_boxes_idx[invalid_gt_mask]
            target_dict['masks'][h_idx][b_idx, invalid_gt_idx] = False

            if replace_only:
                ind_clone = inds.clone()
                ind_clone[invalid_gt_idx] = -1
                tea_cnts = all_cnts[boxes_idx_tea]
                _, unq_inv2, unq_cnt2 = torch.unique(
                    ind_clone[:num_boxes], return_inverse=True, return_counts=True
                )
                all_cnts2 = unq_cnt2[unq_inv2]
                tea_cnts2 = all_cnts2[boxes_idx_tea]
                # do not repeat with gt
                invalid_tea_mask = (tea_cnts2 == tea_cnts)
                invalid_tea_idx = boxes_idx_tea[invalid_tea_mask]
                target_dict['masks'][h_idx][b_idx, invalid_tea_idx] = False

        return target_dict
