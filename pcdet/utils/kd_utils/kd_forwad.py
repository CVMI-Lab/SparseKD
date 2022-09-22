import torch
from torch.nn.utils import clip_grad_norm_

from pcdet.config import cfg
from pcdet.utils import common_utils
from pcdet.models.dense_heads import CenterHead, AnchorHeadTemplate


def adjust_batch_info_teacher(batch):
    if cfg.KD.get('DIFF_VOXEL', None):
        batch['voxels_stu'] = batch.pop('voxels')
        batch['voxel_coords_stu'] = batch.pop('voxel_coords')
        batch['voxel_num_points_stu'] = batch.pop('voxel_num_points')

        batch['voxels'] = batch.pop('voxels_tea')
        batch['voxel_coords'] = batch.pop('voxel_coords_tea')
        batch['voxel_num_points'] = batch.pop('voxel_num_points_tea')

    teacher_pred_flag = False
    teacher_target_dict_flag = False
    teacher_decoded_pred_flag = False

    # LOGIT KD
    if cfg.KD.get('LOGIT_KD', None) and cfg.KD.LOGIT_KD.ENABLED:
        if cfg.KD.LOGIT_KD.MODE in ['raw_pred', 'target']:
            teacher_pred_flag = True
            teacher_target_dict_flag = True
        elif cfg.KD.LOGIT_KD.MODE == 'decoded_boxes':
            teacher_decoded_pred_flag = True
        else:
            raise NotImplementedError

    if cfg.KD.get('LABEL_ASSIGN_KD', None) and cfg.KD.LABEL_ASSIGN_KD.ENABLED:
        teacher_decoded_pred_flag = True
    
    if cfg.KD.get('MASK', None):
        if cfg.KD.MASK.get('FG_MASK', None):
            teacher_target_dict_flag = True
        
        if cfg.KD.MASK.get('BOX_MASK', None):
            teacher_decoded_pred_flag = True
        
        if cfg.KD.MASK.get('SCORE_MASK', None):
            teacher_pred_flag = True

    batch['teacher_pred_flag'] = teacher_pred_flag
    batch['teacher_target_dict_flag'] = teacher_target_dict_flag
    batch['teacher_decoded_pred_flag'] = teacher_decoded_pred_flag


def adjust_batch_info_student(batch):
    if cfg.KD.get('DIFF_VOXEL', None):
        del batch['voxels']
        del batch['voxel_coords']
        del batch['voxel_num_points']

        batch['voxels'] = batch.pop('voxels_stu')
        batch['voxel_coords'] = batch.pop('voxel_coords_stu')
        batch['voxel_num_points'] = batch.pop('voxel_num_points_stu')


def add_teacher_pred_to_batch(teacher_model, batch, pred_dicts=None):
    if cfg.KD.get('FEATURE_KD', None) and cfg.KD.FEATURE_KD.ENABLED:
        feature_name = cfg.KD.FEATURE_KD.get('FEATURE_NAME_TEA', cfg.KD.FEATURE_KD.FEATURE_NAME)
        batch[feature_name + '_tea'] = batch[feature_name].detach()

    if cfg.KD.get('PILLAR_KD', None) and cfg.KD.PILLAR_KD.ENABLED:
        feature_name_tea = cfg.KD.PILLAR_KD.FEATURE_NAME_TEA
        batch['voxel_features_tea'] = batch.pop(feature_name_tea)

    if cfg.KD.get('VFE_KD', None) and cfg.KD.VFE_KD.ENABLED:
        batch['point_features_tea'] = batch.pop('point_features')
        batch['pred_tea'] = teacher_model.dense_head.forward_ret_dict['pred_dicts']
        if cfg.KD.VFE_KD.get('SAVE_INDS', None):
            batch['unq_inv_pfn_tea'] = batch.pop('unq_inv_pfn')
        if cfg.KD.VFE_KD.get('SAVE_3D_FEAT', None):
            batch['spatial_features_tea'] = batch.pop('spatial_features')

    if cfg.KD.get('ROI_KD', None) and cfg.KD.ROI_KD.ENABLED:
        batch['rcnn_cls_tea'] = teacher_model.roi_head.forward_ret_dict.pop('rcnn_cls')
        batch['rcnn_reg_tea'] = teacher_model.roi_head.forward_ret_dict.pop('rcnn_reg')
        batch['roi_head_target_dict_tea'] = teacher_model.roi_head.forward_ret_dict

    if cfg.KD.get('SAVE_COORD_TEA', None):
        batch['voxel_coords_tea'] = batch.pop('voxel_coords')
    
    if batch.get('teacher_target_dict_flag', None):
        if isinstance(teacher_model.dense_head, CenterHead):
            batch['target_dicts_tea'] = teacher_model.dense_head.forward_ret_dict['target_dicts']
        elif isinstance(teacher_model.dense_head, AnchorHeadTemplate):
            batch['spatial_mask_tea'] = batch['spatial_features'].sum(dim=1) != 0

    if batch.get('teacher_pred_flag', None):
        if isinstance(teacher_model.dense_head, CenterHead):
            batch['pred_tea'] = teacher_model.dense_head.forward_ret_dict['pred_dicts']
        elif isinstance(teacher_model.dense_head, AnchorHeadTemplate):
            batch['cls_preds_tea'] = teacher_model.dense_head.forward_ret_dict['cls_preds']
            batch['box_preds_tea'] = teacher_model.dense_head.forward_ret_dict['box_preds']
            batch['dir_cls_preds_tea'] = teacher_model.dense_head.forward_ret_dict['dir_cls_preds']

    if batch.get('teacher_decoded_pred_flag', None):
        if (not teacher_model.training) and teacher_model.roi_head is not None:
            batch['decoded_pred_tea'] = pred_dicts
        elif isinstance(teacher_model.dense_head, CenterHead):
            batch['decoded_pred_tea'] = teacher_model.dense_head.forward_ret_dict['decoded_pred_dicts']
        elif isinstance(teacher_model.dense_head, AnchorHeadTemplate):
            batch['decoded_pred_tea'] = pred_dicts


def forward(model, teacher_model, batch, optimizer, extra_optim, optim_cfg, load_data_to_gpu, **kwargs):
    optimizer.zero_grad()
    if extra_optim is not None:
        extra_optim.zero_grad()

    with torch.no_grad():
        adjust_batch_info_teacher(batch)
        load_data_to_gpu(batch)
        if teacher_model.training:
            batch = teacher_model(batch)
            pred_dicts = None
        else:
            pred_dicts, ret_dict = teacher_model(batch)
        add_teacher_pred_to_batch(teacher_model, batch, pred_dicts=pred_dicts)

    adjust_batch_info_student(batch)

    ret_dict, tb_dict, disp_dict = model(batch)
    loss = ret_dict['loss'].mean()

    loss.backward()
    clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)

    optimizer.step()
    if extra_optim is not None:
        extra_optim.step()

    return loss, tb_dict, disp_dict
