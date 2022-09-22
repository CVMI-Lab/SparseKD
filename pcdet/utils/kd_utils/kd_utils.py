import torch
from torch.nn.utils import clip_grad_norm_
from pcdet.config import cfg


def process_kd_config():
    """
    put kd related config to model
    Returns:

    """
    # Global dense head indicator.
    # Build KD head or not
    cfg.MODEL.KD = True
    cfg.MODEL_TEACHER.KD = True

    # Only student model have KD_LOSS config
    if cfg.KD_LOSS.ENABLED:
        cfg.MODEL.KD_LOSS = cfg.KD_LOSS

    parse_key_list = ['LOGIT_KD', 'FEATURE_KD', 'LABEL_ASSIGN_KD', 'VFE_KD', 'ROI_KD']

    for key in parse_key_list:
        if cfg.KD.get(key, None) and cfg.KD[key].ENABLED:
            cfg.MODEL.DENSE_HEAD[key] = cfg.KD[key]
            cfg.MODEL[key] = cfg.KD[key]
            cfg.MODEL_TEACHER[key] = cfg.KD[key]

    if cfg.KD.get('VFE_KD', None) and cfg.KD.VFE_KD.ENABLED:
        cfg.MODEL.VFE.VFE_KD = cfg.KD.VFE_KD
        cfg.MODEL_TEACHER.VFE.VFE_KD = cfg.KD.VFE_KD


def prepare_kd_modules(teacher_model, model):
    # add a flag to indicate KD for each module in the detector
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        for cur_module in model.module.module_list:
            cur_module.kd = True
    else:
        for cur_module in model.module_list:
            cur_module.kd = True

    put_teacher_prior_to_student(teacher_model, model, cfg.MODEL)

    # if student need some extra learnable moduleq
    if hasattr(model.dense_head.kd_head, 'register_extra_layers'):
        model.dense_head.kd_head.register_extra_layers(model.dense_head)


def put_teacher_prior_to_student(teacher_model, student_model, model_cfg):
    student_model.kd_head.voxel_size_tea = teacher_model.kd_head.voxel_size
    student_model.kd_head.feature_map_stride_tea = teacher_model.kd_head.feature_map_stride

    if model_cfg.get('FEATURE_KD', None) and model_cfg.FEATURE_KD.get('ALIGN', None) and \
            model_cfg.FEATURE_KD.ALIGN.MODE == 'bn':
        student_model.dense_head.kd_head.get_prior_knowledge_from_teacher_model(
            teacher_model, model_cfg
        )

    if model_cfg.get('VFE_KD', None) and model_cfg.VFE_KD.get('CN_ALIGN', None) and \
            model_cfg.VFE_KD.CN_ALIGN.ENABLED and model_cfg.VFE_KD.CN_ALIGN.MODE == 'bn':
        student_model.kd_head.select_topk_channels_in_teacher_vfe(teacher_model)


def pop_teacher_intermediate_features(batch):
    pop_list = ['pillar_features', 'spatial_features', 'spatial_features_2d']

    for key in pop_list:
        if key in batch:
            temp = batch.pop(key)
            del temp


def cal_channel_attention_mask(feature):
    """

    Args:
        feature: [B, C, H, W]

    Returns:
        mask: [B, C]
    """
    bs, ch, height, width = feature.shape
    feat = feature.view(bs, ch, -1)
    mask = torch.abs(feat).mean(dim=-1)

    return mask


def cal_spatial_attention_mask(feature):
    """
    Args:
        feature: [B, C, H, W]

    Returns:
        mask: [B, C]
    """
    mask = torch.abs(feature).mean(dim=1)
    return mask

