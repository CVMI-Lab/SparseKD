import torch
import numpy as np

from pcdet.ops.iou3d_nms import iou3d_nms_utils
from pcdet.models.kd_heads.kd_head import KDHeadTemplate


class AnchorLabelAssignKDHead(KDHeadTemplate):
    def __init__(self, model_cfg, dense_head):
        super(AnchorLabelAssignKDHead, self).__init__(model_cfg, dense_head)
