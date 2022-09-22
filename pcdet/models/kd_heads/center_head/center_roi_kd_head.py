import torch

from pcdet.models.kd_heads.kd_head import KDHeadTemplate
from pcdet.utils import loss_utils


class CenterRoIKDHead(KDHeadTemplate):
    def __init__(self, model_cfg, dense_head):
        super().__init__(model_cfg, dense_head)

    def build_roi_kd_loss(self):
        # logit kd hm loss
        if self.model_cfg.KD_LOSS.ROI_CLS_LOSS.type in ['SmoothL1Loss', 'MSELoss']:
            self.kd_roi_cls_loss_func = getattr(torch.nn, self.model_cfg.KD_LOSS.ROI_CLS_LOSS.type)(reduction='none')
        else:
            raise NotImplementedError

        # logit kd regression loss
        if self.model_cfg.KD_LOSS.ROI_REG_LOSS.type == 'WeightedSmoothL1Loss':
            self.kd_roi_reg_loss_func = getattr(loss_utils, self.model_cfg.KD_LOSS.ROI_REG_LOSS.type)(
                code_weights=self.model_cfg.KD_LOSS.ROI_REG_LOSS.code_weights
            )
        else:
            raise NotImplementedError

    def get_roi_kd_loss(self, batch_dict, tb_dict):
        loss_cfg = self.model_cfg.KD_LOSS

        rcnn_cls_stu = self.roi_head.forward_ret_dict['rcnn_cls']
        rcnn_cls_tea = batch_dict['rcnn_cls_tea']

        # cls loss
        if loss_cfg.ROI_CLS_LOSS.weight == 0:
            kd_roi_cls_loss_raw = 0
        elif loss_cfg.ROI_CLS_LOSS.type in ['SmoothL1Loss', 'MSELoss']:
            kd_roi_cls_loss_raw = self.kd_roi_cls_loss_func(rcnn_cls_stu, rcnn_cls_tea).mean()
        else:
            raise NotImplementedError
        kd_roi_cls_loss = kd_roi_cls_loss_raw * loss_cfg.ROI_CLS_LOSS.weight

        reg_valid_mask = self.roi_head.forward_ret_dict['reg_valid_mask'].view(-1)
        rcnn_reg_stu = self.roi_head.forward_ret_dict['rcnn_reg']
        rcnn_reg_tea = batch_dict['rcnn_reg_tea']
        fg_mask = (reg_valid_mask > 0)
        fg_sum = fg_mask.long().sum().item()
        rcnn_batch_size = reg_valid_mask.shape[0]

        # reg loss
        if loss_cfg.ROI_REG_LOSS.weight == 0:
            kd_roi_reg_loss_raw = 0
        elif loss_cfg.ROI_CLS_LOSS.type in ['SmoothL1Loss', 'MSELoss']:
            kd_roi_reg_loss_raw = self.kd_roi_cls_loss_func(
                rcnn_reg_stu.unsqueeze(0), rcnn_reg_tea.unsqueeze(0)
            )
            kd_roi_reg_loss_raw = (kd_roi_reg_loss_raw.view(rcnn_batch_size, -1) * \
                                   fg_mask.unsqueeze(dim=-1).float()).sum() / max(fg_sum, 1)
        else:
            raise NotImplementedError
        kd_roi_reg_loss = kd_roi_reg_loss_raw * loss_cfg.ROI_REG_LOSS.weight

        kd_roi_loss = kd_roi_cls_loss + kd_roi_reg_loss

        tb_dict['kd_r-c_ls'] = kd_roi_cls_loss if isinstance(kd_roi_cls_loss, float) else kd_roi_cls_loss.item()
        tb_dict['kd_r-r_ls'] = kd_roi_reg_loss if isinstance(kd_roi_reg_loss, float) else kd_roi_reg_loss.item()

        return kd_roi_loss, tb_dict
