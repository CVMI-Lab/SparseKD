import torch
import torch.nn as nn
import numpy as np

from torch.nn.init import kaiming_normal_

from .center_logit_kd_head import CenterLogitKDHead
from .center_label_assign_kd_head import CenterLabelAssignKDHead
from .center_feature_kd_head import CenterFeatureKDHead
from .center_vfe_kd_head import CenterVfeKDHead
from .center_roi_kd_head import CenterRoIKDHead
from pcdet.models.model_utils import centernet_utils


class CenterHeadKD(CenterLogitKDHead, CenterFeatureKDHead, CenterRoIKDHead,
                   CenterLabelAssignKDHead, CenterVfeKDHead):
    """
    An ad-hoc module for knowledge distillation in CenterHead
    """

    def __init__(self, model_cfg, dense_head):
        super().__init__(model_cfg, dense_head)
        self.build_loss(dense_head)

    def register_extra_layers(self, dense_head):
        """
        Register some learnable layers
        Args:
            dense_head:

        Returns:

        """
        if self.model_cfg.get('FEATURE_KD', None) and self.model_cfg.FEATURE_KD.get('ALIGN', None) and \
                self.model_cfg.FEATURE_KD.ALIGN.ENABLED and self.model_cfg.FEATURE_KD.ALIGN.MODE == 'conv':
            num_filters = self.model_cfg.FEATURE_KD.ALIGN.num_filters
            align_block = []
            use_norm = self.model_cfg.FEATURE_KD.ALIGN.use_norm
            use_act = self.model_cfg.FEATURE_KD.ALIGN.use_act
            kernel_size = self.model_cfg.FEATURE_KD.ALIGN.kernel_size
            groups = self.model_cfg.FEATURE_KD.ALIGN.groups
            for i in range(len(num_filters)-1):
                cur_layer = [
                    nn.Conv2d(
                        num_filters[i], num_filters[i+1], kernel_size=kernel_size,
                        padding=int((kernel_size - 1) / 2), groups=groups, bias=(not use_norm)
                    )]
                if use_norm:
                    cur_layer.append(nn.BatchNorm2d(num_filters[i+1]))

                if i < (len(num_filters) - 1) and use_act:
                    cur_layer.append(nn.ReLU())

                fc = nn.Sequential(*cur_layer)
                for m in fc.modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_normal_(m.weight.data)
                        if hasattr(m, "bias") and m.bias is not None:
                            nn.init.constant_(m.bias, 0)

                align_block.append(fc)
            dense_head.__setattr__('align_block', nn.Sequential(*align_block))

    def put_pred_to_ret_dict(self, dense_head, data_dict, pred_dicts):
        if data_dict.get('teacher_decoded_pred_flag', None) and dense_head.training:
            decoded_pred_dicts = dense_head.generate_predicted_boxes(
                data_dict['batch_size'], pred_dicts, no_nms=not self.need_nms, nms_config=self.nms_config
            )
            dense_head.forward_ret_dict['decoded_pred_dicts'] = decoded_pred_dicts

    def get_prior_knowledge_from_teacher_model(self, teacher_model, model_cfg):
        teacher_bn_weights = []
        for i, deblock in enumerate(teacher_model.backbone_2d.deblocks):
            teacher_bn_weights.append(deblock[1].weight)

        student_channel = model_cfg.BACKBONE_2D.NUM_UPSAMPLE_FILTERS

        if not self.model_cfg.FEATURE_KD.ALIGN.pre_block:
            teacher_bn_weights = torch.cat(teacher_bn_weights)
            _, self.teacher_topk_channel_idx = torch.topk(
                teacher_bn_weights.abs(), k=sum(student_channel)
            )
        else:
            channel_idx = []
            base_idx = 0
            for i in range(len(teacher_bn_weights)):
                _, cur_channel_idx = torch.topk(
                    teacher_bn_weights[i].abs(), k=student_channel[i]
                )
                channel_idx.append(cur_channel_idx + base_idx)
                base_idx += teacher_bn_weights[i].shape[0]
            self.teacher_topk_channel_idx = torch.cat(channel_idx)

    def get_kd_loss(self, batch_dict, tb_dict):
        kd_loss = 0.0
        # Logit KD loss
        if self.model_cfg.get('LOGIT_KD', None) and self.model_cfg.LOGIT_KD.ENABLED:
            kd_logit_loss, tb_dict = self.get_logit_kd_loss(batch_dict, tb_dict)
            kd_loss += kd_logit_loss

        if self.model_cfg.get('FEATURE_KD', None) and self.model_cfg.FEATURE_KD.ENABLED:
            kd_feature_loss, tb_dict = self.get_feature_kd_loss(
                batch_dict, tb_dict, self.model_cfg.KD_LOSS.FEATURE_LOSS
            )
            kd_loss += kd_feature_loss

        return kd_loss, tb_dict

    @staticmethod
    def cal_fg_mask_from_target_heatmap_batch(target_dict, soft=False):
        """_summary_

        Args:
            target_dict (_type_): _description_
                heatmaps: [num_class, H, W]
        
        Returns:
            fg_mask: list with each torch.tensor [B, H, W]
        """
        fg_mask_list = []
        for idx, target_hm in enumerate(target_dict['heatmaps']):
            if soft:
                fg_mask = target_hm.max(dim=1)[0]
            else:
                fg_mask = (target_hm.sum(dim=1) > 0).float()
            fg_mask_list.append(fg_mask)
        return fg_mask_list
    
    def cal_center_mask_from_target_inds_batch(self, target_dict):
        """_summary_

        Args:
            target_dict (_type_): _description_
                inds: [B, N]
        
        Returns:
            fg_mask: list with each torch.tensor [B, H, W]
        """
        center_mask_list = []
        bs, ch, height, width = target_dict['heatmaps'][0].shape
        N = target_dict['inds'][0].shape[-1]
        for idx, target_inds in enumerate(target_dict['inds']):
            center_mask = torch.zeros([bs, height, width], dtype=torch.float32).cuda()
            # [B*N, 2]
            feature_idx = self.parse_voxel_inds_to_feature_idx(target_inds, width).view(-1, 2)
            batch_idx = torch.arange(bs*N).long() / N
            center_mask[batch_idx, feature_idx[:, 0], feature_idx[:, 1]] = True
            center_mask[:, 0, 0] = False
            center_mask_list.append(center_mask)
        return center_mask_list

    @staticmethod
    def cal_score_mask_from_teacher_pred(pred_tea, thresh, soft=False):
        score_mask_list = []
        for _, cur_pred_tea in enumerate(pred_tea):
            cur_hm_tea = cur_pred_tea['hm'].sigmoid()
            mask = (torch.max(cur_hm_tea, dim=1)[0] > thresh).float()
            if soft:
                mask = torch.max(cur_hm_tea, dim=1)[0] * mask
            score_mask_list.append(mask)
        
        return score_mask_list
    
    @staticmethod
    def cal_rank_mask_from_teacher_pred(pred_tea, K):
        score_mask_list = []
        bs, ch, height, width = pred_tea[0]['hm'].shape
        mask = torch.zeros([bs, height, width], dtype=torch.float32).cuda()
        for _, cur_pred_tea in enumerate(pred_tea):
            cur_hm_tea = cur_pred_tea['hm'].sigmoid()
            _, _, _, ys, xs = centernet_utils._topk(cur_hm_tea, K=K)
            batch_idx = torch.arange(bs*K).long() / K
            mask[batch_idx, ys.view(-1).long(), xs.view(-1).long()] = True
            score_mask_list.append(mask)
        
        return score_mask_list

    @staticmethod
    def extend_spatial_mask(mask, mode='cross'):
        """_summary_

        Args:
            mask (_type_): [B, H, W]
            mode (str, optional): _description_. Defaults to 'cross'.
        """
        if mode == 'cross':
            conv_kernel = torch.from_numpy(np.array(
                [[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.float32
            )).cuda().view(1, 1, 3, 3)
            new_mask = nn.functional.conv2d(mask.unsqueeze(1), conv_kernel, padding=1).squeeze(1)
        else:
            raise NotImplementedError

        return new_mask

    @staticmethod
    def parse_voxel_inds_to_feature_idx(inds, feature_width):
        """

        Args:
            inds (tensor): [B, N]
            feature_width (scalar)

        Returns:

        """
        feature_index = torch.zeros([inds.shape[0], inds.shape[1], 2], dtype=torch.long).cuda()
        feature_index[..., 0] = inds // feature_width
        feature_index[..., 1] = inds % feature_width

        return feature_index
