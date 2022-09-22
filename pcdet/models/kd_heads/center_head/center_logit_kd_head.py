import torch

from pcdet.models.kd_heads.kd_head import KDHeadTemplate
from pcdet.utils import loss_utils


class CenterLogitKDHead(KDHeadTemplate):
    def __init__(self, model_cfg, dense_head):
        super().__init__(model_cfg, dense_head)

    def build_logit_kd_loss(self):
        # logit kd hm loss
        if self.model_cfg.KD_LOSS.HM_LOSS.type in ['FocalLossCenterNet']:
            self.kd_hm_loss_func = getattr(loss_utils, self.model_cfg.KD_LOSS.HM_LOSS.type)(
                pos_thresh=self.model_cfg.KD_LOSS.HM_LOSS.pos_thresh
            )
        elif self.model_cfg.KD_LOSS.HM_LOSS.type in ['SmoothL1Loss', 'MSELoss']:
            self.kd_hm_loss_func = getattr(torch.nn, self.model_cfg.KD_LOSS.HM_LOSS.type)(reduction='none')
        else:
            raise NotImplementedError

        # logit kd hm_sort loss
        if self.model_cfg.KD_LOSS.get('HM_SORT_LOSS', None):
            self.kd_hm_sort_loss_func = loss_utils.SortLoss(rank=self.model_cfg.KD_LOSS.HM_SORT_LOSS.rank)
        else:
            self.kd_hm_sort_loss_func = None

        # logit kd regression loss
        if self.model_cfg.KD_LOSS.REG_LOSS.type == 'WeightedSmoothL1Loss':
            self.kd_reg_loss_func = getattr(loss_utils, self.model_cfg.KD_LOSS.REG_LOSS.type)(
                code_weights=self.model_cfg.KD_LOSS.reg_loss.code_weights
            )
        elif self.model_cfg.KD_LOSS.REG_LOSS.type == 'RegLossCenterNet':
            self.kd_reg_loss_func = getattr(loss_utils, self.model_cfg.KD_LOSS.REG_LOSS.type)()
        else:
            raise NotImplementedError

    def get_logit_kd_loss(self, batch_dict, tb_dict):
        if self.model_cfg.LOGIT_KD.MODE == 'decoded_boxes':
            pred_tea = batch_dict['decoded_pred_tea']
            kd_logit_loss, kd_hm_loss, kd_reg_loss = self.get_kd_loss_with_decoded_boxes(
                pred_tea, self.model_cfg.KD_LOSS
            )
        elif self.model_cfg.LOGIT_KD.MODE == 'raw_pred':
            pred_tea = batch_dict['pred_tea']
            kd_logit_loss, kd_hm_loss, kd_reg_loss, kd_sort_loss = self.get_kd_loss_with_raw_prediction(
                pred_tea, self.model_cfg.KD_LOSS, target_dict_tea=batch_dict['target_dicts_tea']
            )
        elif self.model_cfg.LOGIT_KD.MODE == 'target':
            kd_logit_loss, kd_hm_loss, kd_reg_loss = self.get_kd_loss_with_target_tea(
                batch_dict['pred_tea'], self.model_cfg.KD_LOSS, target_dict_tea=batch_dict['target_dicts_tea']
            )
        else:
            raise NotImplementedError

        tb_dict['kd_hm_ls'] = kd_hm_loss if isinstance(kd_hm_loss, float) else kd_hm_loss.item()
        tb_dict['kd_loc_ls'] = kd_reg_loss if isinstance(kd_reg_loss, float) else kd_reg_loss.item()
        tb_dict['kd_sort_ls'] = kd_sort_loss if isinstance(kd_sort_loss, float) else kd_sort_loss.item()

        return kd_logit_loss, tb_dict

    def get_kd_loss_with_raw_prediction(self, pred_tea, loss_cfg, target_dict_tea):
        """
        Args:
            pred_tea: pred_dict of teacher
                center: [bs, 2, feat_h, feat_w]. Offset to the nearest center
                center_z: [bs, 1, feat_h, feat_w]. absolute coordinates


            loss_cfg: kd loss config

        Returns:

        """
        pred_stu = self.dense_head.forward_ret_dict['pred_dicts']
        if self.model_cfg.LOGIT_KD.ALIGN.target == 'student':
            target_dicts = self.dense_head.forward_ret_dict['target_dicts']
        else:
            target_dicts = target_dict_tea

        assert len(pred_tea) == len(pred_stu)

        kd_hm_loss = 0
        kd_reg_loss = 0

        for idx, cur_pred_stu in enumerate(pred_stu):
            cur_pred_tea = pred_tea[idx]
            cur_hm_tea = self.dense_head.sigmoid(cur_pred_tea['hm'])

            # interpolate if needed
            if (cur_hm_tea.shape != cur_pred_stu['hm'].shape) and self.model_cfg.LOGIT_KD.get('ALIGN', None):
                hm_tea, hm_stu = self.align_feature_map(
                    cur_hm_tea, cur_pred_stu['hm'], self.model_cfg.LOGIT_KD.ALIGN
                )
            else:
                hm_tea, hm_stu = cur_hm_tea, cur_pred_stu['hm']

            # classification loss
            if loss_cfg.HM_LOSS.weight == 0:
                kd_hm_loss_raw = 0
            elif loss_cfg.HM_LOSS.type == 'FocalLossCenterNet':
                if loss_cfg.HM_LOSS.get('inverse', None):
                    kd_hm_loss_raw = self.kd_hm_loss_func(hm_tea, hm_stu)
                else:
                    kd_hm_loss_raw = self.kd_hm_loss_func(hm_stu, hm_tea)
            elif loss_cfg.HM_LOSS.type == 'WeightedSmoothL1Loss':
                bs, channel = hm_stu.shape[0], hm_stu.shape[1]
                heatmap_stu = hm_stu.view(bs, channel, -1).permute(0, 2, 1)
                heatmap_tea = hm_tea.view(bs, channel, -1).permute(0, 2, 1)
                kd_hm_loss_all = self.kd_hm_loss_func(heatmap_stu, heatmap_tea)
                # position-wise confidence mask: shape [bs, h*w, c]
                mask = (torch.max(heatmap_stu, -1)[0] > loss_cfg.HM_LOSS.thresh).float() * \
                                  (torch.max(heatmap_tea, -1)[0] > loss_cfg.HM_LOSS.thresh).float()
                kd_hm_loss_raw = (kd_hm_loss_all * mask.unsqueeze(-1)).sum() / (mask.sum() + 1e-6)
            elif loss_cfg.HM_LOSS.type in ['SmoothL1Loss', 'MSELoss']:
                kd_hm_loss_all = self.kd_hm_loss_func(hm_stu, hm_tea)
                # position-wise confidence mask: shape [bs, c, h, w]
                mask = (torch.max(hm_tea, dim=1)[0] > loss_cfg.HM_LOSS.thresh).float()
                if loss_cfg.HM_LOSS.get('soft_mask', None):
                    mask = torch.max(hm_tea, dim=1)[0] * mask

                if loss_cfg.HM_LOSS.get('fg_mask', None):
                    fg_mask = self.cal_fg_mask_from_target_heatmap_batch(
                        target_dicts, soft=loss_cfg.HM_LOSS.get('soft_mask', None)
                    )[idx]
                    mask *= fg_mask
                
                if loss_cfg.HM_LOSS.get('rank', -1) != -1:
                    rank_mask = self.cal_rank_mask_from_teacher_pred(pred_tea, K=loss_cfg.HM_LOSS.rank)[idx]
                    mask *= rank_mask

                kd_hm_loss_raw = (kd_hm_loss_all * mask.unsqueeze(1)).sum() / (mask.sum() + 1e-6)
            else:
                raise NotImplementedError
            kd_hm_loss += loss_cfg.HM_LOSS.weight * kd_hm_loss_raw

            if self.kd_hm_sort_loss_func is not None and loss_cfg.HM_SORT_LOSS.weight != 0:
                kd_hm_sort_loss = self.kd_hm_sort_loss_func(hm_stu, hm_tea)
                kd_hm_sort_loss = loss_cfg.HM_SORT_LOSS.weight * kd_hm_sort_loss
            else:
                kd_hm_sort_loss = 0.0

            # localization loss
            # parse teacher prediction to target style
            pred_boxes_tea = torch.cat([cur_pred_tea[head_name] for head_name in self.dense_head.separate_head_cfg.HEAD_ORDER], dim=1)
            pred_boxes_stu = torch.cat([cur_pred_stu[head_name] for head_name in self.dense_head.separate_head_cfg.HEAD_ORDER], dim=1)
            if loss_cfg.REG_LOSS.weight == 0 or (pred_boxes_tea.shape != pred_boxes_stu.shape):
                kd_reg_loss_raw = 0
            elif loss_cfg.REG_LOSS.type == 'RegLossCenterNet':
                pred_boxes_tea_selected = loss_utils._transpose_and_gather_feat(pred_boxes_tea, target_dicts['inds'][idx])

                kd_reg_loss_raw = self.kd_reg_loss_func(
                    pred_boxes_stu, target_dicts['masks'][idx], target_dicts['inds'][idx], pred_boxes_tea_selected
                )
                kd_reg_loss_raw = (kd_reg_loss_raw * kd_reg_loss_raw.new_tensor(
                    loss_cfg.REG_LOSS.code_weights)).sum()
            else:
                raise NotImplementedError
            kd_reg_loss += loss_cfg.REG_LOSS.weight * kd_reg_loss_raw

        kd_loss = (kd_hm_loss + kd_hm_sort_loss + kd_reg_loss) / len(pred_stu)

        return kd_loss, kd_hm_loss / len(pred_stu), kd_reg_loss / len(pred_stu), kd_hm_sort_loss / len(pred_stu)

    def get_kd_loss_with_target_tea(self, pred_tea, loss_cfg, target_dict_tea):
        """
        Args:
            pred_tea: pred_dict of teacher
                center: [bs, 2, feat_h, feat_w]. Offset to the nearest center
                center_z: [bs, 1, feat_h, feat_w]. absolute coordinates


            loss_cfg: kd loss config
            target_dict_tea

        Returns:

        """
        pred_stu = self.dense_head.forward_ret_dict['pred_dicts']

        kd_hm_loss = 0
        kd_reg_loss = 0

        for idx, cur_pred_stu in enumerate(pred_stu):
            cur_pred_tea = pred_tea[idx]
            target_hm = target_dict_tea['heatmaps'][idx]

            # interpolate if needed
            if (target_hm.shape != cur_pred_stu['hm'].shape) and loss_cfg.HM_LOSS.weight != 0 and \
                    self.model_cfg.LOGIT_KD.get('ALIGN', None):
                hm_tea, hm_stu = self.align_feature_map(
                    target_hm, cur_pred_stu['hm'], self.model_cfg.LOGIT_KD.ALIGN
                )
            else:
                hm_tea, hm_stu = target_hm, cur_pred_stu['hm']

            # classification loss
            if loss_cfg.HM_LOSS.weight == 0:
                kd_hm_loss_raw = 0
            elif loss_cfg.HM_LOSS.type == 'FocalLossCenterNet':
                kd_hm_loss_raw = self.kd_hm_loss_func(hm_stu, hm_tea)
            elif loss_cfg.HM_LOSS.type in ['SmoothL1Loss', 'MSELoss']:
                kd_hm_loss_all = self.kd_hm_loss_func(hm_stu, hm_tea)
                # position-wise confidence mask: shape [bs, c, h, w]
                mask = (torch.max(hm_tea, dim=1)[0] > loss_cfg.HM_LOSS.thresh).float()
                if loss_cfg.HM_LOSS.get('fg_mask', None):
                    fg_mask = self.cal_fg_mask_from_target_hm(
                        target_dict_tea['heatmaps'][idx], hm_stu.shape
                    ).squeeze(1)
                    mask *= fg_mask
                kd_hm_loss_raw = (kd_hm_loss_all * mask.unsqueeze(1)).sum() / (mask.sum() + 1e-6)
            else:
                raise NotImplementedError
            kd_hm_loss += loss_cfg.HM_LOSS.weight * kd_hm_loss_raw

            # localization loss
            pred_boxes_stu = torch.cat([cur_pred_stu[head_name] for head_name in self.dense_head.separate_head_cfg.HEAD_ORDER], dim=1)
            if loss_cfg.REG_LOSS.weight == 0:
                kd_reg_loss_raw = 0
            elif loss_cfg.REG_LOSS.type == 'RegLossCenterNet':
                # parse teacher prediction to target style
                pred_boxes_tea = torch.cat([cur_pred_tea[head_name] for head_name in self.dense_head.separate_head_cfg.HEAD_ORDER], dim=1)

                # interpolate if the shape of feature map not match
                if (pred_boxes_tea.shape != pred_boxes_stu.shape) and self.model_cfg.LOGIT_KD.get('ALIGN', None):
                    pred_boxes_tea, pred_boxes_stu = self.align_feature_map(
                        pred_boxes_tea, pred_boxes_stu, self.model_cfg.LOGIT_KD.ALIGN
                    )

                pred_boxes_tea_selected = loss_utils._transpose_and_gather_feat(pred_boxes_tea,
                                                                                target_dict_tea['inds'][idx])

                kd_reg_loss_raw = self.reg_loss_func(
                    pred_boxes_stu, target_dict_tea['masks'][idx], target_dict_tea['inds'][idx], pred_boxes_tea_selected
                )
                kd_reg_loss_raw = (kd_reg_loss_raw * kd_reg_loss_raw.new_tensor(
                    loss_cfg.REG_LOSS.code_weights)).sum()
            else:
                raise NotImplementedError
            kd_reg_loss += loss_cfg.REG_LOSS.weight * kd_reg_loss_raw

        kd_loss = (kd_hm_loss + kd_reg_loss) / len(pred_stu)

        return kd_loss, kd_hm_loss / len(pred_stu), kd_reg_loss / len(pred_stu)

    def get_kd_loss_with_decoded_boxes(self, pred_tea, loss_cfg, dense_head):
        """
        Args:
            pred_tea: list. [batch_size]
                pred_scores:
                pred_boxes:
                pred_labels
            loss_cfg:

        Returns:

        """
        pred_stu = dense_head.forward_ret_dict['decoded_pred_dicts']
        batch_kd_hm_loss = 0
        batch_kd_reg_loss = 0
        for b_idx, cur_pred_stu in enumerate(pred_stu):
            cur_pred_tea = pred_tea[b_idx]
            # filter boxes by confidence with a given threshold
            score_idx_stu = (cur_pred_stu['pred_scores'] >= loss_cfg.PRED_FILTER.score_thresh).nonzero().squeeze(-1)
            score_idx_tea = (cur_pred_tea['pred_scores'] >= loss_cfg.PRED_FILTER.score_thresh).nonzero().squeeze(-1)

            # filter boxes by iou
            iou_mask_stu, iou_mask_tea = self.filter_boxes_by_iou(
                cur_pred_stu['pred_boxes'][score_idx_stu], cur_pred_tea['pred_boxes'][score_idx_tea], loss_cfg
            )

            valid_idx_stu = score_idx_stu[iou_mask_stu]
            valid_idx_tea = score_idx_tea[iou_mask_tea]

            if valid_idx_stu.shape[0] == 0 or valid_idx_tea.shape[0] == 0:
                continue

            # confidence loss
            if loss_cfg.HM_LOSS.type == 'WeightedSmoothL1Loss':
                kd_hm_loss_all = self.kd_hm_loss_func(
                    cur_pred_stu['pred_scores'][None, valid_idx_stu, None],
                    cur_pred_tea['pred_scores'][None, valid_idx_tea, None].detach()
                )
                batch_kd_hm_loss += kd_hm_loss_all.mean()
            else:
                raise NotImplementedError

            # box regression loss
            if loss_cfg.REG_LOSS.type == 'WeightedSmoothL1Loss':
                valid_boxes_stu, valid_boxes_tea = self.add_sin_difference(
                    cur_pred_stu['pred_boxes'][valid_idx_stu], cur_pred_tea['pred_boxes'][valid_idx_tea]
                )

                kd_reg_loss_all = self.kd_reg_loss_func(
                    valid_boxes_stu.unsqueeze(0), valid_boxes_tea.unsqueeze(0).detach()
                )
                batch_kd_reg_loss += kd_reg_loss_all.mean()
            else:
                raise NotImplementedError

        kd_hm_loss = batch_kd_hm_loss * loss_cfg.HM_LOSS.weight / len(pred_stu)
        kd_reg_loss = batch_kd_reg_loss * loss_cfg.REG_LOSS.weight / len(pred_stu)

        kd_loss = kd_hm_loss + kd_reg_loss

        return kd_loss, kd_hm_loss, kd_reg_loss
