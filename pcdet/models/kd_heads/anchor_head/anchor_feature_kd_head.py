import torch
from pcdet.models.kd_heads.kd_head import KDHeadTemplate

from pcdet.models.model_utils.rotated_roi_grid_pool import RotatedGridPool
from pcdet.utils import common_utils, loss_utils


class AnchorFeatureKDHead(KDHeadTemplate):
    def __init__(self, model_cfg, dense_head):
        super(AnchorFeatureKDHead, self).__init__(model_cfg, dense_head)
        if self.model_cfg.get('FEATURE_KD'):
            self._init_feature_kd_head(dense_head)

    def _init_feature_kd_head(self, dense_head):
        if self.model_cfg.FEATURE_KD.get('ROI_POOL', None) and self.model_cfg.FEATURE_KD.ROI_POOL.ENABLED:
            self.roi_pool_func = RotatedGridPool(
                dense_head.point_cloud_range, self.model_cfg.FEATURE_KD.ROI_POOL
            )

    def build_feature_kd_loss(self):
        if self.model_cfg.KD_LOSS.FEATURE_LOSS.type in ['SmoothL1Loss', 'MSELoss']:
            self.kd_feature_loss_func = getattr(torch.nn, self.model_cfg.KD_LOSS.FEATURE_LOSS.type)(reduction='none')
        elif self.model_cfg.KD_LOSS.FEATURE_LOSS.type in ['CosineLoss']:
            self.kd_feature_loss_func = getattr(loss_utils, self.model_cfg.KD_LOSS.FEATURE_LOSS.type)()
        else:
            raise NotImplementedError

    def get_feature_kd_loss(self, batch_dict, tb_dict, loss_cfg):
        if loss_cfg.mode == 'raw':
            kd_feature_loss = self.get_feature_kd_loss_raw(batch_dict, loss_cfg)
        elif loss_cfg.mode == 'rois':
            kd_feature_loss = self.get_feature_kd_loss_rois(batch_dict, loss_cfg)
        else:
            raise NotImplementedError

        tb_dict['kd_feat_ls'] = kd_feature_loss if isinstance(kd_feature_loss, float) else kd_feature_loss.item()

        return kd_feature_loss, tb_dict

    def get_feature_kd_loss_raw(self, batch_dict, loss_cfg):
        """
        Args:
            batch_dict:
            loss_cfg
        Returns:

        """
        feature_name = self.model_cfg.FEATURE_KD.FEATURE_NAME
        feature_stu = batch_dict[feature_name]
        feature_name_tea = self.model_cfg.FEATURE_KD.get('FEATURE_NAME_TEA', feature_name)
        feature_tea = batch_dict[feature_name_tea + '_tea']

        if feature_stu.shape != feature_tea.shape and self.model_cfg.FEATURE_KD.get('ALIGN', None):
            feature_tea, feature_stu = self.align_feature_map(
                feature_tea, feature_stu, align_cfg=self.model_cfg.FEATURE_KD.ALIGN
            )

        # whole feature map mimicking
        bs, channel, height, width = feature_tea.shape
        feature_mask = torch.ones([bs, height, width], dtype=torch.float32).cuda()
        if loss_cfg.get('fg_mask', None):
            fg_mask = self.cal_fg_mask_from_gt_boxes_and_spatial_mask(
                batch_dict['gt_boxes'], batch_dict['spatial_mask_tea']
            )
            feature_mask *= fg_mask

        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(1, 2)
        # ax[0].imshow(feature_mask[0].cpu().numpy(), vmin=0, vmax=1, interpolation='none')
        # ax[1].imshow(batch_dict['spatial_mask_tea'][0].cpu().numpy(), vmin=0, vmax=1, interpolation='none')
        # plt.show()
        kd_feature_loss_all = self.kd_feature_loss_func(feature_stu, feature_tea)
        kd_feature_loss = (kd_feature_loss_all * feature_mask.unsqueeze(1)).sum() / (feature_mask.sum() * channel + 1e-6)

        kd_feature_loss = kd_feature_loss * loss_cfg.weight

        return kd_feature_loss

    def get_feature_kd_loss_rois(self, batch_dict, loss_cfg):
        feature_name = self.model_cfg.FEATURE_KD.FEATURE_NAME
        feature_stu = batch_dict[feature_name]
        feature_name_tea = self.model_cfg.FEATURE_KD.get('FEATURE_NAME_TEA', feature_name)
        feature_tea = batch_dict[feature_name_tea + '_tea']

        feat_height = feature_stu.shape[2]
        feat_height_tea = feature_tea.shape[2]

        bs = feature_stu.shape[0]
        if self.model_cfg.FEATURE_KD.ROI_POOL.ROI == 'gt':
            rois = batch_dict['gt_boxes'].detach()
        elif self.model_cfg.FEATURE_KD.ROI_POOL.ROI == 'tea':
            rois = []
            for b_idx in range(bs):
                cur_pred_tea = batch_dict['decoded_pred_tea'][b_idx]
                pred_scores = cur_pred_tea['pred_scores']
                score_mask = pred_scores > self.model_cfg.FEATURE_KD.ROI_POOL.THRESH
                rois.append(cur_pred_tea['pred_boxes'][score_mask])
        else:
            raise NotImplementedError

        if feature_stu.shape[2] == feat_height_tea:
            voxel_size_stu = self.voxel_size_tea
            feature_map_stride_stu = self.feature_map_stride_tea
        elif feature_stu.shape[2] == feat_height:
            voxel_size_stu = self.voxel_size
            feature_map_stride_stu = self.feature_map_stride
        else:
            raise NotImplementedError

        if feature_tea.shape[2] == feat_height_tea:
            voxel_size_tea = self.voxel_size_tea
            feature_map_stride_tea = self.feature_map_stride_tea
        elif feature_tea.shape[2] == feat_height:
            voxel_size_tea = self.voxel_size
            feature_map_stride_tea = self.feature_map_stride
        else:
            raise NotImplementedError

        num_rois = 0
        for roi in rois:
            num_rois += roi.shape[0]

        if num_rois == 0:
            kd_feature_loss = 0.0
        else:
            roi_feats = self.roi_pool_func(
                feature_stu, rois, voxel_size_stu, feature_map_stride_stu
            )
            roi_feats_tea = self.roi_pool_func(
                feature_tea, rois, voxel_size_tea, feature_map_stride_tea
            )

            kd_feature_loss = loss_cfg.weight * self.kd_feature_loss_func(roi_feats, roi_feats_tea).mean()

            if loss_cfg.get('GID', None):
                cnt = 0
                kd_feat_rel_loss = 0
                for b_roi in rois:
                    num_roi = (b_roi[:, 3] != 0).sum()
                    cur_roi_feats = roi_feats[cnt:cnt + num_roi].view(num_roi, -1)
                    cur_roi_feats_tea = roi_feats_tea[cnt:cnt + num_roi].view(num_roi, -1)

                    rel_tea = common_utils.pair_distance_gpu(cur_roi_feats_tea, cur_roi_feats_tea)
                    rel_tea /= rel_tea.mean()
                    rel_stu = common_utils.pair_distance_gpu(cur_roi_feats, cur_roi_feats)
                    rel_stu /= rel_stu.mean()

                    kd_feat_rel_loss += torch.nn.functional.smooth_l1_loss(rel_tea, rel_stu)
                    cnt += num_roi

                kd_feature_loss += loss_cfg.GID.rel_weight * kd_feat_rel_loss / bs

        return kd_feature_loss
