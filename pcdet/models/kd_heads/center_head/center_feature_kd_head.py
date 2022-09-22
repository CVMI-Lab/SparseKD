import torch
from pcdet.models.kd_heads.kd_head import KDHeadTemplate

from pcdet.models.model_utils.rotated_roi_grid_pool import RotatedGridPool
from pcdet.utils.kd_utils import kd_utils
from pcdet.utils import common_utils, loss_utils


class CenterFeatureKDHead(KDHeadTemplate):
    def __init__(self, model_cfg, dense_head):
        super().__init__(model_cfg, dense_head)
        if self.model_cfg.get('FEATURE_KD'):
            self._init_feature_kd_head(dense_head)
        
        # self.feature_dist = common_utils.AverageMeter()
        # self.feature_dist_top100 = common_utils.AverageMeter()
        # self.feature_dist_spatial = common_utils.AverageMeter()

    def _init_feature_kd_head(self, dense_head):
        if self.model_cfg.FEATURE_KD.get('ROI_POOL', None) and self.model_cfg.FEATURE_KD.ROI_POOL.ENABLED:
            self.roi_pool_func = RotatedGridPool(
                dense_head.point_cloud_range, self.model_cfg.FEATURE_KD.ROI_POOL
            )

    @staticmethod
    def calculate_feature_rois_aligned(kd_fg_mask, corners_3d):
        """
        Given corner points in 3D, filling the kd fg mask

        Args:
            kd_fg_mask: [h, w]
            corners_3d: [4, 2]. [num_boxes, corners in bev, x,y], position of corner points in BEV coordinates

        Returns:

        """
        left = corners_3d[:, 0].min().floor().int()
        right = corners_3d[:, 0].max().ceil().int()

        top = corners_3d[:, 1].min().floor().int()
        bottom = corners_3d[:, 1].max().ceil().int()

        kd_fg_mask[top:bottom, left:right] = 1

    def build_feature_kd_loss(self):
        if self.model_cfg.KD_LOSS.FEATURE_LOSS.type in ['SmoothL1Loss', 'MSELoss', 'KLDivLoss']:
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
        elif loss_cfg.mode == 'spatial':
            kd_feature_loss = self.get_feature_kd_loss_spatial(batch_dict, loss_cfg)
        elif loss_cfg.mode == 'aff':
            kd_feature_loss = self.get_feature_kd_loss_affinity(batch_dict, loss_cfg)
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

        target_dicts = batch_dict['target_dicts_tea']

        if feature_stu.shape != feature_tea.shape and self.model_cfg.FEATURE_KD.get('ALIGN', None):
            feature_tea, feature_stu = self.align_feature_map(
                feature_tea, feature_stu, align_cfg=self.model_cfg.FEATURE_KD.ALIGN
            )

        # whole feature map mimicking
        bs, channel, height, width = feature_tea.shape
        feature_mask = torch.ones([bs, height, width], dtype=torch.float32).cuda()
        if loss_cfg.get('fg_mask', None):
            fg_mask = self.cal_fg_mask_from_target_heatmap_batch(
                target_dict=target_dicts, soft=loss_cfg.get('soft_mask', None)
            )[0]
            feature_mask *= fg_mask

        if loss_cfg.get('score_mask', None):
            score_mask = self.cal_score_mask_from_teacher_pred(batch_dict['pred_tea'], loss_cfg.score_thresh)[0]
            feature_mask *= score_mask

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
        elif self.model_cfg.FEATURE_KD.ROI_POOL.ROI == 'stu':
            pred_dict_stu = self.dense_head.forward_ret_dict['decoded_pred_dicts']
            rois = [pred_dict_stu[i]['pred_boxes'] for i in range(bs)]
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

        # import ipdb;
        # ipdb.set_trace(context=20)
        # from pcdet.datasets.dataset import DatasetTemplate
        # DatasetTemplate.__vis_open3d__(points=batch_dict['points'][:, 1:].cpu().numpy(),
        #                                gt_boxes=batch_dict['gt_boxes'][0].detach().cpu().numpy(),
        #                                ref_boxes=rois[0].cpu().numpy())

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
                    cur_roi_feats_tea = roi_feats_tea[cnt:cnt+num_roi].view(num_roi, -1)

                    rel_tea = common_utils.pair_distance_gpu(cur_roi_feats_tea, cur_roi_feats_tea)
                    rel_tea /= rel_tea.mean()
                    rel_stu = common_utils.pair_distance_gpu(cur_roi_feats, cur_roi_feats)
                    rel_stu /= rel_stu.mean()

                    kd_feat_rel_loss += torch.nn.functional.smooth_l1_loss(rel_tea, rel_stu)
                    cnt += num_roi

                kd_feature_loss += loss_cfg.GID.rel_weight * kd_feat_rel_loss / bs

        return kd_feature_loss

    def get_feature_kd_loss_spatial(self, batch_dict, loss_cfg):
        feature_name = self.model_cfg.FEATURE_KD.FEATURE_NAME
        feature_stu = batch_dict[feature_name]
        feature_name_tea = self.model_cfg.FEATURE_KD.get('FEATURE_NAME_TEA', feature_name)
        feature_tea = batch_dict[feature_name_tea + '_tea']

        if self.model_cfg.FEATURE_KD.ALIGN.target == 'student':
            target_dicts = self.dense_head.forward_ret_dict['target_dicts']
        else:
            target_dicts = batch_dict['target_dicts_tea']

        if feature_stu.shape != feature_tea.shape and self.model_cfg.FEATURE_KD.get('ALIGN', None):
            feature_tea, feature_stu = self.align_feature_map(
                feature_tea, feature_stu, align_cfg=self.model_cfg.FEATURE_KD.ALIGN
            )

        spatial_mask = kd_utils.cal_spatial_attention_mask(feature_stu)
        spatial_mask_tea = kd_utils.cal_spatial_attention_mask(feature_tea)

        # whole feature map mimicking
        bs, channel, height, width = feature_tea.shape
        feature_mask = torch.ones([bs, height, width], dtype=torch.float32).cuda()
        if loss_cfg.get('fg_mask', None):
            fg_mask = self.cal_fg_mask_from_target_heatmap_batch(target_dict=target_dicts)[0]
            feature_mask *= fg_mask

        if loss_cfg.get('score_mask', None):
            score_mask = self.cal_score_mask_from_teacher_pred(batch_dict['pred_tea'], loss_cfg.score_thresh)[0]
            feature_mask *= score_mask

        # self.feature_dist.update(kd_vis_utils.cal_feature_dist(feature_stu, feature_tea).item())
        # self.feature_dist_top100.update(kd_vis_utils.cal_feature_dist(feature_stu, feature_tea, topk=100).item())
        # self.feature_dist_spatial.update(kd_vis_utils.cal_feature_dist(spatial_mask, spatial_mask_tea).item())

        # calculate spatial magnitute inside objects and non-empty regions
        # non_empty_mask = torch.zeros(feature_stu.shape[2:]).cuda()
        # voxel_coords = batch_dict['voxel_coords'].long()
        # non_empty_mask[voxel_coords[:, 2], voxel_coords[:, -1]] = True

        # self.spatial_fg_meter.update(spatial_mask[fg_mask.bool()].mean())
        # self.spatial_nonempty_meter.update(spatial_mask[non_empty_mask.unsqueeze(0).bool()].mean())
        # self.spatial_all_meter.update(spatial_mask.mean())

        kd_feature_loss_all = self.kd_feature_loss_func(spatial_mask, spatial_mask_tea)
        kd_feature_loss = (kd_feature_loss_all * feature_mask).sum() / (feature_mask.sum() + 1e-6)

        kd_feature_loss = kd_feature_loss * loss_cfg.weight

        return kd_feature_loss
    
    def get_feature_kd_loss_affinity(self, batch_dict, loss_cfg):
        feature_name = self.model_cfg.FEATURE_KD.FEATURE_NAME
        feature_stu = batch_dict[feature_name]
        feature_name_tea = self.model_cfg.FEATURE_KD.get('FEATURE_NAME_TEA', feature_name)
        feature_tea = batch_dict[feature_name_tea + '_tea']

        feat_height = feature_stu.shape[2]
        feat_height_tea = feature_tea.shape[2]

        bs, ch = feature_stu.shape[0], feature_stu.shape[1]
        if self.model_cfg.FEATURE_KD.ROI_POOL.ROI == 'gt':
            rois = batch_dict['gt_boxes'].detach()
        elif self.model_cfg.FEATURE_KD.ROI_POOL.ROI == 'tea':
            rois = []
            for b_idx in range(bs):
                cur_pred_tea = batch_dict['decoded_pred_tea'][b_idx]
                pred_scores = cur_pred_tea['pred_scores']
                score_mask = pred_scores > self.model_cfg.FEATURE_KD.ROI_POOL.THRESH
                rois.append(cur_pred_tea['pred_boxes'][score_mask])
        elif self.model_cfg.FEATURE_KD.ROI_POOL.ROI == 'stu':
            pred_dict_stu = self.dense_head.forward_ret_dict['decoded_pred_dicts']
            rois = [pred_dict_stu[i]['pred_boxes'] for i in range(bs)]
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

        roi_feats = self.roi_pool_func(
            feature_stu, rois, voxel_size_stu, feature_map_stride_stu
        )
        roi_feats_tea = self.roi_pool_func(
            feature_tea, rois, voxel_size_tea, feature_map_stride_tea
        )

        # calculate intro object affinity
        intra_aff_matrix = self.cal_cos_sim_affinity_matrix(roi_feats.view(roi_feats.shape[0], ch, -1))
        intra_aff_matrix_tea = self.cal_cos_sim_affinity_matrix(roi_feats_tea.view(roi_feats.shape[0], ch, -1))

        kd_feature_loss = loss_cfg.weight * self.kd_feature_loss_func(
            intra_aff_matrix, intra_aff_matrix_tea
        ).mean()

        return kd_feature_loss

    @staticmethod
    def cal_cos_sim_affinity_matrix(roi_features):
        """_summary_

        Args:
            roi_features (_type_): [N, C, K]
        """
        # [N, K, K]
        sim_matrix = torch.matmul(roi_features.transpose(1, 2), roi_features)
        norm = torch.norm(roi_features, dim=1, keepdim=True)
        affinity_matrix = sim_matrix / torch.clamp((norm * norm.transpose(1, 2)), min=1e-6)

        return affinity_matrix
