import torch
from pcdet.models.kd_heads.kd_head import KDHeadTemplate
from pcdet.models.model_utils.basic_block_2d import focus
from pcdet.models.model_utils.positional_encoding import sinusoidal_positional_encoding_2d

try:
    import torch_scatter
except Exception as e:
    # Incase someone doesn't want to use dynamic pillar vfe and hasn't installed torch_scatter
    pass

from pcdet.models.backbones_3d.vfe.dynamic_kp_vfe import VirtualPointAggerator


class CenterVfeKDHead(KDHeadTemplate):
    def __init__(self, model_cfg, dense_head):
        super().__init__(model_cfg, dense_head)
        if model_cfg.get('VFE_KD', None) and model_cfg.VFE_KD.get('KERNEL', None) and \
                model_cfg.VFE_KD.KERNEL.ENABLED:
            voxel_size = torch.tensor(dense_head.voxel_size).cuda()
            self.kp_aggregator = VirtualPointAggerator(model_cfg.VFE_KD.KERNEL, voxel_size)
    
    def build_vfe_kd_loss(self):
        if self.model_cfg.KD_LOSS.VFE_LOSS.type in ['SmoothL1Loss', 'MSELoss']:
            self.vfe_kd_loss_func = getattr(torch.nn, self.model_cfg.KD_LOSS.VFE_LOSS.type)(reduction='none')
        else:
            raise NotImplementedError
    
    def select_topk_channels_in_teacher_vfe(self, teacher_model):
        teacher_bn_weights = teacher_model.vfe.pfn_layers[1].norm.weight
        n_channel = self.model_cfg.VFE_KD.CN_ALIGN.N_CHANNEL
        _, self.teacher_vfe_topk_channel_idx = torch.topk(
            teacher_bn_weights.abs(), k=n_channel
        )

    def get_vfe_kd_loss(self, batch_dict, tb_dict, loss_cfg):
        if loss_cfg.mode == 'point':
            vfe_kd_loss = self.get_vfe_kd_loss_point(batch_dict, loss_cfg)
        elif loss_cfg.mode == 'voxel':
            vfe_kd_loss = self.get_vfe_kd_loss_voxel(batch_dict, loss_cfg)
        elif loss_cfg.mode == 'bev':
            vfe_kd_loss = self.get_vfe_kd_loss_bev_focus(batch_dict, loss_cfg)
        elif loss_cfg.mode == 'kp':
            vfe_kd_loss = self.get_vfe_kd_loss_kp(batch_dict, loss_cfg)
        else:
            raise NotImplementedError
        
        tb_dict['kd_vfe_ls'] = vfe_kd_loss if isinstance(vfe_kd_loss, float) else vfe_kd_loss.item()
        return vfe_kd_loss, tb_dict

    def get_vfe_kd_loss_point(self, batch_dict, loss_cfg):
        """
        Calculate point-wise vfe feature knowledge distillation loss
        """
        point_features = batch_dict['point_features']
        point_features_tea = batch_dict['point_features_tea']
        if loss_cfg.weight != 0:
            assert point_features.shape == point_features_tea.shape
            vfe_kd_loss_raw = self.vfe_kd_loss_func(point_features, point_features_tea)
            vfe_kd_loss = loss_cfg.weight * vfe_kd_loss_raw.mean()
        else:
            vfe_kd_loss = 0.0

        return vfe_kd_loss

    def get_vfe_kd_loss_voxel(self, batch_dict, loss_cfg):
        point_features = batch_dict['point_features']
        point_features_tea = batch_dict['point_features_tea']
        f_center = batch_dict.get('f_center', None)

        if loss_cfg.target == 'teacher':
            unq_inv_pfn = batch_dict['unq_inv_pfn_tea']
            voxel_coord = batch_dict['voxel_coords_tea']
            target_dicts = batch_dict['target_dicts_tea']
        elif loss_cfg.target == 'student':
            unq_inv_pfn = batch_dict['unq_inv_pfn']
            voxel_coord = batch_dict['voxel_coords']
            target_dicts = self.dense_head.forward_ret_dict['target_dicts']
        else:
            raise NotImplementedError

        if loss_cfg.weight != 0:
            assert point_features.shape == point_features_tea.shape
            voxel_features = self.aggregate_voxel_features(point_features, unq_inv_pfn, loss_cfg, f_center)
            voxel_features_tea = self.aggregate_voxel_features(point_features_tea, unq_inv_pfn, loss_cfg, f_center)

            if self.model_cfg.VFE_KD.get('POS_EMB', None) and self.model_cfg.VFE_KD.POS_EMB.ENABLED:
                pos_emb = self.generate_positional_encoding(voxel_coord, self.model_cfg.VFE_KD.POS_EMB)
                voxel_features_tea += pos_emb

            vfe_kd_loss_raw = self.vfe_kd_loss_func(voxel_features, voxel_features_tea)
            
            pillar_mask = torch.ones(vfe_kd_loss_raw.shape[0], dtype=torch.float32).cuda()
            if loss_cfg.get('fg_mask', None):
                bev_fg_mask = self.cal_fg_mask_from_target_heatmap_batch(target_dicts)[0]
                pillar_fg_mask = self.extract_pillar_mask_from_bev_mask(voxel_coord, bev_fg_mask)
                pillar_mask *= pillar_fg_mask

            if loss_cfg.get('score_mask', None):
                bev_score_mask = self.cal_score_mask_from_teacher_pred(batch_dict['pred_tea'], loss_cfg.score_thresh)[0]
                pillar_score_mask = self.extract_pillar_mask_from_bev_mask(voxel_coord, bev_score_mask)
                pillar_mask *= pillar_score_mask

            vfe_kd_loss = loss_cfg.weight * (vfe_kd_loss_raw * pillar_mask.unsqueeze(1)).sum() / (pillar_mask.sum() * vfe_kd_loss_raw.shape[1] + 1e-6)
        else:
            vfe_kd_loss = 0.0

        return vfe_kd_loss

    def get_vfe_kd_loss_rois(self, batch_dict, loss_cfg):
        raise NotImplementedError

    @staticmethod
    def extract_pillar_mask_from_bev_mask(voxel_coords, bev_mask):
        """


        Args:
            voxel_coords (_type_): [B*N, b_idx, z, y , x]
            bev_mask (_type_): [B, H, W]
        """
        pillar_mask_list = []
        for b_idx in range(bev_mask.shape[0]):
            b_mask = voxel_coords[:, 0] == b_idx
            pillar_coords_xy = voxel_coords[b_mask][:, [0, 3, 2]].long()
            pillar_mask = bev_mask[b_idx, pillar_coords_xy[:, 2], pillar_coords_xy[:, 1]]
            pillar_mask_list.append(pillar_mask)

        batch_pillar_mask = torch.cat(pillar_mask_list, dim=0)

        assert batch_pillar_mask.shape[0] == voxel_coords.shape[0]

        return batch_pillar_mask

    def get_vfe_kd_loss_bev_focus(self, batch_dict, loss_cfg):
        voxel_feat_stu = batch_dict['spatial_features']
        raw_voxel_feat_tea = batch_dict['spatial_features_tea']

        if self.model_cfg.VFE_KD.get('CN_ALIGN', None) and self.model_cfg.VFE_KD.CN_ALIGN.ENABLED:
            raw_voxel_feat_tea = self.vfe_channel_align(
                self.model_cfg.VFE_KD.CN_ALIGN, raw_voxel_feat_tea
            )

        voxel_feat_tea = focus(raw_voxel_feat_tea)

        if loss_cfg.get('fusion_tea', None):
            voxel_feat_tea += voxel_feat_tea.mean(dim=1, keepdim=True)

        target_dicts = self.dense_head.forward_ret_dict['target_dicts']
        if loss_cfg.weight != 0:
            vfe_kd_loss_raw = self.vfe_kd_loss_func(voxel_feat_stu, voxel_feat_tea)
            channel = voxel_feat_stu.shape[1]
            vfe_mask = torch.any(voxel_feat_stu > 0, dim=1).float()
            if loss_cfg.get('fg_mask', None):
                bev_fg_mask = self.cal_fg_mask_from_target_heatmap_batch(target_dicts)[0]
                vfe_mask *= bev_fg_mask

            if loss_cfg.get('score_mask', None):
                bev_score_mask_tea = self.cal_score_mask_from_teacher_pred(batch_dict['pred_tea'], loss_cfg.score_thresh)[0]
                bev_score_mask_coarse = focus(bev_score_mask_tea.unsqueeze(1))
                bev_score_mask_coarse = torch.any(bev_score_mask_coarse > 0, dim=1).float()
                vfe_mask *= bev_score_mask_coarse

            vfe_kd_loss = (vfe_kd_loss_raw * vfe_mask.unsqueeze(1)).sum() / (vfe_mask.sum() * channel + 1e-6)
            vfe_kd_loss *= loss_cfg.weight
        else:
            vfe_kd_loss = 0
        
        return vfe_kd_loss

    def vfe_channel_align(self, align_cfg, feature):
        if align_cfg.MODE == 'bn':
            new_feature = self.bn_channel_align(
                feature, channel_idx=self.teacher_vfe_topk_channel_idx
            )
        else:
            raise NotImplementedError

        return new_feature

    @staticmethod
    def generate_positional_encoding(voxel_coord, pos_cfg):
        voxel_coord_xy = voxel_coord[:, [3, 2]]
        if pos_cfg.win_size != -1:
            voxel_coord_xy = voxel_coord_xy % pos_cfg.win_size

        pos_emb = sinusoidal_positional_encoding_2d(
            voxel_coord_xy, hidden_dim=pos_cfg.hidden_dim, min_timescale=pos_cfg.min_scale,
            max_timescale=pos_cfg.max_scale
        )
        return pos_emb

    def aggregate_voxel_features(self, point_feat, unq_inv_pfn, loss_cfg, f_center=None):
        if loss_cfg.get('vp', None):
            point_feat = self.kp_aggregator(point_feat, f_center)

        if loss_cfg.pool_mode == 'max':
            voxel_feat = torch_scatter.scatter_max(point_feat, unq_inv_pfn, dim=0)[0]
        elif loss_cfg.pool_mode == 'avg':
            voxel_feat = torch_scatter.scatter_mean(point_feat, unq_inv_pfn, dim=0)
        else:
            raise NotImplementedError

        if loss_cfg.get('vp', None):
            if loss_cfg.agg_vp:
                voxel_feat = torch.max(voxel_feat, dim=1)[0]
            else:
                voxel_feat = voxel_feat.view(voxel_feat.shape[0], -1)

        return voxel_feat
