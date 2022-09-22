import torch
import torch.nn as nn
from .pillar_adaptor_template import PillarAdaptorTemplate
from pcdet.models.model_utils.batch_norm_utils import get_norm_layer
from pcdet.models.model_utils import basic_block_2d


class SparseNaiveAdaptor(PillarAdaptorTemplate):
    def __init__(self, model_cfg, in_channel, point_cloud_range):
        super(SparseNaiveAdaptor, self).__init__(model_cfg, in_channel, point_cloud_range)

        use_norm = self.model_cfg.CONV.USE_NORM
        use_act = self.model_cfg.CONV.USE_ACT
        kernel_size = self.model_cfg.CONV.KERNEL_SIZE

        num_filters = self.model_cfg.CONV.NUM_FILTERS
        num_filters = [in_channel] + num_filters

        norm_layer = get_norm_layer(self.model_cfg.CONV.get('NORM_TYPE', 'BatchNorm2d'))

        if self.model_cfg.get('DOWNSAMPLE', None):
            self.downsample_block = basic_block_2d.build_downsample_block(
                self.model_cfg.DOWNSAMPLE.MODE, in_channel, in_channel, norm_layer
            )

        group_list = []
        for i in range(len(num_filters) - 1):
            group_list.append(
                nn.Conv2d(num_filters[i], num_filters[i+1], kernel_size,
                          padding=int((kernel_size-1) / 2), bias=(not use_norm)))
            if use_norm:
                group_list.append(norm_layer(num_filters[i+1], eps=1e-3, momentum=0.01))
            if use_act and (i < len(num_filters) - 2):
                group_list.append(nn.ReLU())

        self.group_block = nn.Sequential(*group_list)

        self.final_act = nn.ReLU()

        self.init_weights()
        if self.cal_loss:
            self.build_loss()

    def forward(self, batch_dict):
        if not self.training and self.kd_only:
            return batch_dict

        # [B, C1, H, W]
        dense_feat_stu = batch_dict['spatial_features']

        if hasattr(self, 'downsample_block'):
            dense_feat_stu = self.downsample_block(dense_feat_stu)

        # import ipdb; ipdb.set_trace(context=20)
        # A = dense_feat_stu.cpu().numpy()
        # mask = (A[0] > 0).sum(axis=0)
        # mask_ds = mask > 0
        # import matplotlib.pyplot as plt
        # plt.imshow(mask_ds, cmap='gray', vmin=0, vmax=1, interpolation='none')
        # plt.show()
        dense_pillar_features_pre_act = self.group_block(dense_feat_stu)

        batch_dict['pillar_adaptor_features_pre-act'] = dense_pillar_features_pre_act

        dense_pillar_features = self.final_act(dense_pillar_features_pre_act)
        batch_dict['pillar_adaptor_features'] = dense_pillar_features

        # B = dense_pillar_features.cpu().numpy()
        # mask = (B[0] > 0).sum(axis=0)
        # mask_ad = mask > 0
        # plt.imshow(mask_ad, cmap='gray', vmin=0, vmax=1, interpolation='none')
        # plt.show()

        return batch_dict

    def get_loss(self, batch_dict, tb_dict):
        dense_pillar_feat = batch_dict[self.position]
        sparse_voxel_tensor_tea = batch_dict['voxel_features_tea']

        feat_shape_stu = dense_pillar_feat.shape
        batch_size = feat_shape_stu[0]

        dense_pillar_group_feat = dense_pillar_feat.view(
            batch_size, self.groups, feat_shape_stu[1] // self.groups, feat_shape_stu[2], feat_shape_stu[3]
        )

        # [M, 4] (batch id, z, y, x)
        voxel_coords_tea = sparse_voxel_tensor_tea.indices
        # [M, C2]
        sparse_voxel_feat_tea = sparse_voxel_tensor_tea.features

        loss = 0
        for b_idx in range(batch_size):
            batch_mask = voxel_coords_tea[:, 0] == b_idx
            this_feat_tea = sparse_voxel_feat_tea[batch_mask]
            this_coord_tea = voxel_coords_tea[batch_mask]

            group_mask_list = self.group_teacher_voxel_coord_by_z(this_coord_tea[:, 1:])

            for g_idx, mask in enumerate(group_mask_list):
                this_coord_group_tea = this_coord_tea[mask]
                this_feat_stu = dense_pillar_group_feat[b_idx, g_idx, :, this_coord_group_tea[:, 2].long(), this_coord_group_tea[:, 3].long()]
                loss += self.loss_func(this_feat_stu.t(), this_feat_tea[mask])

        loss = self.model_cfg.LOSS_CONFIG.WEIGHT * loss / (batch_size * self.groups)

        tb_dict['kd_pill_ls'] = loss.item()

        return loss, tb_dict
