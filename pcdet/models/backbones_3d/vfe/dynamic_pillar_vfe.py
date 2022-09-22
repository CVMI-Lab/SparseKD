import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pcdet.models.model_utils.positional_encoding import sinusoidal_positional_encoding_2d

try:
    import torch_scatter
except Exception as e:
    # Incase someone doesn't want to use dynamic pillar vfe and hasn't installed torch_scatter
    pass

from .vfe_template import VFETemplate
from .dynamic_kp_vfe import VirtualPointAggerator


class PFNLayerV2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()
        
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)
        
        self.relu = nn.ReLU()

    def forward(self, inputs, unq_inv):
        x = self.linear(inputs)
        x = self.norm(x) if self.use_norm else x
        x = self.relu(x)

        if self.last_vfe:
            return x
        else:
            x_max = torch_scatter.scatter_max(x, unq_inv, dim=0)[0]
            x_concatenated = torch.cat([x, x_max[unq_inv, :]], dim=1)
            return x_concatenated


class DynamicPillarVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, grid_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayerV2(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        self.scale_xy = grid_size[0] * grid_size[1]
        self.scale_y = grid_size[1]
        
        self.grid_size = torch.tensor(grid_size).cuda().int()
        self.voxel_size = torch.tensor(voxel_size).cuda()
        self.point_cloud_range = torch.tensor(point_cloud_range).cuda()

        if self.model_cfg.get('KERNEL', None) and self.model_cfg.KERNEL.ENABLED:
            # build kp kernel
            # [k, 3] relative lidar point coordinates
            self.kp_aggregator = VirtualPointAggerator(self.model_cfg.KERNEL, self.voxel_size)

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def forward(self, batch_dict, **kwargs):
        points = batch_dict['points']  # (batch_idx, x, y, z, i, e)

        points_coords = torch.floor((points[:, [1, 2]] - self.point_cloud_range[[0, 1]]) / self.voxel_size[[0,1]]).int()
        mask = ((points_coords >= 0) & (points_coords < self.grid_size[[0, 1]])).all(dim=1)
        points = points[mask]
        points_coords = points_coords[mask]
        points_xyz = points[:, [1, 2, 3]].contiguous()

        merge_coords = points[:, 0].int() * self.scale_xy + \
                       points_coords[:, 0] * self.scale_y + \
                       points_coords[:, 1]
        
        unq_coords, unq_inv, _ = torch.unique(merge_coords, return_inverse=True, return_counts=True, dim=0)

        points_mean = torch_scatter.scatter_mean(points_xyz, unq_inv, dim=0)
        f_cluster = points_xyz - points_mean[unq_inv, :]

        f_center = torch.zeros_like(points_xyz)
        f_center[:, 0] = points_xyz[:, 0] - (points_coords[:, 0].to(points_xyz.dtype) * self.voxel_x + self.x_offset)
        f_center[:, 1] = points_xyz[:, 1] - (points_coords[:, 1].to(points_xyz.dtype) * self.voxel_y + self.y_offset)
        f_center[:, 2] = points_xyz[:, 2] - self.z_offset

        if self.use_absolute_xyz:
            features = [points[:, 1:], f_cluster, f_center]
        else:
            features = [points[:, 4:], f_cluster, f_center]
        
        if self.with_distance:
            points_dist = torch.norm(points[:, 1:4], 2, dim=1, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)
        
        if self.training and not getattr(self, 'is_teacher', False) and self.model_cfg.get('VFE_KD', None) and\
                self.model_cfg.VFE_KD.get('PFN_INDS_TEA', None):
            unq_inv_pfn = batch_dict['unq_inv_pfn_tea']
        else:
            unq_inv_pfn = unq_inv

        for pfn in self.pfn_layers:
            features = pfn(features, unq_inv_pfn)

        if hasattr(self, 'kp_aggregator'):
            # [n_points, k, c_out]
            features = self.kp_aggregator(features, f_center)
            # [n_voxels, k, c_out]
            features_max = torch_scatter.scatter_max(features, unq_inv, dim=0)[0]
            # [n_voxels, c_out]
            features_max = torch.max(features_max, dim=1)[0]
        else:
            features_max = torch_scatter.scatter_max(features, unq_inv, dim=0)[0]
        
        # generate voxel coordinates
        unq_coords = unq_coords.int()
        voxel_coords = torch.stack((unq_coords // self.scale_xy,
                                   (unq_coords % self.scale_xy) // self.scale_y,
                                   unq_coords % self.scale_y,
                                   torch.zeros(unq_coords.shape[0]).to(unq_coords.device).int()
                                   ), dim=1)
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]

        if self.model_cfg.get('POS_EMB', None) and self.model_cfg.POS_EMB.ENABLED:
            pos_emb = self.generate_positional_encoding(voxel_coords, self.model_cfg.POS_EMB)
            features_max += pos_emb

        batch_dict['pillar_features'] = features_max
        batch_dict['voxel_coords'] = voxel_coords

        if self.model_cfg.get('VFE_KD', None):
            batch_dict['point_features'] = features
            if self.model_cfg.VFE_KD.get('SAVE_INDS', None):
                batch_dict['unq_inv_pfn'] = unq_inv

            if not hasattr(self, 'is_teacher') and self.model_cfg.VFE_KD.get('KERNEL', None) and \
                    self.model_cfg.VFE_KD.KERNEL.ENABLED:
                batch_dict['f_center'] = f_center

        return batch_dict

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


class DynamicPillarVFETea(DynamicPillarVFE):
    def __init__(self, model_cfg, num_point_features, voxel_size, grid_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg, num_point_features, voxel_size, grid_size, point_cloud_range, **kwargs)
        
        voxel_size_pre = self.model_cfg.VOXEL_SIZE_PRE
        self.voxel_x_pre = voxel_size_pre[0]
        self.voxel_y_pre = voxel_size_pre[1]
        self.voxel_z_pre = voxel_size_pre[2]

        self.x_offset_pre = self.voxel_x_pre / 2 + point_cloud_range[0]
        self.y_offset_pre = self.voxel_y_pre / 2 + point_cloud_range[1]
        self.z_offset_pre = self.voxel_z_pre / 2 + point_cloud_range[2]

        self.voxel_size_pre = torch.from_numpy(np.array(voxel_size_pre)).cuda().float()
        grid_size_pre = (point_cloud_range[3:6] - point_cloud_range[0:3]) / np.array(voxel_size_pre)
        grid_size_pre = np.round(grid_size_pre).astype(np.int64)


        self.scale_xy_pre = grid_size_pre[0] * grid_size_pre[1]
        self.scale_y_pre = grid_size_pre[1]

        self.grid_size_pre = torch.tensor(grid_size_pre).cuda().int()

    def forward(self, batch_dict, **kwargs):
        points = batch_dict['points']  # (batch_idx, x, y, z, i, e)
        merge_coords, points_coords, points = self.cal_merge_coords(
            points, self.grid_size_pre, self.voxel_size_pre, self.scale_xy_pre, self.scale_y_pre
        )
        points_xyz = points[:, [1, 2, 3]].contiguous()

        unq_coords, unq_inv, _ = torch.unique(merge_coords, return_inverse=True, return_counts=True, dim=0)

        points_mean = torch_scatter.scatter_mean(points_xyz, unq_inv, dim=0)
        f_cluster = points_xyz - points_mean[unq_inv, :]

        f_center = torch.zeros_like(points_xyz)
        f_center[:, 0] = points_xyz[:, 0] - (points_coords[:, 0].to(points_xyz.dtype) * self.voxel_x_pre + self.x_offset_pre)
        f_center[:, 1] = points_xyz[:, 1] - (points_coords[:, 1].to(points_xyz.dtype) * self.voxel_y_pre + self.y_offset_pre)
        f_center[:, 2] = points_xyz[:, 2] - self.z_offset_pre

        if self.use_absolute_xyz:
            features = [points[:, 1:], f_cluster, f_center]
        else:
            features = [points[:, 4:], f_cluster, f_center]
        
        if self.with_distance:
            points_dist = torch.norm(points[:, 1:4], 2, dim=1, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)

        for pfn in self.pfn_layers:
            features = pfn(features, unq_inv)
        
        merge_coords_final, _, _ = self.cal_merge_coords(
            points, self.grid_size, self.voxel_size, self.scale_xy, self.scale_y
        )
        assert merge_coords_final.shape == merge_coords.shape

        unq_coords_final, unq_inv_final, _ = torch.unique(merge_coords_final, return_inverse=True, return_counts=True, dim=0)

        features_max = torch_scatter.scatter_max(features, unq_inv_final, dim=0)[0]

        # generate voxel coordinates
        unq_coords_final = unq_coords_final.int()
        voxel_coords = torch.stack((unq_coords_final // self.scale_xy,
                                   (unq_coords_final % self.scale_xy) // self.scale_y,
                                   unq_coords_final % self.scale_y,
                                   torch.zeros(unq_coords_final.shape[0]).to(unq_coords_final.device).int()
                                   ), dim=1)
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]
        
        batch_dict['pillar_features'] = features_max
        batch_dict['voxel_coords'] = voxel_coords

        if self.model_cfg.get('SAVE_TEA_FEAT', None):
            features_max_pre = torch_scatter.scatter_max(features, unq_inv, dim=0)[0]
            
            unq_coords = unq_coords.int()
            voxel_coords_pre = torch.stack((unq_coords // self.scale_xy_pre,
                                    (unq_coords % self.scale_xy_pre) // self.scale_y_pre,
                                    unq_coords % self.scale_y_pre,
                                    torch.zeros(unq_coords.shape[0]).to(unq_coords.device).int()
                                    ), dim=1)

            voxel_coords_pre = voxel_coords_pre[:, [0, 3, 2, 1]]

            batch_dict['pillar_features_tea'] = features_max_pre
            batch_dict['voxel_coords_tea'] = voxel_coords_pre

        return batch_dict

    def cal_merge_coords(self, points, grid_size, voxel_size, scale_xy, scale_y):
        points_coords = torch.floor((points[:, [1, 2]] - self.point_cloud_range[[0, 1]]) / voxel_size[[0,1]]).int()
        mask = ((points_coords >= 0) & (points_coords < grid_size[[0, 1]])).all(dim=1)
        points = points[mask]
        points_coords = points_coords[mask]

        merge_coords = points[:, 0].int() * scale_xy + \
                       points_coords[:, 0] * scale_y + \
                       points_coords[:, 1]

        return merge_coords, points_coords, points
