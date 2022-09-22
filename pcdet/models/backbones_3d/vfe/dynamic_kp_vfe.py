import torch
import numpy as np
import torch.nn as nn

try:
    import torch_scatter
except Exception as e:
    # Incase someone doesn't want to use dynamic pillar vfe and hasn't installed torch_scatter
    pass

from .vfe_template import VFETemplate, PFNLayerKP
from pcdet.utils import common_utils


class DynamicKPVFE(VFETemplate):
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
            pfn_layers.append(PFNLayerKP(in_filters, out_filters, use_norm=self.use_norm))

        self.pfn_layers = nn.Sequential(*pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        self.scale_xyz = grid_size[0] * grid_size[1] * grid_size[2]
        self.scale_yz = grid_size[1] * grid_size[2]
        self.scale_z = grid_size[2]

        self.grid_size = torch.tensor(grid_size).cuda().int()
        self.voxel_size = torch.tensor(voxel_size).cuda()
        self.point_cloud_range = torch.tensor(point_cloud_range).cuda()

        # [k, 3] relative lidar point coordinates
        self.kp_aggregator = VirtualPointAggerator(self.model_cfg.KERNEL, self.voxel_size)

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def forward(self, batch_dict, **kwargs):
        points = batch_dict['points']  # (batch_idx, x, y, z, i, e)

        # calculate voxel coordinate for points in the point cloud range
        point_coords = torch.floor(
            (points[:, 1:4] - self.point_cloud_range[0:3] - 0.5 * self.voxel_size) / self.voxel_size).int()
        mask = ((point_coords >= 0) & (point_coords < self.grid_size)).all(dim=1)
        points = points[mask]
        point_coords = point_coords[mask]

        points_xyz = points[:, [1, 2, 3]].contiguous()
        merge_coords = points[:, 0].int() * self.scale_xyz + point_coords[:, 0] * self.scale_yz + \
                        point_coords[:, 1] * self.scale_z + point_coords[:, 2]

        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True)

        points_mean = torch_scatter.scatter_mean(points_xyz, unq_inv, dim=0)
        f_cluster = points_xyz - points_mean[unq_inv, :]

        # relative x y z compared to the voxel center
        f_center = torch.zeros_like(points_xyz)
        f_center[:, 0] = points_xyz[:, 0] - (point_coords[:, 0].to(points_xyz.dtype) * self.voxel_x + self.x_offset)
        f_center[:, 1] = points_xyz[:, 1] - (point_coords[:, 1].to(points_xyz.dtype) * self.voxel_y + self.y_offset)
        f_center[:, 2] = points_xyz[:, 2] - (point_coords[:, 2].to(points_xyz.dtype) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            features = [points[:, 1:], f_cluster, f_center]
        else:
            features = [points[:, 4:], f_cluster, f_center]

        if self.with_distance:
            points_dist = torch.norm(points[:, 1:4], 2, dim=1, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)

        # calculate distance between relative points to kernel points
        kernel_features = self.kp_aggregator(features, f_center)

        # forward convolution block
        kernel_features = self.pfn_layers(kernel_features)

        # aggregate points features to voxels
        kernel_features = torch_scatter.scatter_max(kernel_features, unq_inv, dim=0)[0]

        # aggregate features from kernel points
        kernel_features_pooled = torch.max(kernel_features, dim=1)[0]

        # generate voxel coordinates
        unq_coords = unq_coords.int()
        voxel_coords = torch.stack((unq_coords // self.scale_xyz,
                                    (unq_coords % self.scale_xyz) // self.scale_yz,
                                    (unq_coords % self.scale_yz) // self.scale_z,
                                    unq_coords % self.scale_z), dim=1)
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]

        out_feat_name = self.model_cfg.get('OUT_FEAT_NAME', 'voxel_features')
        batch_dict[out_feat_name] = kernel_features_pooled
        batch_dict['voxel_coords'] = voxel_coords
        return batch_dict


class VirtualPointAggerator(nn.Module):
    def __init__(self, kernel_cfg, voxel_size):
        super(VirtualPointAggerator, self).__init__()
        self.kernel_cfg = kernel_cfg
        self.sigma = kernel_cfg.SIGMA
        self.voxel_size = voxel_size
        self.influence = kernel_cfg.INFLUENCE

        self.max_distance = torch.sqrt(torch.norm(voxel_size))

        # [k, 3] relative lidar point coordinates
        self.kp_kernel = self.build_kernel(kernel_cfg)

    def forward(self, features, relative_pos):
        # calculate distance between relative points to kernel points
        sq_distance = common_utils.pair_distance_gpu(relative_pos, self.kp_kernel)

        # calculate weights for points to kernel points.
        # [n_points, K]
        weights = self.cal_weights(sq_distance)

        # [n_points, K, C]
        kernel_features = features.unsqueeze(1) * weights.unsqueeze(-1)

        return kernel_features

    def cal_weights(self, dist):
        """
        Args:
            kernel_cfg:
            dist: [n_points, K].

        Returns:
            weights: [n_points, K]
        """
        if self.influence == 'linear':
            weights = torch.clamp(self.max_distance - torch.sqrt(dist), min=0.0)
        elif self.influence == 'gaussian':
            weights = self.radius_gaussian(dist, self.sigma)
        else:
            raise NotImplementedError

        return weights

    @staticmethod
    def radius_gaussian(sq_r, sig, eps=1e-9):
        """
        Compute a radius gaussian (gaussian of distance)
        :param sq_r: input radius [dn, ..., d1, d0]
        :param sig: extents of gaussian [d1, d0] or [d0] or float
        :return: gaussian of sq_r [dn, ..., d1, d0]
        """
        return torch.exp(-sq_r / (2 * sig ** 2 + eps))

    def build_kernel(self, kernel_cfg):
        if kernel_cfg.MODE == 'cubic':
            return self.build_cubic_kernel(kernel_cfg)
        else:
            raise NotImplementedError

    def build_cubic_kernel(self, kernel_cfg):
        """

        Args:
            kernel_cfg

        Returns:
            kernel_points: [K, 3]
        """
        n_points_xyz = kernel_cfg.N_POINTS_XYZ
        # [3]
        n_points_xyz = torch.from_numpy(np.array(n_points_xyz)).cuda()
        # [3]
        if kernel_cfg.get('UNIFORM', True):
            virtual_point_step_xyz = (self.voxel_size / (n_points_xyz.float() + 1)).unsqueeze(0)
        else:
            virtual_point_step_xyz = (self.voxel_size / n_points_xyz.float()).unsqueeze(0)

        # [n_points_x * n_points_y * n_points_z, 3]
        point_index = torch.arange(0, n_points_xyz[0] * n_points_xyz[1] * n_points_xyz[2]).unsqueeze(1).repeat([1, 3]).cuda()

        point_index[:, 0] = point_index[:, 0] % n_points_xyz[0] + 1
        point_index[:, 1] = (point_index[:, 1] // n_points_xyz[0]) % n_points_xyz[1] + 1
        point_index[:, 2] = (point_index[:, 2] // (n_points_xyz[0] * n_points_xyz[1])) % n_points_xyz[2] + 1

        if kernel_cfg.get('UNIFORM', True):
            kernel_points = point_index.float() * virtual_point_step_xyz - self.voxel_size / 2
        else:
            kernel_points = point_index.float() * virtual_point_step_xyz - (self.voxel_size + virtual_point_step_xyz) / 2

        return kernel_points
