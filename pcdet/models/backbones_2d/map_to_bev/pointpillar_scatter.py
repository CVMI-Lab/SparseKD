import torch
import torch.nn as nn

from pcdet.models.model_utils.basic_block_2d import Focus


class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES if 'in_channel' not in kwargs else kwargs['in_channel']
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1

        if kwargs.get('focus', None):
            self.focus = Focus()

    def forward(self, batch_dict, **kwargs):
        if kwargs.get('pillar_feature_tea', None):
            pillar_features, coords = batch_dict['pillar_features_tea'], batch_dict['voxel_coords_tea']
        else:
            pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1

        num_bev_features = self.num_bev_features // 4 if getattr(self, 'focus', None) else self.num_bev_features
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, num_bev_features * self.nz, self.ny, self.nx)
        
        if getattr(self, 'focus', None):
            batch_spatial_features = self.focus(batch_spatial_features)
        
        if kwargs.get('out_feature_name', None):
            batch_dict[kwargs['out_feature_name']] = batch_spatial_features
        else:
            batch_dict['spatial_features'] = batch_spatial_features
        return batch_dict


def point_pillar_scatter(num_bev_features, grid_size, pillar_features, coords):
    nx, ny, nz = grid_size
    batch_spatial_features = []
    batch_size = coords[:, 0].max().int().item() + 1

    for batch_idx in range(batch_size):
        spatial_feature = torch.zeros(
            num_bev_features,
            nz * nx * ny,
            dtype=pillar_features.dtype,
            device=pillar_features.device)

        batch_mask = coords[:, 0] == batch_idx
        this_coords = coords[batch_mask, :]
        indices = this_coords[:, 1] + this_coords[:, 2] * nx + this_coords[:, 3]
        indices = indices.type(torch.long)
        pillars = pillar_features[batch_mask, :]
        pillars = pillars.t()
        spatial_feature[:, indices] = pillars
        batch_spatial_features.append(spatial_feature)

    batch_spatial_features = torch.stack(batch_spatial_features, 0)
    batch_spatial_features = batch_spatial_features.view(batch_size, num_bev_features * nz, ny, nx)

    return batch_spatial_features
