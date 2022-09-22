import torch
import torch.nn as nn

from pcdet.utils import common_utils


class RotatedGridPool(nn.Module):
    def __init__(self, point_cloud_range, pool_cfg):
        super(RotatedGridPool, self).__init__()
        self.min_x = point_cloud_range[0]
        self.min_y = point_cloud_range[1]
        self.grid_size = pool_cfg.GRID_SIZE

    def forward(self, features_2d, rois, voxel_size, feature_map_stride):
        """
        Args:
            features_2d: (B, C, H, W)
            rois: (B, num_rois, 7 + C) tensor or list [num_rois, 7 + C]
            voxel_size
        Returns:
        """
        batch_size = features_2d.shape[0]
        height, width = features_2d.size(2), features_2d.size(3)

        pooled_features_list = []
        torch.backends.cudnn.enabled = False
        for b_id in range(batch_size):
            batch_rois = rois[b_id]
            if batch_rois.shape[0] == 0:
                continue

            valid_mask = batch_rois[:, 3] != 0
            # no valid box in the scene
            if valid_mask.sum() == 0:
                continue
            batch_rois = batch_rois[valid_mask]

            voxel_size_x = voxel_size[0]
            voxel_size_y = voxel_size[1]

            # Map global boxes coordinates to feature map coordinates
            x1 = (batch_rois[:, 0] - batch_rois[:, 3] / 2 - self.min_x) / (voxel_size_x * feature_map_stride)
            x2 = (batch_rois[:, 0] + batch_rois[:, 3] / 2 - self.min_x) / (voxel_size_x * feature_map_stride)
            y1 = (batch_rois[:, 1] - batch_rois[:, 4] / 2 - self.min_y) / (voxel_size_y * feature_map_stride)
            y2 = (batch_rois[:, 1] + batch_rois[:, 4] / 2 - self.min_y) / (voxel_size_y * feature_map_stride)

            angle, _ = common_utils.check_numpy_to_torch(batch_rois[:, 6])

            cosa = torch.cos(angle)
            sina = torch.sin(angle)

            theta = torch.stack((
                (x2 - x1) / (width - 1) * cosa, (x2 - x1) / (width - 1) * (-sina), (x1 + x2 - width + 1) / (width - 1),
                (y2 - y1) / (height - 1) * sina, (y2 - y1) / (height - 1) * cosa, (y1 + y2 - height + 1) / (height - 1)
            ), dim=1).view(-1, 2, 3).float()

            # Correct grid
            scale1 = (y2 - y1) / torch.clamp(x2 - x1, min=0.01)
            scale2 = (x2 - x1) / torch.clamp(y2 - y1, min=0.01)

            theta[:, 0, 1] *= scale1
            theta[:, 1, 0] *= scale2

            grid = nn.functional.affine_grid(
                theta,
                torch.Size((batch_rois.size(0), features_2d.size(1), self.grid_size, self.grid_size))
            )

            new_grid = grid.view(1, batch_rois.size(0), self.grid_size * self.grid_size, 2)

            pooled_features = nn.functional.grid_sample(
                features_2d[b_id].unsqueeze(0), new_grid
            ).squeeze(0)

            pooled_features = pooled_features.permute(1, 0, 2)
            pooled_features = pooled_features.view(
                batch_rois.size(0), features_2d.size(1), self.grid_size, self.grid_size
            )
            pooled_features_list.append(pooled_features)

        torch.backends.cudnn.enabled = True
        pooled_features = torch.cat(pooled_features_list, dim=0)

        return pooled_features
