import torch.nn as nn


class PillarAdaptorTemplate(nn.Module):
    def __init__(self, model_cfg, in_channel, point_cloud_range):
        super(PillarAdaptorTemplate, self).__init__()
        self.model_cfg = model_cfg
        self.kd_only = self.model_cfg.KD_ONLY
        self.cal_loss = self.model_cfg.CAL_LOSS

        self.position = self.model_cfg.POSITION
        self.groups = self.model_cfg.CONV.get('GROUPS', None)
        self.point_could_range = point_cloud_range

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def build_loss(self):
        self.add_module('loss_func', getattr(nn, self.model_cfg.LOSS_CONFIG.NAME)())

    def group_teacher_voxel_coord_by_z(self, voxel_coord):
        """
        Args:
            voxel_coord: [N, 3] (z, y, x)

        Returns:

        """
        assert voxel_coord[:, 0].max() <= (self.groups - 1)

        group_mask_list = []
        for idx in range(self.groups):
            mask = voxel_coord[:, 0] == idx
            group_mask_list.append(mask)

        return group_mask_list
