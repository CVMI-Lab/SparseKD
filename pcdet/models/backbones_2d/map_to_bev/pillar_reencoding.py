import torch
import torch.nn as nn

from .pointpillar_scatter import PointPillarScatter
from pcdet.models.model_utils.efficientnet_utils import get_act_layer
from pcdet.models.model_utils.basic_block_2d import build_block


class PillarReencoding(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        input_channels = model_cfg.IN_CHANNEL
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.pillar_scatter = PointPillarScatter(model_cfg, grid_size, in_channel=input_channels, **kwargs)

        act_fn = get_act_layer(self.model_cfg.get('ACT_FN', 'ReLU'))

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                act_fn()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend(
                    build_block(
                        self.model_cfg.get('CONV_BLOCK', 'BasicBlock2D'), num_filters[idx],
                        num_filters[idx], act_fn=act_fn, kernel_size=3, padding=1, bias=False
                    ),
                )
            self.blocks.append(nn.Sequential(*cur_layers))

    def forward(self, batch_dict, **kwargs):
        data_dict = self.pillar_scatter(batch_dict, **kwargs)

        spatial_features = data_dict['spatial_features']
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

        data_dict['spatial_features'] = x

        return data_dict
