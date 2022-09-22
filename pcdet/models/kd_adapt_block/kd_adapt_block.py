import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_

from pcdet.models.model_utils.basic_block_2d import build_block


class KDAdaptBlock(nn.Module):
    def __init__(self, model_cfg, point_cloud_range):
        super().__init__()
        self.model_cfg = model_cfg
        self.point_cloud_range = point_cloud_range
        self.align_module_list = []
 
        for adapt_layer_name, adapt_layer_cfg in self.model_cfg.MODULE.items():
            self.add_module(adapt_layer_name.lower(), BasicAdaptLayer(adapt_layer_cfg))
            self.align_module_list.append(getattr(self, adapt_layer_name.lower()))

    def forward(self, batch_dict):
        if self.training:
            for adapt_layer in self.align_module_list:
                batch_dict = adapt_layer(batch_dict)

        return batch_dict


class BasicAdaptLayer(nn.Module):
    def __init__(self, block_cfg):
        super().__init__()
        self.block_cfg = block_cfg
        self.in_feature_name = block_cfg.in_feature_name
        self.out_feature_name = block_cfg.out_feature_name

        self.build_adaptation_layer(block_cfg)

        self.init_weights(weight_init='xavier')

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def build_adaptation_layer(self, block_cfg):
        align_block = []

        in_channel = block_cfg.in_channel
        block_types = block_cfg.block_type
        num_filters = block_cfg.num_filters
        kernel_sizes = block_cfg.kernel_size
        num_strides = block_cfg.strides
        paddings = block_cfg.padding

        for i in range(len(num_filters)):
            align_block.extend(build_block(
                    block_types[i], in_channel, num_filters[i], kernel_size=kernel_sizes[i],
                    stride=num_strides[i], padding=paddings[i], bias=False
                ))

        self.adapt_layer = nn.Sequential(*align_block)

    def forward(self, batch_dict):
        in_feature = batch_dict[self.in_feature_name]

        out_feature = self.adapt_layer(in_feature)

        batch_dict[self.out_feature_name] = out_feature

        return batch_dict
