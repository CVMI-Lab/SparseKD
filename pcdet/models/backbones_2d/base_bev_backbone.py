import numpy as np
import torch
import torch.nn as nn
from pcdet.models.model_utils.efficientnet_utils import get_act_layer
from pcdet.models.model_utils.basic_block_2d import build_block, build_deconv_block, Focus
from pcdet.models.model_utils.batch_norm_utils import get_norm_layer


class BaseBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        act_fn = get_act_layer(self.model_cfg.get('ACT_FN', 'ReLU'))
        norm_layer = get_norm_layer(self.model_cfg.get('NORM_TYPE', 'BatchNorm2d'))

        if self.model_cfg.get('IN_CHANNEL', None):
            input_channels = self.model_cfg.IN_CHANNEL

        self.input_feature_name = self.model_cfg.get('IN_FEATURE_NAME', 'spatial_features')

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        if model_cfg.get('WIDTH', None):
            num_filters = (np.array(num_filters, dtype=np.int32) * model_cfg.WIDTH).astype(int)
            num_upsample_filters = (np.array(num_upsample_filters, dtype=np.int32) * model_cfg.WIDTH).astype(int)

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                # TODO: only support downsampling here, cannot cover the case of stride=1
                Focus() if self.model_cfg.get('FOCUS', None) and layer_strides[idx] > 1 else nn.Identity(),
                nn.Conv2d(
                    c_in_list[idx]*4 if self.model_cfg.get('FOCUS', None) and layer_strides[idx] > 1 else c_in_list[idx],
                    num_filters[idx], kernel_size=3, 
                    stride=layer_strides[idx] if not self.model_cfg.get('FOCUS', None) else 1, 
                    padding=1, bias=False
                ),
                norm_layer(num_filters[idx], eps=1e-3, momentum=0.01),
                act_fn()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend(
                    build_block(
                        self.model_cfg.get('CONV_BLOCK', 'BasicBlock2D'), num_filters[idx],
                        num_filters[idx], act_fn=act_fn, kernel_size=3, padding=1, bias=False,
                        norm_layer=norm_layer
                    ),
                )
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(
                        build_deconv_block(
                            self.model_cfg.get('DECONV_BLOCK', 'ConvTranspose2dBlock'),
                            num_filters[idx], num_upsample_filters[idx],
                            kernel_size=upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False,
                            act_fn=act_fn,
                            norm_layer=norm_layer
                        )
                    )
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            kernel_size=stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        act_fn()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(
                build_deconv_block(
                    self.model_cfg.get('DECONV_BLOCK', 'ConvTranspose2dBlock'),
                    c_in, c_in,
                    kernel_size=upsample_strides[-1],
                    stride=upsample_strides[-1], bias=False,
                    act_fn=act_fn
                )
            )

        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict[self.input_feature_name]
        ups = []

        if self.model_cfg.get('SKIP_CONNECT_FEAT', 'None') != 'None':
            skip_feat_list = data_dict[self.model_cfg.SKIP_CONNECT_FEAT]
        else:
            skip_feat_list = None

        # ret_dict = {}
        intermediate_features = []
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            # stride = int(spatial_features.shape[2] / x.shape[2])
            # ret_dict['spatial_features_%dx' % stride] = x
            if skip_feat_list is not None:
                x += skip_feat_list[i]

            if len(self.deblocks) > 0:
                # out = self.deblocks[i](x)
                out = x
                for j, layer in enumerate(self.deblocks[i]):
                    out = layer(out)
                    if j == 1:
                        intermediate_features.append(out)
                ups.append(out)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x
        if len(intermediate_features) > 0:
            data_dict['spatial_features_2d_pre-relu'] = torch.cat(intermediate_features, dim=1)
        # data_dict['spatial_features_4x'] = ret_dict['spatial_features_4x']

        return data_dict


class BaseBEVBackboneSkipPost(BaseBEVBackbone):
    def __init__(self, model_cfg, input_channels):
        super().__init__(model_cfg, input_channels)
    
    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict[self.input_feature_name]
        ups = []

        skip_feat_list = data_dict[self.model_cfg.SKIP_CONNECT_FEAT]

        # ret_dict = {}
        intermediate_features = []
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            # stride = int(spatial_features.shape[2] / x.shape[2])
            # ret_dict['spatial_features_%dx' % stride] = x                

            if len(self.deblocks) > 0:
                # out = self.deblocks[i](x)
                out = x
                for j, layer in enumerate(self.deblocks[i]):
                    out = layer(out)
                    if j == 1:
                        intermediate_features.append(out)
                out += skip_feat_list[i]
                ups.append(out)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x
        if len(intermediate_features) > 0:
            data_dict['spatial_features_2d_pre-relu'] = torch.cat(intermediate_features, dim=1)
        # data_dict['spatial_features_4x'] = ret_dict['spatial_features_4x']

        return data_dict
