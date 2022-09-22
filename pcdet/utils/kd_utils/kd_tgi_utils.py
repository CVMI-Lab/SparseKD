"""
Author: Runyu Ding
Copyright 2022 - Present
"""

import torch
import torch.nn as nn
from collections import OrderedDict

from ...utils.spconv_utils import spconv, find_all_spconv_keys

try:
    from torch.nn.modules.conv import _ConvTransposeNd as _ConvTransposeNd
except:
    from torch.nn.modules.conv import _ConvTransposeMixin as _ConvTransposeNd


def _select_bn_idx(bn_weight, topk, abs=True):
    if abs:
        descending_idx = torch.abs(bn_weight).argsort(descending=True)
    else:
        descending_idx = bn_weight.argsort(descending=True)
    return descending_idx[:topk]


def _save_dict_to_module(module, destination, prefix):
    for name, param in module._parameters.items():
        if param is not None:
            destination[prefix + name] = type(module)
    for name, buf in module._buffers.items():
        if buf is not None:  # and name not in self._non_persistent_buffers_set:  # pytorch 1.11
            destination[prefix + name] = type(module)


def _map_state_dict_to_module(model, destination, prefix):
    r"""Saves module state to `destination` dictionary, containing a state
    of the module, but not its descendants. This is called on every
    submodule in :meth:`~torch.nn.Module.state_dict`.

    In rare cases, subclasses can achieve class-specific behavior by
    overriding this method with custom logic.

    Args:
        destination (dict): a dict where state will be stored
        prefix (str): the prefix for parameters and buffers used in this
            module
    """
    _save_dict_to_module(model, destination, prefix)
    for name, module in model._modules.items():
        if module is not None:
            _map_state_dict_to_module(module, destination, prefix + name + '.')


def is_last_layer(p, model_state):
    if len(p.split('.')) == 3:  # second
        return True
    if 'dense_head.heads_list' in p:
        # dense_head.heads_list.0.center.1.bias
        prefix_list = p.split('.')
        prefix_list[4] = chr(ord(prefix_list[4]) + 1)
        for k in model_state.keys():
            if '.'.join(prefix_list[:5]) in k:
                return False
        return True
    return False


def _remap_to_current_model_by_bn_scale(model, model_state, cfg):
    stu_model_state = {}  # self.state_dict().copy()
    self_model_state = model.state_dict().copy()

    stu_bn_idx_dict = {None: None}
    stu_bn_idx_list = [None]
    bn_weight_key = None
    param_queue = []
    param_name_to_class_type = OrderedDict()
    _map_state_dict_to_module(param_name_to_class_type, '')
    for k in model_state.keys():
        k_list = k.split('.')
        param_queue.append(k)
        if issubclass(param_name_to_class_type[k], nn.modules.batchnorm._BatchNorm) and \
                k_list[-1] == 'weight':
            # e.g. backbone_3d.conv1.0.bn1.weight
            bn_weight_key = k
        elif issubclass(param_name_to_class_type[k], nn.modules.batchnorm._BatchNorm) and \
                k_list[-1] == 'running_var':
            # e.g. backbone_3d.conv1.0.bn1.running_var
            _stu_bn_idx = _select_bn_idx(
                model_state[bn_weight_key], self_model_state[bn_weight_key].shape[-1],
                abs=cfg.BN_SCALE.ABS
            )
            stu_bn_idx_dict[k] = _stu_bn_idx
            stu_bn_idx_list.append(k)
            _remap_param_in_queue(
                param_queue, model_state, stu_model_state, self_model_state, stu_bn_idx_dict,
                stu_bn_idx_list, param_name_to_class_type
            )
            param_queue.clear()
        elif is_last_layer(k, model_state) and k_list[-1] == 'bias':
            # last layer, a conv layer
            stu_bn_idx_list.append(None)
            _remap_param_in_queue(
                param_queue, model_state, stu_model_state, self_model_state, stu_bn_idx_dict,
                stu_bn_idx_list, param_name_to_class_type
            )
            param_queue.clear()
            stu_bn_idx_list.pop()
    assert len(model_state) == len(stu_model_state)
    return stu_model_state


def _remap_to_current_model_by_ofa(model, model_state, cfg):
    stu_model_state = {}
    self_model_state = model.state_dict().copy()
    stu_conv_idx_dict = {None: None}
    stu_conv_idx_list = [None]
    conv_weight_key = None
    topk = None
    param_queue = []
    param_name_to_class_type = OrderedDict()
    _map_state_dict_to_module(model, param_name_to_class_type, '')
    for k in model_state.keys():
        k_list = k.split('.')
        param_queue.append(k)
        if issubclass(param_name_to_class_type[k], _ConvTransposeNd) and \
                k_list[-1] == 'weight':
            conv_weight_key = k
            shrink_dim_idx = [0, 2, 3]
            topk = self_model_state[conv_weight_key].shape[1]
        elif issubclass(param_name_to_class_type[k], nn.modules.conv._ConvNd) and \
                k_list[-1] == 'weight':
            # e.g. backbone_2d.blocks.0.0.weight
            conv_weight_key = k
            shrink_dim_idx = [1, 2, 3]
            topk = self_model_state[conv_weight_key].shape[0]
        elif issubclass(param_name_to_class_type[k], spconv.conv.SparseConvolution) and \
                k_list[-1] == 'weight':
            #  e.g. backbone_3d.conv0.0.0.weight
            conv_weight_key = k
            shrink_dim_idx = [0, 1, 2, 3]
            topk = self_model_state[conv_weight_key].shape[-1]
        elif issubclass(param_name_to_class_type[k], nn.modules.batchnorm._BatchNorm) and \
                k_list[-1] == 'running_var':
            # e.g. backbone_3d.conv1.0.bn1.running_var
            _input_dim = _find_conv_input_dim(conv_weight_key, param_name_to_class_type)
            _input_idx = _find_input_idx(
                conv_weight_key, model_state, self_model_state, stu_conv_idx_dict, stu_conv_idx_list,
                param_name_to_class_type
            )
            _stu_conv_idx = _select_conv_idx(
                model_state[conv_weight_key], topk, dim=shrink_dim_idx, input_dim=_input_dim,
                input_idx=_input_idx, descending=(cfg.OFA.l1_norm == 'max')
            )
            stu_conv_idx_dict[conv_weight_key] = _stu_conv_idx
            stu_conv_idx_list.append(conv_weight_key)
            _remap_param_in_queue(
                param_queue, model_state, stu_model_state, self_model_state, stu_conv_idx_dict,
                stu_conv_idx_list, param_name_to_class_type,
            )
            param_queue.clear()
        elif is_last_layer(k, model_state) and k_list[-1] == 'bias':
            # last layer, a conv layer
            stu_conv_idx_list.append(None)
            _remap_param_in_queue(
                param_queue, model_state, stu_model_state, self_model_state, stu_conv_idx_dict,
                stu_conv_idx_list, param_name_to_class_type
            )
            param_queue.clear()
            stu_conv_idx_list.pop()
    assert len(model_state) == len(stu_model_state)
    return stu_model_state


def _remap_to_current_model_by_fnav1(model, model_state, cfg):
    stu_model_state = model.state_dict().copy()
    spconv_keys = find_all_spconv_keys(model)
    for k in model_state.keys():
        curr_v = model_state[k].clone()
        stu_model_state[k] = _narrow_weight(k, curr_v, stu_model_state[k], spconv_keys)
    return stu_model_state


def _remap_to_current_model_by_fnav2(model, model_state, cfg):
    stu_model_state = {}  # self.state_dict().copy()
    self_model_state = model.state_dict().copy()

    stu_bn_idx_dict = {None: None}
    stu_bn_idx_list = [None]
    bn_weight_key = None
    param_name_to_class_type = OrderedDict()
    _map_state_dict_to_module(model, param_name_to_class_type, '')

    spconv_keys = find_all_spconv_keys(model)

    for k in model_state.keys():
        k_list = k.split('.')
        curr_v = model_state[k].clone()
        # update stu_bn_idx_list & stu_bn_idx_dict
        if issubclass(param_name_to_class_type[k], nn.modules.batchnorm._BatchNorm) and \
                k_list[-1] == 'weight':
            # e.g. backbone_3d.conv1.0.bn1.weight
            bn_weight_key = k
        elif issubclass(param_name_to_class_type[k], nn.modules.batchnorm._BatchNorm) and \
                k_list[-1] == 'running_var':
            # e.g. backbone_3d.conv1.0.bn1.running_var
            stu_bn_idx_dict[k] = torch.arange(self_model_state[bn_weight_key].shape[-1])
            stu_bn_idx_list.append(k)
        # map_to_bev & map_to_densehead
        if len(model_state[k].shape) == 4 and 'backbone_2d' in k and \
                'backbone_2d' not in stu_bn_idx_list[-1]:
            # map to bev
            stu_bn_idx_list.append(None)
            _stu_input_dim_idx = _find_input_idx_with_key(
                k, model_state, self_model_state, stu_bn_idx_dict, stu_bn_idx_list,
                param_name_to_class_type, key='map_to_bev'
            )
            stu_bn_idx_list.pop()
            curr_v = curr_v.index_select(1, _stu_input_dim_idx)
        elif len(model_state[k].shape) == 4 and 'dense_head.shared_conv' in k and \
                'dense_head.shared_conv' not in stu_bn_idx_list[-1]:
            # map to densehead
            stu_bn_idx_list.append(None)
            _stu_input_dim_idx = _find_input_idx_with_key(
                k, model_state, self_model_state, stu_bn_idx_dict, stu_bn_idx_list,
                param_name_to_class_type, key='map_to_densehead'
            )
            stu_bn_idx_list.pop()
            curr_v = curr_v.index_select(1, _stu_input_dim_idx)
        stu_model_state[k] = _narrow_weight(k, curr_v, self_model_state, spconv_keys)
    assert len(model_state) == len(stu_model_state)
    return stu_model_state


def _remap_param_in_queue(param_queue, model_state, new_model_state, self_model_state,
                          stu_idx_dict, stu_idx_list, param_name_to_class_type):
    stu_input_dim_idx = stu_idx_dict[stu_idx_list[-2]]
    stu_output_dim_idx = stu_idx_dict[stu_idx_list[-1]]
    for p in param_queue:
        if model_state[p].shape == self_model_state[p].shape:
            new_model_state[p] = model_state[p]
        elif stu_input_dim_idx is None:  # input
            if len(model_state[p].shape) == 5 or len(model_state[p].shape) == 1:
                new_model_state[p] = model_state[p][..., stu_output_dim_idx]
            else:
                new_model_state[p] = model_state[p]
        elif stu_output_dim_idx is None:  # output, 2d conv
            if self_model_state[p].shape[1] > len(stu_input_dim_idx) and 'dense_head' in p:
                # map_to_densehead (SECOND)
                _stu_input_dim_idx = _find_input_idx_with_key(
                    p, model_state, self_model_state, stu_idx_dict, stu_idx_list,
                    param_name_to_class_type, key='map_to_densehead'
                )
                new_model_state[p] = model_state[p][:, _stu_input_dim_idx]
            else:
                new_model_state[p] = model_state[p][:, stu_input_dim_idx]
        else:
            if len(model_state[p].shape) == 5:  # 3d.conv.weight [k,k,k,i,o]
                new_model_state[p] = model_state[p][..., stu_input_dim_idx, :][..., stu_output_dim_idx]
            elif len(model_state[p].shape) == 1:  # bias
                new_model_state[p] = model_state[p][stu_output_dim_idx]
            elif len(model_state[p].shape) == 4:  # 2d.conv.weight [o,i,k,k]
                if self_model_state[p].shape[1] > len(stu_input_dim_idx) and 'backbone_2d' in p:
                    # map_to_bev
                    _stu_input_dim_idx = _find_input_idx_with_key(
                        p, model_state, self_model_state, stu_idx_dict, stu_idx_list,
                        param_name_to_class_type, key='map_to_bev'
                    )
                    new_model_state[p] = model_state[p][stu_output_dim_idx][:,  _stu_input_dim_idx]
                elif self_model_state[p].shape[1] > len(stu_input_dim_idx) and 'dense_head' in p:
                    # map_to_densehead
                    _stu_input_dim_idx = _find_input_idx_with_key(
                        p, model_state, self_model_state, stu_idx_dict, stu_idx_list,
                        param_name_to_class_type, key='map_to_densehead'
                    )
                    new_model_state[p] = model_state[p][stu_output_dim_idx][:, _stu_input_dim_idx]
                elif p.find('deblocks') > 0 and \
                    issubclass(param_name_to_class_type[p], _ConvTransposeNd):
                    # deconv block, convTranspose
                    # the input of deconv block is the corresponding conv block
                    _stu_input_dim_idx = _find_input_idx_with_key(
                        p, model_state, self_model_state, stu_idx_dict, stu_idx_list,
                        param_name_to_class_type, key='deblocks'
                    )
                    new_model_state[p] = model_state[p][_stu_input_dim_idx][:, stu_output_dim_idx]
                elif p.find('heads_list') > 0 and p.split('.')[4] == '0':
                    # dense_head.heads_list.x.x.0
                    # the input of this block is the output of the shared_conv
                    _stu_input_dim_idx = _find_input_idx_with_key(
                        p, model_state, self_model_state, stu_idx_dict, stu_idx_list,
                        param_name_to_class_type, key='heads_list'
                    )
                    new_model_state[p] = model_state[p][stu_output_dim_idx][:, _stu_input_dim_idx]
                else:
                    new_model_state[p] = model_state[p][stu_output_dim_idx][:, stu_input_dim_idx]
            else:
                new_model_state[p] = model_state[p]
        assert new_model_state[p].shape == self_model_state[p].shape


def _narrow_weight(k, curr_v, self_model_state, spconv_keys):
    # adapt spconv weights from version 1.x to version 2.x if you used weights from spconv 1.x
    if k in spconv_keys and self_model_state[k].shape != curr_v.shape:
        # (k1, k2, k3, c_in, c_out) to (c_out, k1, k2, k3, c_in)
        curr_v = curr_v.permute(4, 0, 1, 2, 3)

    for d in range(len(self_model_state[k].shape)):
        curr_v = curr_v.narrow(d, 0, self_model_state[k].shape[d])

    return curr_v


def _select_conv_idx(weight, topk, dim=None, input_dim=None, input_idx=None, descending=True):
    _weight = weight.clone()
    if input_idx is not None:
        _weight = _weight.index_select(input_dim, input_idx)
    importance = torch.sum(torch.abs(_weight), dim=dim)
    sorted_idx = importance.argsort(dim=0, descending=descending)
    return sorted_idx[:topk]


def _find_conv_input_dim(p, param_name_to_class_type):
    if issubclass(param_name_to_class_type[p], spconv.conv.SparseConvolution):
        return 4
    elif issubclass(param_name_to_class_type[p], _ConvTransposeNd):
        return 0
    elif issubclass(param_name_to_class_type[p], nn.modules.conv._ConvNd):
        return 1
    else:
        raise NotImplementedError


def _find_input_idx_with_key(p, model_state, new_model_state, stu_idx_dict,
                             stu_idx_list, param_name_to_class_type, key=None):
    if key and key == 'deblocks':
        corr_conv_name = '.'.join(p.replace('deblocks', 'blocks').split('.')[:3])
        corr_conv_last_bn = reverse_search(stu_idx_list, corr_conv_name)
        return stu_idx_dict[corr_conv_last_bn]
    elif key and key == 'heads_list':
        corr_conv_last_bn = reverse_search(stu_idx_list, 'shared_conv')
        return stu_idx_dict[corr_conv_last_bn]
    elif key and key == 'map_to_densehead':
        name_list = search_concat_name_for_densehead(stu_idx_list)
        stu_idx_list = [stu_idx_dict[n].clone() for n in name_list]
        tea_idx_len_list = [0]
        for n in name_list:
            if issubclass(param_name_to_class_type[n], _ConvTransposeNd):
                tea_idx_len_list.append(model_state[n].shape[1])
            else:
                tea_idx_len_list.append(model_state[n].shape[0])
        return get_concat_idx(stu_idx_list, tea_idx_len_list)
    elif key and key == 'map_to_bev':
        stu_input_dim_idx = stu_idx_dict[stu_idx_list[-2]]
        multipler = new_model_state[p].shape[1] // len(stu_input_dim_idx)
        # e.g.[[1,2],[1,2]]
        idx_list = [stu_idx_dict[stu_idx_list[-2]].clone() for _ in range(multipler)]
        # e.g.[[1,2],[1+4,2+4]]
        return get_concat_idx(idx_list, model_state[p].shape[1] // multipler)
    else:
        return stu_idx_dict[stu_idx_list[-1]]


def _find_input_idx(p, model_state, new_model_state, stu_idx_dict,
                    stu_idx_list, param_name_to_class_type):
    stu_input_dim_idx = stu_idx_dict[stu_idx_list[-1]]
    if stu_input_dim_idx is None:
        return None
    if len(model_state[p].shape) == 4:  # 2d.conv.weight [o,i,k,k]
        if new_model_state[p].shape[1] > len(stu_input_dim_idx) and 'backbone_2d' in p:
            # map_to_bev
            return _find_input_idx_with_key(
                p, model_state, new_model_state, stu_idx_dict, stu_idx_list,
                param_name_to_class_type, key='map_to_bev'
            )
        elif new_model_state[p].shape[1] > len(stu_input_dim_idx) and 'dense_head' in p:
            # map_to_densehead
            return _find_input_idx_with_key(
                p, model_state, new_model_state, stu_idx_dict, stu_idx_list,
                param_name_to_class_type, key='map_to_densehead'
            )
        elif p.find('deblocks') > 0 and \
            issubclass(param_name_to_class_type[p], _ConvTransposeNd):
            # deconv block, convTranspose
            # the input of deconv block is the corresponding conv block
            return _find_input_idx_with_key(
                p, model_state, new_model_state, stu_idx_dict, stu_idx_list,
                param_name_to_class_type, key='deblocks'
            )
        elif p.find('heads_list') > 0 and p.split('.')[4] == '0':
            # dense_head.heads_list.x.x.0
            # the input of this block is the output of the shared_conv
            return _find_input_idx_with_key(
                p, model_state, new_model_state, stu_idx_dict, stu_idx_list,
                param_name_to_class_type, key='heads_list'
            )
        else:
            return stu_input_dim_idx


def reverse_search(stu_idx_list, corr_conv_name):
    for key in stu_idx_list[::-1]:
        if key and corr_conv_name in key:
            return key


def get_concat_idx(idx_list, len_list):
    if not isinstance(len_list, list):
        len_list = [len_list * i for i in range(len(idx_list))]
    else:
        len_list = torch.tensor(len_list).cumsum(0)
    return torch.cat([idx + len_list[i] for (i, idx) in enumerate(idx_list)])


def search_concat_name_for_densehead(stu_idx_list):
    name_list = []
    curr_name = None
    for key in stu_idx_list[::-1]:
        if key and 'deblocks' in key and (curr_name is None or curr_name not in key):
            curr_name = '.'.join(key.split('.')[:3])  # e.g. backbone_2d.deblocks.0
            name_list.append(key)
    return name_list[::-1]


def debug_print_param(param, model_state, new_model_state):
    for p in param:
       print(p, model_state[p].shape, new_model_state[p].shape)
