from collections import namedtuple

import numpy as np
import torch

from .detectors import build_detector
from pcdet.utils import commu_utils

try:
    import kornia
except:
    pass 
    # print('Warning: kornia is not installed. This package is only required by CaDDN')


def build_teacher_network(cfg, args, train_set, dist, logger):
    teacher_model = build_network(model_cfg=cfg.MODEL_TEACHER, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    teacher_model.cuda()

    for param_k in teacher_model.parameters():
        param_k.requires_grad = False  # not update by gradient

    teacher_model.train()

    if args.teacher_ckpt is not None:
        logger.info('Loading teacher parameters >>>>>>')
        teacher_model.load_params_from_file(filename=args.teacher_ckpt, to_cpu=dist, logger=logger)

    teacher_model.is_teacher = True
    for cur_module in teacher_model.module_list:
        cur_module.is_teacher = True
        cur_module.kd = True

    return teacher_model


def build_network(model_cfg, num_class, dataset):
    model = build_detector(
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )
    return model


def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        elif key in ['frame_id', 'metadata', 'calib']:
            continue
        elif key in ['images']:
            batch_dict[key] = kornia.image_to_tensor(val).float().cuda().contiguous()
        elif key in ['image_shape']:
            batch_dict[key] = torch.from_numpy(val).int().cuda()
        else:
            batch_dict[key] = torch.from_numpy(val).float().cuda()


def model_fn_decorator():
    ModelReturn = namedtuple('ModelReturn', ['loss', 'tb_dict', 'disp_dict'])

    def model_func(model, batch_dict):
        load_data_to_gpu(batch_dict)
        ret_dict, tb_dict, disp_dict = model(batch_dict)

        loss = ret_dict['loss'].mean()
        if hasattr(model, 'update_global_step'):
            model.update_global_step()
        else:
            model.module.update_global_step()

        return ModelReturn(loss, tb_dict, disp_dict)

    return model_func
