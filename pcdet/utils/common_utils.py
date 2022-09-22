import logging
import os
import pickle
import random
import shutil
import subprocess
import SharedArray

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from .spconv_utils import spconv

try:
    from thop import profile, clever_format, profile_acts
except:
    pass
    # you cannot use cal_param without profile


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def limit_period(val, offset=0.5, period=np.pi):
    val, is_numpy = check_numpy_to_torch(val)
    ans = val - torch.floor(val / period + offset) * period
    return ans.numpy() if is_numpy else ans


def drop_info_with_name(info, name):
    ret_info = {}
    keep_indices = [i for i, x in enumerate(info['name']) if x != name]
    for key in info.keys():
        ret_info[key] = info[key][keep_indices]
    return ret_info


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


def mask_points_by_range(points, limit_range):
    mask = (points[:, 0] >= limit_range[0]) & (points[:, 0] <= limit_range[3]) \
           & (points[:, 1] >= limit_range[1]) & (points[:, 1] <= limit_range[4])
    return mask


def get_voxel_centers(voxel_coords, downsample_times, voxel_size, point_cloud_range):
    """
    Args:
        voxel_coords: (N, 3)
        downsample_times:
        voxel_size:
        point_cloud_range:

    Returns:

    """
    assert voxel_coords.shape[1] == 3
    voxel_centers = voxel_coords[:, [2, 1, 0]].float()  # (xyz)
    voxel_size = torch.tensor(voxel_size, device=voxel_centers.device).float() * downsample_times
    pc_range = torch.tensor(point_cloud_range[0:3], device=voxel_centers.device).float()
    voxel_centers = (voxel_centers + 0.5) * voxel_size + pc_range
    return voxel_centers


def create_logger(log_file=None, rank=0, log_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level if rank == 0 else 'ERROR')
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
    console = logging.StreamHandler()
    console.setLevel(log_level if rank == 0 else 'ERROR')
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level if rank == 0 else 'ERROR')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id, seed=666):
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)
    torch.manual_seed(seed + worker_id)
    torch.cuda.manual_seed(seed + worker_id)
    torch.cuda.manual_seed_all(seed + worker_id)


def get_pad_params(desired_size, cur_size):
    """
    Get padding parameters for np.pad function
    Args:
        desired_size: int, Desired padded output size
        cur_size: int, Current size. Should always be less than or equal to cur_size
    Returns:
        pad_params: tuple(int), Number of values padded to the edges (before, after)
    """
    assert desired_size >= cur_size

    # Calculate amount to pad
    diff = desired_size - cur_size
    pad_params = (0, diff)

    return pad_params


def keep_arrays_by_name(gt_names, used_classes):
    inds = [i for i, x in enumerate(gt_names) if x in used_classes]
    inds = np.array(inds, dtype=np.int64)
    return inds


def init_dist_slurm(tcp_port, local_rank, backend='nccl'):
    """
    modified from https://github.com/open-mmlab/mmdetection
    Args:
        tcp_port:
        backend:

    Returns:

    """
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput('scontrol show hostname {} | head -n1'.format(node_list))
    os.environ['MASTER_PORT'] = str(tcp_port)
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend=backend)

    total_gpus = dist.get_world_size()
    rank = dist.get_rank()
    return total_gpus, rank


def init_dist_pytorch(tcp_port, local_rank, backend='nccl'):
    # if mp.get_start_method(allow_none=True) is None:
    #     mp.set_start_method('spawn')

    # os.environ['MASTER_PORT'] = str(tcp_port)
    # os.environ['MASTER_ADDR'] = 'localhost'
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(local_rank % num_gpus)
    dist.init_process_group(
        backend=backend,
        # init_method='tcp://127.0.0.1:%d' % tcp_port,
        # rank=local_rank,
        # world_size=num_gpus
    )
    rank = dist.get_rank()
    return num_gpus, rank


def get_dist_info(return_gpu_per_machine=False):
    if torch.__version__ < '1.0':
        initialized = dist._initialized
    else:
        if dist.is_available():
            initialized = dist.is_initialized()
        else:
            initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    if return_gpu_per_machine:
        gpu_per_machine = torch.cuda.device_count()
        return rank, world_size, gpu_per_machine

    return rank, world_size


def merge_results_dist(result_part, size, tmpdir):
    rank, world_size = get_dist_info()
    os.makedirs(tmpdir, exist_ok=True)

    dist.barrier()
    pickle.dump(result_part, open(os.path.join(tmpdir, 'result_part_{}.pkl'.format(rank)), 'wb'))
    dist.barrier()

    if rank != 0:
        return None

    part_list = []
    for i in range(world_size):
        part_file = os.path.join(tmpdir, 'result_part_{}.pkl'.format(i))
        part_list.append(pickle.load(open(part_file, 'rb')))

    ordered_results = []
    for res in zip(*part_list):
        ordered_results.extend(list(res))
    ordered_results = ordered_results[:size]
    shutil.rmtree(tmpdir)
    return ordered_results


def scatter_point_inds(indices, point_inds, shape):
    ret = -1 * torch.ones(*shape, dtype=point_inds.dtype, device=point_inds.device)
    ndim = indices.shape[-1]
    flattened_indices = indices.view(-1, ndim)
    slices = [flattened_indices[:, i] for i in range(ndim)]
    ret[slices] = point_inds
    return ret


def generate_voxel2pinds(sparse_tensor):
    device = sparse_tensor.indices.device
    batch_size = sparse_tensor.batch_size
    spatial_shape = sparse_tensor.spatial_shape
    indices = sparse_tensor.indices.long()
    point_indices = torch.arange(indices.shape[0], device=device, dtype=torch.int32)
    output_shape = [batch_size] + list(spatial_shape)
    v2pinds_tensor = scatter_point_inds(indices, point_indices, output_shape)
    return v2pinds_tensor


def sa_create(name, var):
    x = SharedArray.create(name, var.shape, dtype=var.dtype)
    x[...] = var[...]
    x.flags.writeable = False
    return x


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        result = f"average value: {self.avg:.3f}"
        return result


class DictAverageMeter(object):
    """
    Contain AverageMeter as dict and update respectively or simultaneously
    """
    def __init__(self):
        self.meters = {}

    def update(self, key, val, n=1):
        if key not in self.meters:
            self.meters[key] = AverageMeter()
        self.meters[key].update(val, n)

    def __repr__(self):
        result = ""
        sum = 0
        for key in self.meters.keys():
            result += f'{key}: {self.meters[key].avg:.2f}\n'
            sum += self.meters[key].avg
        result += f'Total: {sum:.2f}\n'
        return result


def calculate_trainable_params(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable_params


def calculate_trainable_params_for_submodules(model):
    from prettytable import PrettyTable
    table = PrettyTable(["Modules", "Parameters"])
    param_list = []
    for cur_module in model.module_list:
        module_name = str(type(cur_module)).split('.')[-1][:-2]
        n_params = calculate_trainable_params(cur_module)
        table.add_row([module_name, n_params])
        param_list.append(n_params)
    print(table)
    print(f"Total Trainable Params: {sum(param_list)}")
    return param_list


def add_postfix_to_dict(dict, postfix):
    for key in list(dict.keys()):
        dict[key + '_' + postfix] = dict.pop(key)
    return dict


def set_bn_train(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.train()


def cal_flops(model, batch_dict):
    macs, params, acts = profile_acts(model, inputs=(batch_dict,),
                           custom_ops={spconv.SubMConv3d: spconv.SubMConv3d.count_your_model,
                                       spconv.SparseConv3d: spconv.SparseConv3d.count_your_model}
                           )
    return macs, params, acts


def pair_distance_np(a, b):
    inner = 2 * np.dot(a, b.T)
    aa = np.sum(a ** 2, axis=1, keepdims=True)
    bb = np.sum(b ** 2, axis=1, keepdims=True)
    pairwise_distance = aa + bb.T - inner

    return pairwise_distance


def nearest_neighbor(a, b):
    """
    Find the nearest neighbor in b for each element in a. CPU version is too slow.
    Args:
        a: [N, C] numpy array
        b: [M, C] numpy array
    Returns:
        idx: [N] numpy array
    """
    pairwise_distance = pair_distance_np(a, b)

    idx = np.argmin(pairwise_distance, axis=1)
    return idx


def pair_distance_gpu(a, b):
    """
        Find the nearest neighbor in b for each element in a.
        Args:
            a: [N, C] torch cuda tensor
            b: [M, C] torch cuda tensor
        Returns:
            pairwise_distance: [N, M]
        """
    inner = 2 * torch.mm(a, b.transpose(1, 0))
    aa = torch.sum(a ** 2, dim=1, keepdim=True)
    bb = torch.sum(b ** 2, dim=1, keepdim=True)
    pairwise_distance = aa + bb.transpose(1, 0) - inner

    return pairwise_distance


def batch_pair_distance_gpu(a, b):
    """
    Find the nearest neighbor in b for each element in a.
    Args:
        a: [B, N, C] torch cuda tensor
        b: [B, M, C] torch cuda tensor
    Returns:
        pairwise_distance: [B, N, M]
    """
    inner = 2 * torch.matmul(a, b.transpose(2, 1))
    aa = torch.sum(a ** 2, dim=-1, keepdim=True)
    bb = torch.sum(b ** 2, dim=-1, keepdim=True)
    pairwise_distance = aa + bb.transpose(2, 1) - inner

    return pairwise_distance


def nearest_neighbor_gpu(a, b):
    """
    Find the nearest neighbor in b for each element in a.
    Args:
        a: [N, C] torch cuda tensor
        b: [M, C] torch cuda tensor
    Returns:
        idx: [N]
    """
    pairwise_distance = pair_distance_gpu(a, b)

    idx = torch.argmin(pairwise_distance, dim=1)
    return idx
