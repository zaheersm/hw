import os

import numpy as np
import torch


def select_device(gpu_id):
    # if torch.cuda.is_available() and gpu_id >= 0:
    if gpu_id >= 0:
        return torch.device('cuda:%d' % (gpu_id))
    else:
        return torch.device('cpu')


def tensor(x, device):
    if isinstance(x, torch.Tensor):
        return x
    x = torch.tensor(x, dtype=torch.float32).to(device)
    return x


def range_tensor(end, device):
    return torch.arange(end).long().to(device)


def to_np(t):
    return t.cpu().detach().numpy()


def random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_one_thread():
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
