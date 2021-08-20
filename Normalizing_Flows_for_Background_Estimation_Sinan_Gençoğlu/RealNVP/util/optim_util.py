import numpy as np
import torch.nn.utils as utils


def bits_per_dim(x, nll):
    dim = np.prod(x.size()[1:])
    bpd = nll / (np.log(2) * dim)
    return bpd

def clip_grad_norm(optimizer, max_norm, norm_type=2):
    for group in optimizer.param_groups:
        utils.clip_grad_norm_(group['params'], max_norm, norm_type)
