import numpy as np
import torch.nn as nn
import torch

class RealNVPLoss(nn.Module):
    def __init__(self, k=256):
        super(RealNVPLoss, self).__init__()
        self.k = k

    def forward(self, z, sldj):
        prior_ll = -0.5 * (z ** 2 + np.log(2 * np.pi))
        prior_ll = torch.reshape(prior_ll, (z.size(0), -1)).sum(-1)
        prior_ll = prior_ll - np.log(self.k) * np.prod(z.size()[1:])
        ll = prior_ll + sldj
        nll = -ll.mean()
        return nll
