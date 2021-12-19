import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from fourier_neural_operator.utilities3 import *

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
import scipy.io

activation = F.relu

################################################################
# lowrank layers
################################################################
class LowRank2d(nn.Module):
    def __init__(self, in_channels, out_channels, s, ker_width, rank):
        super(LowRank2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.s = s
        self.n = s*s
        self.rank = rank

        self.phi = DenseNet([in_channels, ker_width, in_channels*out_channels*rank], torch.nn.ReLU)
        self.psi = DenseNet([in_channels, ker_width, in_channels*out_channels*rank], torch.nn.ReLU)


    def forward(self, v):
        batch_size = v.shape[0]

        phi_eval = self.phi(v).reshape(batch_size, self.n, self.out_channels, self.in_channels, self.rank)
        psi_eval = self.psi(v).reshape(batch_size, self.n, self.out_channels, self.in_channels, self.rank)

        # print(psi_eval.shape, v.shape, phi_eval.shape)
        v = torch.einsum('bnoir,bni,bmoir->bmo',psi_eval, v, phi_eval)

        return v

