import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import operator
from functools import reduce
from functools import partial

from timeit import default_timer



################################################################
# 3d lowrank layers
################################################################

class LowRank2d(nn.Module):
    def __init__(self, in_channels, out_channels, n, ker_width, rank):
        super(LowRank2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n = n
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



class MyNet(torch.nn.Module):
    def __init__(self, n, width=16, ker_width=256, rank=16):
        super(MyNet, self).__init__()
        self.n = n
        self.width = width
        self.ker_width = ker_width
        self.rank = rank

        self.fc0 = nn.Linear(13, self.width)

        self.conv0 = LowRank2d(width, width, n, ker_width, rank)
        self.conv1 = LowRank2d(width, width, n, ker_width, rank)
        self.conv2 = LowRank2d(width, width, n, ker_width, rank)
        self.conv3 = LowRank2d(width, width, n, ker_width, rank)

        self.w0 = nn.Linear(self.width, self.width)
        self.w1 = nn.Linear(self.width, self.width)
        self.w2 = nn.Linear(self.width, self.width)
        self.w3 = nn.Linear(self.width, self.width)
        self.bn0 = torch.nn.BatchNorm1d(self.width)
        self.bn1 = torch.nn.BatchNorm1d(self.width)
        self.bn2 = torch.nn.BatchNorm1d(self.width)
        self.bn3 = torch.nn.BatchNorm1d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)


    def forward(self, x):
        batch_size = x.shape[0]
        size_x, size_y, size_z = x.shape[1], x.shape[2], x.shape[3]
        x = x.view(batch_size, size_x*size_y*size_z, -1)

        x = self.fc0(x)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = self.bn0(x.reshape(-1, self.width)).view(batch_size, size_x*size_y*size_z, self.width)
        x = F.relu(x)
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = self.bn1(x.reshape(-1, self.width)).view(batch_size, size_x*size_y*size_z, self.width)
        x = F.relu(x)
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = self.bn2(x.reshape(-1, self.width)).view(batch_size, size_x*size_y*size_z, self.width)
        x = F.relu(x)
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        x = self.bn3(x.reshape(-1, self.width)).view(batch_size, size_x*size_y*size_z, self.width)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = x.view(batch_size, size_x, size_y, size_z)
        return x

class Net2d(nn.Module):
    def __init__(self, width=8, ker_width=128, rank=4):
        super(Net2d, self).__init__()

        self.conv1 = MyNet(n=64*64*40, width=width, ker_width=ker_width, rank=rank)


    def forward(self, x):
        x = self.conv1(x)
        return x


    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c
