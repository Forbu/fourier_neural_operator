"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 2D problem such as the Darcy Flow discussed in Section 5.2 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import matplotlib.pyplot as plt

import operator
from functools import reduce
from functools import partial

from timeit import default_timer

from fourier_neural_operator.utilities3 import *
from fourier_neural_operator.Adam import Adam
from fourier_neural_operator.fourier_3d import *

torch.manual_seed(0)
np.random.seed(0)

if __name__ == "__main__":
    
    ################################################################
    # configs
    ################################################################
    TRAIN_PATH = 'data/piececonst_r421_N1024_smooth1.mat'
    TEST_PATH = 'data/piececonst_r421_N1024_smooth2.mat'

    ntrain = 1000
    ntest = 100

    batch_size = 20
    learning_rate = 0.001

    epochs = 500
    step_size = 100
    gamma = 0.5

    modes = 12
    width = 32

    r = 5
    h = int(((421 - 1)/r) + 1)
    s = h

    ################################################################
    # load data and data normalization
    ################################################################
    reader = MatReader(TRAIN_PATH)
    x_train = reader.read_field('coeff')[:ntrain,::r,::r][:,:s,:s]
    y_train = reader.read_field('sol')[:ntrain,::r,::r][:,:s,:s]

    reader.load_file(TEST_PATH)
    x_test = reader.read_field('coeff')[:ntest,::r,::r][:,:s,:s]
    y_test = reader.read_field('sol')[:ntest,::r,::r][:,:s,:s]

    x_normalizer = UnitGaussianNormalizer(x_train)
    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)

    y_normalizer = UnitGaussianNormalizer(y_train)
    y_train = y_normalizer.encode(y_train)

    x_train = x_train.reshape(ntrain,s,s,1)
    x_test = x_test.reshape(ntest,s,s,1)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

    ################################################################
    # training and evaluation
    ################################################################
    model = FNO2d(modes, modes, width).cuda()
    print(count_params(model))

    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    myloss = LpLoss(size_average=False)
    y_normalizer.cuda()
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_l2 = 0
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()

            optimizer.zero_grad()
            out = model(x).reshape(batch_size, s, s)
            out = y_normalizer.decode(out)
            y = y_normalizer.decode(y)

            loss = myloss(out.view(batch_size,-1), y.view(batch_size,-1))
            loss.backward()

            optimizer.step()
            train_l2 += loss.item()

        scheduler.step()

        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()

                out = model(x).reshape(batch_size, s, s)
                out = y_normalizer.decode(out)

                test_l2 += myloss(out.view(batch_size,-1), y.view(batch_size,-1)).item()

        train_l2/= ntrain
        test_l2 /= ntest

        t2 = default_timer()
        print(ep, t2-t1, train_l2, test_l2)
