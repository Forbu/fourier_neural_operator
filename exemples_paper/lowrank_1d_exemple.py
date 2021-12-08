import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
import h5py
import scipy.io
import matplotlib.pyplot as plt
from timeit import default_timer
import sys
import math

import operator
from functools import reduce

from timeit import default_timer
from fourier_neural_operator.utilities3 import *
from fourier_neural_operator.lowrank_1d import *

torch.manual_seed(0)
np.random.seed(0)

if __name__ == "__main__":
    ################################################################
    # configs
    ################################################################

    ntrain = 1000
    ntest = 200

    sub = 1 #subsampling rate
    h = 2**13 // sub
    s = h

    batch_size = 5
    learning_rate = 0.001


    ################################################################
    # reading data and normalization
    ################################################################
    dataloader = MatReader('data/burgers_data_R10.mat')
    x_data = dataloader.read_field('a')[:,::sub]
    y_data = dataloader.read_field('u')[:,::sub]

    x_train = x_data[:ntrain,:]
    y_train = y_data[:ntrain,:]
    x_test = x_data[-ntest:,:]
    y_test = y_data[-ntest:,:]

    x_normalizer = UnitGaussianNormalizer(x_train)
    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)

    y_normalizer = UnitGaussianNormalizer(y_train)
    y_train = y_normalizer.encode(y_train)

    grid = np.linspace(0, 2*np.pi, s).reshape(1, s, 1)
    grid = torch.tensor(grid, dtype=torch.float)
    x_train = torch.cat([x_train.reshape(ntrain,s,1), grid.repeat(ntrain,1,1)], dim=2)
    x_test = torch.cat([x_test.reshape(ntest,s,1), grid.repeat(ntest,1,1)], dim=2)


    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

    model = MyNet(s).cuda()
    print(model.count_params())

    ################################################################
    # training and evaluation
    ################################################################

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    epochs = 500

    myloss = LpLoss(size_average=False)
    y_normalizer.cuda()
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_mse = 0
        train_l2 = 0
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()

            optimizer.zero_grad()
            out = model(x).reshape(batch_size, s)

            mse = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1), reduction='mean')
            # mse.backward()

            out = y_normalizer.decode(out)
            y = y_normalizer.decode(y)
            loss = myloss(out.view(batch_size,-1), y.view(batch_size,-1))
            loss.backward()

            optimizer.step()
            train_mse += mse.item()
            train_l2 += loss.item()

        scheduler.step()

        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()

                out = model(x).reshape(batch_size, s)
                out = y_normalizer.decode(out)
                test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

        train_mse /= len(train_loader)
        train_l2 /= ntrain
        test_l2 /= ntest

        t2 = default_timer()
        print(ep, t2-t1, train_mse, train_l2, test_l2)
