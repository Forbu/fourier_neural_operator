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
from fourier_neural_operator.utilities import *
from fourier_neural_operator.lowrank_2d import *

torch.manual_seed(0)
np.random.seed(0)

if __name__ == "__main__":
    TRAIN_PATH = 'data/piececonst_r421_N1024_smooth1.mat'
    TEST_PATH = 'data/piececonst_r421_N1024_smooth2.mat'

    ntrain = 1000
    ntest = 100

    batch_size = 10

    r = 5
    h = int(((421 - 1)/r) + 1)
    s = h

    learning_rate = 0.00025

    reader = MatReader(TRAIN_PATH)
    x_train = reader.read_field('coeff')[:ntrain,::r,::r][:,:s,:s].reshape(ntrain,s*s)
    y_train = reader.read_field('sol')[:ntrain,::r,::r][:,:s,:s].reshape(ntrain,s*s)

    reader.load_file(TEST_PATH)
    x_test = reader.read_field('coeff')[:ntest,::r,::r][:,:s,:s].reshape(ntest,s*s)
    y_test = reader.read_field('sol')[:ntest,::r,::r][:,:s,:s].reshape(ntest,s*s)


    x_normalizer = UnitGaussianNormalizer(x_train)
    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)

    y_normalizer = UnitGaussianNormalizer(y_train)
    y_train = y_normalizer.encode(y_train)

    grids = []
    grids.append(np.linspace(0, 1, s))
    grids.append(np.linspace(0, 1, s))
    grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T
    grid = grid.reshape(1,s*s,2)
    grid = torch.tensor(grid, dtype=torch.float)
    x_train = torch.cat([x_train.reshape(ntrain,s*s,1), grid.repeat(ntrain,1,1)], dim=2)
    x_test = torch.cat([x_test.reshape(ntest,s*s,1), grid.repeat(ntest,1,1)], dim=2)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

    model = MyNet(s).cuda()
    # model = MyNet_old(s).cuda()

    print(model.count_params())

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    epochs = 200

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
            out = model(x).reshape(batch_size, s*s)

            mse = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1), reduction='mean')
            mse.backward()

            out = y_normalizer.decode(out)
            y = y_normalizer.decode(y)
            loss = myloss(out.view(batch_size,-1), y.view(batch_size,-1))
            # loss.backward()

            optimizer.step()
            train_mse += mse.item()
            train_l2 += loss.item()

        scheduler.step()

        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()

                out = model(x).reshape(batch_size, s*s)
                out = y_normalizer.decode(out)
                test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

        train_mse /= len(train_loader)
        train_l2 /= ntrain
        test_l2 /= ntest

        t2 = default_timer()
        print(ep, t2-t1, train_mse, train_l2, test_l2)
