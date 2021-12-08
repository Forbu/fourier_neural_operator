"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 1D problem such as the (time-independent) Burgers equation discussed in Section 5.1 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import operator
from functools import reduce
from functools import partial
from timeit import default_timer

from fourier_neural_operator.utilities3 import *
from fourier_neural_operator.Adam import Adam
from fourier_neural_operator.fourier_1d import *

torch.manual_seed(0)
np.random.seed(0)

if __name__ == "__main__":
    ################################################################
    #  configurations
    ################################################################
    
    import matplotlib.pyplot as plt
    ntrain = 1000
    ntest = 100

    sub = 2**3 #subsampling rate
    h = 2**13 // sub #total grid size divided by the subsampling rate
    s = h

    batch_size = 20
    learning_rate = 0.001

    epochs = 500
    step_size = 50
    gamma = 0.5

    modes = 16
    width = 64


    ################################################################
    # read data
    ################################################################

    # Data is of the shape (number of samples, grid size)
    dataloader = MatReader('data/burgers_data_R10.mat')
    x_data = dataloader.read_field('a')[:,::sub]
    y_data = dataloader.read_field('u')[:,::sub]

    x_train = x_data[:ntrain,:]
    y_train = y_data[:ntrain,:]
    x_test = x_data[-ntest:,:]
    y_test = y_data[-ntest:,:]

    x_train = x_train.reshape(ntrain,s,1)
    x_test = x_test.reshape(ntest,s,1)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

    # model
    model = FNO1d(modes, width).cuda()
    print(count_params(model))

    ################################################################
    # training and evaluation
    ################################################################
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    myloss = LpLoss(size_average=False)
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_mse = 0
        train_l2 = 0
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()

            optimizer.zero_grad()
            out = model(x)

            mse = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1), reduction='mean')
            l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
            l2.backward() # use the l2 relative loss

            optimizer.step()
            train_mse += mse.item()
            train_l2 += l2.item()

        scheduler.step()
        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()

                out = model(x)
                test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

        train_mse /= len(train_loader)
        train_l2 /= ntrain
        test_l2 /= ntest

        t2 = default_timer()
        print(ep, t2-t1, train_mse, train_l2, test_l2)

    # torch.save(model, 'model/ns_fourier_burgers')
    pred = torch.zeros(y_test.shape)
    index = 0
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False)
    with torch.no_grad():
        for x, y in test_loader:
            test_l2 = 0
            x, y = x.cuda(), y.cuda()

            out = model(x).view(-1)
            pred[index] = out

            test_l2 += myloss(out.view(1, -1), y.view(1, -1)).item()
            print(index, test_l2)
            index = index + 1

    # scipy.io.savemat('pred/burger_test.mat', mdict={'pred': pred.cpu().numpy()})
