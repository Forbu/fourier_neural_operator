import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
import scipy.io

from fourier_neural_operator.utilities3 import *
from fourier_neural_operator.lowrank_3d import *

torch.manual_seed(0)
np.random.seed(0)

if __name__ == "__main__":
    
    ################################################################
    # configs
    ################################################################
    # TRAIN_PATH = 'data/ns_data_V10000_N1200_T20.mat'
    # TEST_PATH = 'data/ns_data_V10000_N1200_T20.mat'
    # TRAIN_PATH = 'data/ns_data_V1000_N1000_train.mat'
    # TEST_PATH = 'data/ns_data_V1000_N1000_train_2.mat'
    # TRAIN_PATH = 'data/ns_data_V1000_N5000.mat'
    # TEST_PATH = 'data/ns_data_V1000_N5000.mat'
    TRAIN_PATH = 'data/ns_data_V100_N1000_T50_1.mat'
    TEST_PATH = 'data/ns_data_V100_N1000_T50_2.mat'

    ntrain = 1000
    ntest = 200

    batch_size = 2
    batch_size2 = batch_size

    epochs = 500
    learning_rate = 0.0025
    scheduler_step = 100
    scheduler_gamma = 0.5

    print(epochs, learning_rate, scheduler_step, scheduler_gamma)

    path = 'ns_lowrank_V100_T40_N'+str(ntrain)+'_ep' + str(epochs)
    path_model = 'model/'+path
    path_train_err = 'results/'+path+'train.txt'
    path_test_err = 'results/'+path+'test.txt'
    path_image = 'image/'+path

    runtime = np.zeros(2, )
    t1 = default_timer()


    sub = 1
    S = 64
    T_in = 10
    T = 40

    ################################################################
    # load data
    ################################################################

    reader = MatReader(TRAIN_PATH)
    train_a = reader.read_field('u')[:ntrain,::sub,::sub,:T_in]
    train_u = reader.read_field('u')[:ntrain,::sub,::sub,T_in:T+T_in]

    reader = MatReader(TEST_PATH)
    test_a = reader.read_field('u')[-ntest:,::sub,::sub,:T_in]
    test_u = reader.read_field('u')[-ntest:,::sub,::sub,T_in:T+T_in]

    print(train_u.shape)
    print(test_u.shape)
    assert (S == train_u.shape[-2])
    assert (T == train_u.shape[-1])


    a_normalizer = UnitGaussianNormalizer(train_a)
    train_a = a_normalizer.encode(train_a)
    test_a = a_normalizer.encode(test_a)

    y_normalizer = UnitGaussianNormalizer(train_u)
    train_u = y_normalizer.encode(train_u)

    train_a = train_a.reshape(ntrain,S,S,1,T_in).repeat([1,1,1,T,1])
    test_a = test_a.reshape(ntest,S,S,1,T_in).repeat([1,1,1,T,1])

    gridx = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
    gridx = gridx.reshape(1, S, 1, 1, 1).repeat([1, 1, S, T, 1])
    gridy = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
    gridy = gridy.reshape(1, 1, S, 1, 1).repeat([1, S, 1, T, 1])
    gridt = torch.tensor(np.linspace(0, 1, T+1)[1:], dtype=torch.float)
    gridt = gridt.reshape(1, 1, 1, T, 1).repeat([1, S, S, 1, 1])

    train_a = torch.cat((gridx.repeat([ntrain,1,1,1,1]), gridy.repeat([ntrain,1,1,1,1]),
                           gridt.repeat([ntrain,1,1,1,1]), train_a), dim=-1)
    test_a = torch.cat((gridx.repeat([ntest,1,1,1,1]), gridy.repeat([ntest,1,1,1,1]),
                           gridt.repeat([ntest,1,1,1,1]), test_a), dim=-1)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)

    t2 = default_timer()

    print('preprocessing finished, time used:', t2-t1)
    device = torch.device('cuda')


    ################################################################
    # training and evaluation
    ################################################################
    model = Net2d().cuda()
    # model = torch.load('model/ns_fourier_V100_N1000_ep100_m8_w20')

    print(model.count_params())
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)


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
            out = model(x)

            mse = F.mse_loss(out, y, reduction='mean')
            # mse.backward()

            y = y_normalizer.decode(y)
            out = y_normalizer.decode(out)
            l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
            l2.backward()

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
                out = y_normalizer.decode(out)
                test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

        train_mse /= len(train_loader)
        train_l2 /= ntrain
        test_l2 /= ntest

        t2 = default_timer()
        print(ep, t2-t1, train_mse, train_l2, test_l2)
    # torch.save(model, path_model)


    pred = torch.zeros(test_u.shape)
    index = 0
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=1, shuffle=False)
    with torch.no_grad():
        for x, y in test_loader:
            test_l2 = 0;
            x, y = x.cuda(), y.cuda()

            out = model(x)
            out = y_normalizer.decode(out)
            pred[index] = out

            test_l2 += myloss(out.view(1, -1), y.view(1, -1)).item()
            print(index, test_l2)
            index = index + 1

    # scipy.io.savemat('pred/'+path+'.mat', mdict={'pred': pred.cpu().numpy()})




