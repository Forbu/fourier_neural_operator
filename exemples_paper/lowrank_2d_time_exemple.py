import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from fourier_neural_operator.utilities3 import *

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
import scipy.io

torch.manual_seed(0)
np.random.seed(0)

from fourier_neural_operator.lowrank_2d_time import *

activation = F.relu

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

    batch_size = 5
    batch_size2 = batch_size

    epochs = 500
    learning_rate = 0.0025
    scheduler_step = 100
    scheduler_gamma = 0.5

    print(epochs, learning_rate, scheduler_step, scheduler_gamma)

    path = 'ns_lowrank_rnn_V100_T40_N'+str(ntrain)+'_ep' + str(epochs) + '_m'
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
    step = 1


    ################################################################
    # load dataset
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


    train_a = train_a.reshape(ntrain,S,S,T_in)
    test_a = test_a.reshape(ntest,S,S,T_in)

    gridx = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
    gridx = gridx.reshape(1, S, 1, 1).repeat([1, 1, S, 1])
    gridy = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
    gridy = gridy.reshape(1, 1, S, 1).repeat([1, S, 1, 1])

    train_a = torch.cat((train_a, gridx.repeat([ntrain,1,1,1]), gridy.repeat([ntrain,1,1,1])), dim=-1)
    test_a = torch.cat((test_a, gridx.repeat([ntest,1,1,1]), gridy.repeat([ntest,1,1,1])), dim=-1)

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
    gridx = gridx.to(device)
    gridy = gridy.to(device)

    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_l2_step = 0
        train_l2_full = 0
        for xx, yy in train_loader:
            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)

            for t in range(0, T, step):
                y = yy[..., t:t + step]
                im = model(xx)
                loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)

                xx = torch.cat((xx[..., step:-2], im,
                                gridx.repeat([batch_size, 1, 1, 1]), gridy.repeat([batch_size, 1, 1, 1])), dim=-1)

            train_l2_step += loss.item()
            l2_full = myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1))
            train_l2_full += l2_full.item()

            optimizer.zero_grad()
            loss.backward()
            # l2_full.backward()
            optimizer.step()

        test_l2_step = 0
        test_l2_full = 0
        with torch.no_grad():
            for xx, yy in test_loader:
                loss = 0
                xx = xx.to(device)
                yy = yy.to(device)

                for t in range(0, T, step):
                    y = yy[..., t:t + step]
                    im = model(xx)
                    loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

                    if t == 0:
                        pred = im
                    else:
                        pred = torch.cat((pred, im), -1)

                    xx = torch.cat((xx[..., step:-2], im,
                                    gridx.repeat([batch_size, 1, 1, 1]), gridy.repeat([batch_size, 1, 1, 1])), dim=-1)


                test_l2_step += loss.item()
                test_l2_full += myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()

        t2 = default_timer()
        scheduler.step()
        print(ep, t2 - t1, train_l2_step / ntrain / (T / step), train_l2_full / ntrain, test_l2_step / ntest / (T / step),
              test_l2_full / ntest)
    # torch.save(model, path_model)


    # pred = torch.zeros(test_u.shape)
    # index = 0
    # test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=1, shuffle=False)
    # with torch.no_grad():
    #     for x, y in test_loader:
    #         test_l2 = 0;
    #         x, y = x.cuda(), y.cuda()
    #
    #         out = model(x)
    #         out = y_normalizer.decode(out)
    #         pred[index] = out
    #
    #         test_l2 += myloss(out.view(1, -1), y.view(1, -1)).item()
    #         print(index, test_l2)
    #         index = index + 1

    # scipy.io.savemat('pred/'+path+'.mat', mdict={'pred': pred.cpu().numpy()})

