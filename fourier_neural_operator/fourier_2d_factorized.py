"""
@author: Zongyi Li This file is the Fourier Neural Operator for 2D problem such
as the Navier-Stokes equation discussed in Section 5.3 in the
[paper](https://arxiv.org/pdf/2010.08895.pdf), which uses a recurrent structure
to propagates in time.

this part of code is taken from :
https://github.com/alasdairtran/fourierflow/tree/97e6cfb0848e44d3a7bc1d063b1ab86bc4c603ee

"""


from functools import partial

import torch
import torch.nn as nn
from einops import rearrange

import fourier_neural_operator.layers.fourier_2d_factorized as fourier_2d_factorized
import fourier_neural_operator.layers.linear as linear 

class ffno(nn.Module):
    def __init__(self, modes1, width, input_dim=12, dropout=0.1, n_layers=4, output_dim=1, residual=False, conv_residual=True):
        super(ffno, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes1 = modes1
        self.width = width
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.in_proj = nn.Linear(input_dim, self.width)
        self.residual = residual
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        self.spectral_layers = nn.ModuleList([])
        for i in range(n_layers):
            self.spectral_layers.append(fourier_2d_factorized.SpectralConv2d(in_dim=width,
                                                       out_dim=width,
                                                       n_modes=modes1,
                                                       resdiual=conv_residual,
                                                       dropout=dropout))

        self.feedforward = nn.Sequential(
            nn.Linear(self.width, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.output_dim))

    def forward(self, x, **kwargs):
        # x.shape == [n_batches, *dim_sizes, input_size]
        x = self.in_proj(x)
        for layer in self.spectral_layers:
            x = layer(x) + x if self.residual else layer(x)

        x = self.feedforward(x)
        # x.shape == [n_batches, *dim_sizes, 1]

        return x
