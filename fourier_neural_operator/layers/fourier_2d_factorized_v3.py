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

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class SpectralConv2d(nn.Module):
    def __init__(self, in_dim, out_dim, n_modes, resdiual=True, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_modes = n_modes
        
        self.dilatation_layer = conv3x3(in_dim, out_dim)
        
        self.residual = resdiual
        self.act = nn.ReLU(inplace=True)

        fourier_weight = [nn.Parameter(torch.FloatTensor(
            in_dim, out_dim, n_modes, n_modes, 2)) for _ in range(2)]
        self.fourier_weight = nn.ParameterList(fourier_weight)
        for param in self.fourier_weight:
            nn.init.xavier_normal_(param, gain=1/(in_dim*out_dim))

    @staticmethod
    def complex_matmul_2d(a, b):
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        op = partial(torch.einsum, "bixy,ioxy->boxy")
        return torch.stack([
            op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
            op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
        ], dim=-1)

    def forward(self, x):
        # x.shape == [batch_size, grid_size, grid_size, in_dim]
        B, M, N, I = x.shape
        O = self.out_dim
        
        # res.shape == [batch_size, grid_size, grid_size, out_dim]

        x = rearrange(x, 'b m n i -> b i m n')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        res = self.dilatation_layer(x)

        x_ft = torch.fft.rfft2(x, s=(M, N), norm='ortho')
        # x_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1]

        x_ft = torch.stack([x_ft.real, x_ft.imag], dim=4)
        # x_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1, 2]

        out_ft = torch.zeros(B, O, N, M // 2 + 1, 2, device=x.device)
        # out_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1, 2]

        out_ft[:, :, :self.n_modes, :self.n_modes] = self.complex_matmul_2d(
            x_ft[:, :, :self.n_modes, :self.n_modes], self.fourier_weight[0])

        out_ft[:, :, -self.n_modes:, :self.n_modes] = self.complex_matmul_2d(
            x_ft[:, :, -self.n_modes:, :self.n_modes], self.fourier_weight[1])

        out_ft = torch.complex(out_ft[..., 0], out_ft[..., 1])

        x = torch.fft.irfft2(out_ft, s=(N, M), norm='ortho')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        if self.residual:
            x = self.act(x + res)
        
        x = rearrange(x, 'b i m n -> b m n i')
        # x.shape == [batch_size, grid_size, grid_size, out_dim]

        return x
    
    def forward_film_simple(self, x, gamma, beta):
        
        # x.shape == [batch_size, grid_size, grid_size, in_dim]
        B, M, N, I = x.shape
        O = self.out_dim
 
        # res.shape == [batch_size, grid_size, grid_size, out_dim]

        x = rearrange(x, 'b m n i -> b i m n')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]
        
        res = self.dilatation_layer(x)

        x_ft = torch.fft.rfft2(x, s=(M, N), norm='ortho')
        # x_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1]

        x_ft = torch.stack([x_ft.real, x_ft.imag], dim=4)
        # x_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1, 2]

        out_ft = torch.zeros(B, O, N, M // 2 + 1, 2, device=x.device)
        # out_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1, 2]

        out_ft[:, :, :self.n_modes, :self.n_modes] = self.complex_matmul_2d(
            x_ft[:, :, :self.n_modes, :self.n_modes], self.fourier_weight[0])

        out_ft[:, :, -self.n_modes:, :self.n_modes] = self.complex_matmul_2d(
            x_ft[:, :, -self.n_modes:, :self.n_modes], self.fourier_weight[1])

        out_ft = torch.complex(out_ft[..., 0], out_ft[..., 1])

        x = torch.fft.irfft2(out_ft, s=(N, M), norm='ortho')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        if self.residual:
            
            x = x + res

            beta = beta.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            gamma = gamma.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            
            x = self.act(gamma*x + beta)

        
        x = rearrange(x, 'b i m n -> b m n i')
        # x.shape == [batch_size, grid_size, grid_size, out_dim]


            
        return x
