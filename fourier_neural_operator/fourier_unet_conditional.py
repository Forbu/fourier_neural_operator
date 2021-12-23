from functools import partial

import torch
import torch.nn as nn
from einops import rearrange
import numpy as np

import fourier_neural_operator.layers.fourier_2d_factorized as fourier_2d_factorized
import fourier_neural_operator.layers.linear as linear 

import torch.nn.functional as F

class fourier_UNet_conditional(nn.Module):
    def __init__(self, n_channels, n_classes, n_conditions,n_modes=32):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_modes = n_modes

        self.inc = DoubleFourier(n_channels, 64, n_modes=n_modes)
        self.down1 = Down(64, 128, n_modes=n_modes)
        self.down2 = Down(128, 256, n_modes=n_modes//2)
        self.down3 = Down(256, 512, n_modes=n_modes//4)
        factor = 2
        self.down4 = Down(512, 1024 // factor, n_modes=n_modes//8)
        self.up1 = Up(1024, 512 // factor, n_modes=n_modes//8)
        self.up2 = Up(512, 256 // factor, n_modes=n_modes//4)
        self.up3 = Up(256, 128 // factor, n_modes=n_modes//2)
        self.up4 = Up(128, 64, n_modes=n_modes)
        self.outc = OutConv(64, n_classes)
        
        # film layer
        self.generator = nn.Sequential(
                        nn.Linear(n_conditions, 16),
                        nn.ReLU(),
                        nn.Linear(16, 256),
                        nn.ReLU(),            
                        nn.Dropout(0.5),
                        nn.BatchNorm1d(256, momentum=0.9),
                        nn.Linear(256, 1024),
                        nn.ReLU(),            
                        nn.Dropout(0.5),
                        nn.BatchNorm1d(1024, momentum=0.9),
                        )
        
        self.final_gamma = nn.Linear(1024, 4)
        self.final_beta = nn.Linear(1024, 4)

    def forward(self, x, condition):

        gammas, betas = self.get_film_layer_params(condition)
        
        x1 = self.inc(x)
        
        x2 = self.down1(x1)
        x2 = self.applied_film_layer(x2, gammas[:, 0], betas[:, 0])
        
        x3 = self.down2(x2)
        x3 = self.applied_film_layer(x3, gammas[:, 1], betas[:, 1])
        
        x4 = self.down3(x3)
        x4 = self.applied_film_layer(x4, gammas[:, 2], betas[:, 2])
        
        x5 = self.down4(x4)
        x5 = self.applied_film_layer(x5, gammas[:, 3], betas[:, 3])
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    
    def forward_film(self, x, condition):
        
        gammas, betas = self.get_film_layer_params(condition)
        
        x1 = self.inc(x)
        
        x2 = self.down1(x1, gammas[:, 0], betas[:, 0])
        x3 = self.down2(x2, gammas[:, 1], betas[:, 1])
        x4 = self.down3(x3, gammas[:, 2], betas[:, 2])
        x5 = self.down4(x4, gammas[:, 3], betas[:, 3])

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits      
    
    def get_film_layer_params(self,condition):
        
        x = self.generator(condition)
        return self.final_gamma(x), self.final_beta(x) 
        
    def applied_film_layer(self, x, gamma, beta):
        
        beta = beta.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        gamma = gamma.unsqueeze(1).unsqueeze(1).unsqueeze(1)

        return gamma*x + beta            

class DoubleFourier(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_dim, out_dim, n_modes=32):
        super().__init__()
        self.double_fourier = fourier_2d_factorized.SpectralConv2d(in_dim, out_dim, n_modes, resdiual=True, dropout=0.)

    def forward(self, x, gamma=None, beta=None):
        
        if gamma is not None:
            return self.double_fourier.forward_film_simple(x, gamma, beta)
        else:
            return self.double_fourier.forward(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, n_modes=32):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.fourier = DoubleFourier(in_channels, out_channels, n_modes=n_modes)
        
    def forward(self, x, gamma, beta):
        x = x.permute(0, 3, 1, 2)
        x = self.maxpool(x)
        x = x.permute(0, 2, 3, 1)
        return self.fourier(x, gamma, beta)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, n_modes=32):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleFourier(in_channels, out_channels, n_modes=n_modes)


    def forward(self, x1, x2):
        x1 = x1.permute(0, 3, 1, 2)
        x1 = self.up(x1)
        x1 = x1.permute(0, 2, 3, 1)
        
        # input is CHW
        diffY = x2.size()[1] - x1.size()[1]
        diffX = x2.size()[2] - x1.size()[2]

        x1 = F.pad(x1, [0, 0, diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=-1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        return self.conv(x)
