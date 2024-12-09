import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_dim, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_dim, affine=True, track_running_stats=True)
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, conv_dim=64, c_dim=5):
        super(Generator, self).__init__()
        
        layers = []
        layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        curr_dim = conv_dim
        for _ in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        for _ in range(6):
            layers.append(ResidualBlock(in_dim=curr_dim, out_dim=curr_dim))
        
        for _ in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.gen = nn.Sequential(*layers)

    def forward(self, x, c):
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        return self.gen(x)


class Discriminator(nn.Module):
    def __init__(self, image_sz, conv_dim, c_dim):
        super(Discriminator, self).__init__()

        layers = []
        
        ## First Conv layer 3*x*y => conv_dim * x *y
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1, bias=False))
        layers.append(nn.LeakyReLU(0.01, inplace=True))

        curr_dim = conv_dim
        num_layers = 6
        for _ in range(1,num_layers):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(curr_dim*2))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            curr_dim *= 2
        
        self.disc = nn.Sequential(*layers)
        self.label = nn.Sequential(self.disc, nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False))
        self.classes = nn.Sequential(self.disc, nn.Flatten(), nn.Linear(((image_sz // (2 ** (num_layers)))**2)*curr_dim, c_dim, bias=False))

    def forward(self, x):
        out_label = self.label(x)
        out_classes = self.classes(x)
        return out_label, out_classes.view(out_label.size(0), -1)

        