import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
import os
from torchvision.utils import save_image

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


class GeneratorSTAR(nn.Module):
    def __init__(self, conv_dim=64, c_dim=5):
        super(GeneratorSTAR, self).__init__()
        
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

def create_labels(c_org, selected_attrs, attr_idx):
    """Generate target domain labels for debugging and testing."""
    c_trg = c_org.clone()
    if selected_attrs[attr_idx] in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:  # Set one hair color to 1 and the rest to 0.
        c_trg[:, attr_idx] = 1
    else:
        c_trg[:, attr_idx] = (c_trg[:, attr_idx] == 0)  # Reverse attribute value.
    
    return c_trg

def generate_star_image(model, data_loader, selected_attrs, type):

    def denorm(x):
        out = (x + 1) / 2
        return out.clamp_(0, 1)
        
    # device = ("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for i, (x_real, c_org) in enumerate(data_loader):
            for idx, selected_attr in enumerate(selected_attrs):

                c_trg = create_labels(c_org, selected_attrs, idx)
                
                x_concat = torch.cat([model(x_real, c_trg)], dim=3)
                result_path = os.path.join(f"results/{type}/stargan/{selected_attr}", f'{i+1}_{selected_attr}.jpg')
                save_image(denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
