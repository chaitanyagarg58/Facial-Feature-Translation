import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
import os
from torchvision.utils import save_image

class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)
    
class GeneratorATTENTION(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super(GeneratorATTENTION, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3+1, kernel_size=7, stride=1, padding=3, bias=False))
        # layers.append(nn.Tanh()) # ht
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        input_image = x
        x = torch.cat([x, c], dim=1)
        # return self.main(x)
        # print(x.size())
        output = self.main(x)
        # print(output.size())
        attention_mask = F.sigmoid(output[:, :1])
        content_mask = output[:, 1:]
        attention_mask = attention_mask.repeat(1, 3, 1, 1)
        result = content_mask * attention_mask + input_image * (1 - attention_mask)
        return result, attention_mask, content_mask

def create_labels(c_org, selected_attrs, attr_idx):
    """Generate target domain labels for debugging and testing."""
    c_trg = c_org.clone()
    if selected_attrs[attr_idx] in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:  # Set one hair color to 1 and the rest to 0.
        c_trg[:, attr_idx] = 1
    else:
        c_trg[:, attr_idx] = (c_trg[:, attr_idx] == 0)  # Reverse attribute value.
    
    return c_trg
    
def generate_attention_image(model, data_loader, selected_attrs, type):
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    def denorm(x):
        out = (x + 1) / 2
        return out.clamp_(0, 1)
    
    
    with torch.no_grad():
        for i, (x_real, c_org) in enumerate(data_loader):
            for idx, selected_attr in enumerate(selected_attrs):
                c_trg = create_labels(c_org, selected_attrs, idx)
                
                fake, _, _ = model(x_real, c_trg)
                
                # Save the translated images.
                x_concat = torch.cat([fake], dim=3)
                result_path = os.path.join(f"results/{type}/attentiongan/{selected_attr}", f'{i+1}_{selected_attr}.jpg')
                save_image(denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)