import torch
import torch.nn as nn
from torchsummary import summary
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
from PIL import Image
import os


def add_normalization_1d(layers, fn, n_out):
    if fn == 'none':
        pass
    elif fn == 'batchnorm':
        layers.append(nn.BatchNorm1d(n_out))
    elif fn == 'instancenorm':
        layers.append(Unsqueeze(-1))
        layers.append(nn.InstanceNorm1d(n_out, affine=True))
        layers.append(Squeeze(-1))
    elif fn == 'switchnorm':
        layers.append(SwitchNorm1d(n_out))
    else:
        raise Exception('Unsupported normalization: ' + str(fn))
    return layers

def add_normalization_2d(layers, fn, n_out):
    if fn == 'none':
        pass
    elif fn == 'batchnorm':
        layers.append(nn.BatchNorm2d(n_out))
    elif fn == 'instancenorm':
        layers.append(nn.InstanceNorm2d(n_out, affine=True))
    elif fn == 'switchnorm':
        layers.append(SwitchNorm2d(n_out))
    else:
        raise Exception('Unsupported normalization: ' + str(fn))
    return layers

def add_activation(layers, fn):
    if fn == 'none':
        pass
    elif fn == 'relu':
        layers.append(nn.ReLU())
    elif fn == 'lrelu':
        layers.append(nn.LeakyReLU())
    elif fn == 'sigmoid':
        layers.append(nn.Sigmoid())
    elif fn == 'tanh':
        layers.append(nn.Tanh())
    else:
        raise Exception('Unsupported activation function: ' + str(fn))
    return layers

class Squeeze(nn.Module):
    def __init__(self, dim):
        super(Squeeze, self).__init__()
        self.dim = dim
    
    def forward(self, x):
        return x.squeeze(self.dim)

class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim
    
    def forward(self, x):
        return x.unsqueeze(self.dim)


class SwitchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.997, using_moving_average=True):
        super(SwitchNorm1d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.weight = nn.Parameter(torch.ones(1, num_features))
        self.bias = nn.Parameter(torch.zeros(1, num_features))
        self.mean_weight = nn.Parameter(torch.ones(2))
        self.var_weight = nn.Parameter(torch.ones(2))
        self.register_buffer('running_mean', torch.zeros(1, num_features))
        self.register_buffer('running_var', torch.zeros(1, num_features))
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.zero_()
        self.weight.data.fill_(1)
        self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 2:
            raise ValueError('expected 2D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        mean_ln = x.mean(1, keepdim=True)
        var_ln = x.var(1, keepdim=True)

        if self.training:
            mean_bn = x.mean(0, keepdim=True)
            var_bn = x.var(0, keepdim=True)
            if self.using_moving_average:
                self.running_mean.mul_(self.momentum)
                self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                self.running_var.mul_(self.momentum)
                self.running_var.add_((1 - self.momentum) * var_bn.data)
            else:
                self.running_mean.add_(mean_bn.data)
                self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
        else:
            mean_bn = torch.autograd.Variable(self.running_mean)
            var_bn = torch.autograd.Variable(self.running_var)

        softmax = nn.Softmax(0)
        mean_weight = softmax(self.mean_weight)
        var_weight = softmax(self.var_weight)

        mean = mean_weight[0] * mean_ln + mean_weight[1] * mean_bn
        var = var_weight[0] * var_ln + var_weight[1] * var_bn

        x = (x - mean) / (var + self.eps).sqrt()
        return x * self.weight + self.bias

class SwitchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.997, using_moving_average=True, using_bn=True,
                 last_gamma=False):
        super(SwitchNorm2d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.last_gamma = last_gamma
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        if self.using_bn:
            self.mean_weight = nn.Parameter(torch.ones(3))
            self.var_weight = nn.Parameter(torch.ones(3))
        else:
            self.mean_weight = nn.Parameter(torch.ones(2))
            self.var_weight = nn.Parameter(torch.ones(2))
        if self.using_bn:
            self.register_buffer('running_mean', torch.zeros(1, num_features, 1))
            self.register_buffer('running_var', torch.zeros(1, num_features, 1))

        self.reset_parameters()

    def reset_parameters(self):
        if self.using_bn:
            self.running_mean.zero_()
            self.running_var.zero_()
        if self.last_gamma:
            self.weight.data.fill_(0)
        else:
            self.weight.data.fill_(1)
        self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        N, C, H, W = x.size()
        x = x.view(N, C, -1)
        mean_in = x.mean(-1, keepdim=True)
        var_in = x.var(-1, keepdim=True)

        mean_ln = mean_in.mean(1, keepdim=True)
        temp = var_in + mean_in ** 2
        var_ln = temp.mean(1, keepdim=True) - mean_ln ** 2

        if self.using_bn:
            if self.training:
                mean_bn = mean_in.mean(0, keepdim=True)
                var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2
                if self.using_moving_average:
                    self.running_mean.mul_(self.momentum)
                    self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                    self.running_var.mul_(self.momentum)
                    self.running_var.add_((1 - self.momentum) * var_bn.data)
                else:
                    self.running_mean.add_(mean_bn.data)
                    self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
            else:
                mean_bn = torch.autograd.Variable(self.running_mean)
                var_bn = torch.autograd.Variable(self.running_var)

        softmax = nn.Softmax(0)
        mean_weight = softmax(self.mean_weight)
        var_weight = softmax(self.var_weight)

        if self.using_bn:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln + mean_weight[2] * mean_bn
            var = var_weight[0] * var_in + var_weight[1] * var_ln + var_weight[2] * var_bn
        else:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln
            var = var_weight[0] * var_in + var_weight[1] * var_ln

        x = (x-mean) / (var+self.eps).sqrt()
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias


class LinearBlock(nn.Module):
    def __init__(self, n_in, n_out, norm_fn='none', acti_fn='none'):
        super(LinearBlock, self).__init__()
        layers = [nn.Linear(n_in, n_out, bias=(norm_fn=='none'))]
        layers = add_normalization_1d(layers, norm_fn, n_out)
        layers = add_activation(layers, acti_fn)
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

class Conv2dBlock(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride=1, padding=0, 
                 norm_fn=None, acti_fn=None):
        super(Conv2dBlock, self).__init__()
        layers = [nn.Conv2d(n_in, n_out, kernel_size, stride=stride, padding=padding, bias=(norm_fn=='none'))]
        layers = add_normalization_2d(layers, norm_fn, n_out)
        layers = add_activation(layers, acti_fn)
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

class ConvTranspose2dBlock(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride=1, padding=0, 
                 norm_fn=False, acti_fn=None):
        super(ConvTranspose2dBlock, self).__init__()
        layers = [nn.ConvTranspose2d(n_in, n_out, kernel_size, stride=stride, padding=padding, bias=(norm_fn=='none'))]
        layers = add_normalization_2d(layers, norm_fn, n_out)
        layers = add_activation(layers, acti_fn)
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)
    
    
class Generator(nn.Module):
    def __init__(self, enc_dim=64, enc_layers=5, enc_norm_fn='batchnorm', enc_acti_fn='lrelu',
                 dec_dim=64, dec_layers=5, dec_norm_fn='batchnorm', dec_acti_fn='relu',
                 n_attrs=13, shortcut_layers=1, inject_layers=1, img_size=128):
        super(Generator, self).__init__()
        self.shortcut_layers = min(shortcut_layers, dec_layers - 1)
        self.inject_layers = min(inject_layers, dec_layers - 1)
        self.f_size = img_size // 2**enc_layers  # f_size = 4 for 128x128
        
        layers = []
        n_in = 3
        for i in range(enc_layers):
            n_out = min(enc_dim * 2**i, 64 * 16)
            layers += [Conv2dBlock(
                n_in, n_out, (4, 4), stride=2, padding=1, norm_fn=enc_norm_fn, acti_fn=enc_acti_fn
            )]
            n_in = n_out
        self.enc_layers = nn.ModuleList(layers)
        
        layers = []
        n_in = n_in + n_attrs  # 1024 + 13
        for i in range(dec_layers):
            if i < dec_layers - 1:
                n_out = min(dec_dim * 2**(dec_layers-i-1), 64 * 16)
                layers += [ConvTranspose2dBlock(
                    n_in, n_out, (4, 4), stride=2, padding=1, norm_fn=dec_norm_fn, acti_fn=dec_acti_fn
                )]
                n_in = n_out
                n_in = n_in + n_in//2 if self.shortcut_layers > i else n_in
                n_in = n_in + n_attrs if self.inject_layers > i else n_in
            else:
                layers += [ConvTranspose2dBlock(
                    n_in, 3, (4, 4), stride=2, padding=1, norm_fn='none', acti_fn='tanh'
                )]
        self.dec_layers = nn.ModuleList(layers)
    
    def encode(self, x):
        z = x
        zs = []
        for layer in self.enc_layers:
            z = layer(z)
            zs.append(z)
        return zs
    
    def decode(self, zs, a):
        a_tile = a.view(a.size(0), -1, 1, 1).repeat(1, 1, self.f_size, self.f_size)
        z = torch.cat([zs[-1], a_tile], dim=1)
        for i, layer in enumerate(self.dec_layers):
            z = layer(z)
            if self.shortcut_layers > i:  # Concat 1024 with 512
                z = torch.cat([z, zs[len(self.dec_layers) - 2 - i]], dim=1)
            if self.inject_layers > i:
                a_tile = a.view(a.size(0), -1, 1, 1) \
                          .repeat(1, 1, self.f_size * 2**(i+1), self.f_size * 2**(i+1))
                z = torch.cat([z, a_tile], dim=1)
        return z
    
    def forward(self, x, a=None, mode='enc-dec'):
        if mode == 'enc-dec':
            assert a is not None, 'No given attribute.'
            return self.decode(self.encode(x), a)
        if mode == 'enc':
            return self.encode(x)
        if mode == 'dec':
            assert a is not None, 'No given attribute.'
            return self.decode(x, a)
        raise Exception('Unrecognized mode: ' + mode)


class Discriminators(nn.Module):
    # No instancenorm in fcs in source code, which is different from paper.
    def __init__(self, dim=64, norm_fn='instancenorm', acti_fn='lrelu',
                 fc_dim=1024, fc_norm_fn='none', fc_acti_fn='lrelu', n_layers=5, img_size=128):
        super(Discriminators, self).__init__()
        self.f_size = img_size // 2**n_layers
        
        layers = []
        n_in = 3
        for i in range(n_layers):
            n_out = min(dim * 2**i, 64 * 16)
            layers += [Conv2dBlock(
                n_in, n_out, (4, 4), stride=2, padding=1, norm_fn=norm_fn, acti_fn=acti_fn
            )]
            n_in = n_out
        self.conv = nn.Sequential(*layers)
        self.fc_adv = nn.Sequential(
            LinearBlock(1024 * self.f_size * self.f_size, fc_dim, fc_norm_fn, fc_acti_fn),
            LinearBlock(fc_dim, 1, 'none', 'none')
        )
        self.fc_cls = nn.Sequential(
            LinearBlock(1024 * self.f_size * self.f_size, fc_dim, fc_norm_fn, fc_acti_fn),
            LinearBlock(fc_dim, 13, 'none', 'none')
        )
    
    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc_adv(h), self.fc_cls(h)
    

class AttGAN():
    def __init__(self, args):
        self.mode = args.mode
        self.gpu = args.gpu
        self.multi_gpu = args.multi_gpu if 'multi_gpu' in args else False
        self.lambda_1 = args.lambda_1
        self.lambda_2 = args.lambda_2
        self.lambda_3 = args.lambda_3
        self.lambda_gp = args.lambda_gp
        
        self.G = Generator(
            args.enc_dim, args.enc_layers, args.enc_norm, args.enc_acti,
            args.dec_dim, args.dec_layers, args.dec_norm, args.dec_acti,
            args.n_attrs, args.shortcut_layers, args.inject_layers, args.img_size
        )
        self.G.train()
        if self.gpu: self.G.cuda()
        summary(self.G, [(3, args.img_size, args.img_size), (args.n_attrs, 1, 1)], batch_size=4, device='cuda' if args.gpu else 'cpu')
        
        self.D = Discriminators(
            args.dis_dim, args.dis_norm, args.dis_acti,
            args.dis_fc_dim, args.dis_fc_norm, args.dis_fc_acti, args.dis_layers, args.img_size
        )
        self.D.train()
        if self.gpu: self.D.cuda()
        summary(self.D, [(3, args.img_size, args.img_size)], batch_size=4, device='cuda' if args.gpu else 'cpu')
        
        if self.multi_gpu:
            self.G = nn.DataParallel(self.G)
            self.D = nn.DataParallel(self.D)
        
        self.optim_G = optim.Adam(self.G.parameters(), lr=args.lr, betas=args.betas)
        self.optim_D = optim.Adam(self.D.parameters(), lr=args.lr, betas=args.betas)
    
    def set_lr(self, lr):
        for g in self.optim_G.param_groups:
            g['lr'] = lr
        for g in self.optim_D.param_groups:
            g['lr'] = lr
    
    def train(self):
        self.G.train()
        self.D.train()
    
    def eval(self):
        self.G.eval()
        self.D.eval()
    
    def save(self, path):
        states = {
            'G': self.G.state_dict(),
            'D': self.D.state_dict(),
            'optim_G': self.optim_G.state_dict(),
            'optim_D': self.optim_D.state_dict()
        }
        torch.save(states, path)
    
    def load(self, path):
        states = torch.load(path, map_location=lambda storage, loc: storage)
        if 'G' in states:
            self.G.load_state_dict(states['G'])
        if 'D' in states:
            self.D.load_state_dict(states['D'])
        if 'optim_G' in states:
            self.optim_G.load_state_dict(states['optim_G'])
        if 'optim_D' in states:
            self.optim_D.load_state_dict(states['optim_D'])
    
    def saveG(self, path):
        states = {
            'G': self.G.state_dict()
        }
        torch.save(states, path)



class Custom(data.Dataset):
    def __init__(self, data_path, attr_path, image_size, selected_attrs):
        self.data_path = data_path
        att_list = open(attr_path, 'r', encoding='utf-8').readlines()[1].split()
        atts = [att_list.index(att) + 1 for att in selected_attrs]
        self.images = np.loadtxt(attr_path, skiprows=2, usecols=[0], dtype=str)
        self.labels = np.loadtxt(attr_path, skiprows=2, usecols=atts, dtype=int)
        
        self.tf = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def __getitem__(self, index):
        img = self.tf(Image.open(os.path.join(self.data_path, self.images[index])))
        att = torch.tensor((self.labels[index] + 1) // 2)
        return img, att
    
    def __len__(self):
        return len(self.images)


def check_attribute_conflict(att_batch, att_name, att_names):
    def _get(att, att_name):
        if att_name in att_names:
            return att[att_names.index(att_name)]
        return None
    def _set(att, value, att_name):
        if att_name in att_names:
            att[att_names.index(att_name)] = value
    att_id = att_names.index(att_name)
    for att in att_batch:
        if att_name in ['Bald', 'Receding_Hairline'] and att[att_id] != 0:
            if _get(att, 'Bangs') != 0:
                _set(att, 1-att[att_id], 'Bangs')
        elif att_name == 'Bangs' and att[att_id] != 0:
            for n in ['Bald', 'Receding_Hairline']:
                if _get(att, n) != 0:
                    _set(att, 1-att[att_id], n)
                    _set(att, 1-att[att_id], n)
        elif att_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair'] and att[att_id] != 0:
            for n in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                if n != att_name and _get(att, n) != 0:
                    _set(att, 1-att[att_id], n)
        elif att_name in ['Straight_Hair', 'Wavy_Hair'] and att[att_id] != 0:
            for n in ['Straight_Hair', 'Wavy_Hair']:
                if n != att_name and _get(att, n) != 0:
                    _set(att, 1-att[att_id], n)
        elif att_name in ['Mustache', 'No_Beard'] and att[att_id] != 0:
            for n in ['Mustache', 'No_Beard']:
                if n != att_name and _get(att, n) != 0:
                    _set(att, 1-att[att_id], n)
    return att_batch

def generate_att_image(model, data_loader, selected_attrs_idx, attrs, type):
    model.eval()


    for idx, (img_a, att_a) in enumerate(data_loader):
        
        att_a = att_a.type(torch.float)
        att_b_list = []
        for i in selected_attrs_idx:
            tmp = att_a.clone()
            tmp[:, i] = 1 - tmp[:, i]
            tmp = check_attribute_conflict(tmp, attrs[i], attrs)
            att_b_list.append(tmp)

        with torch.no_grad():
            for i, att_b in enumerate(att_b_list):
                att_b_ = (att_b * 2 - 1) * 0.5
                att_b_[..., i - 1] = att_b_[..., i - 1] * 2.0


                vutils.save_image(
                    model(img_a, att_b_), os.path.join(f"results/{type}/attgan/{attrs[selected_attrs_idx[i]]}", f"{idx+1}_{attrs[selected_attrs_idx[i]]}.jpg"),
                    nrow=1, normalize=True, range=(-1., 1.)
                )
