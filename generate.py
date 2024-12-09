import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
import numpy as np
import warnings
import os
import argparse
from glob import glob
import torch.utils.data as data
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str, default="testData")
typ = parser.parse_args().type

############################################# DataLoader #############################################

class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, attr_path, selected_attrs, transform):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()

        self.num_images = len(self.test_dataset)

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')

            self.test_dataset.append([filename, label])
        print('Finished preprocessing the CelebA dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.test_dataset
        filename, label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        """Return the number of images."""
        return self.num_images


def get_loader(image_dir, attr_path, selected_attrs, crop_size=178, image_size=128):
    """Build and return a data loader."""
    transform = []
    transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    dataset = CelebA(image_dir, attr_path, selected_attrs, transform)
        
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=1)
    return data_loader

############################################# PARAMETERS #############################################
selected_attrs = ['Blond_Hair', 'Bald', 'Smiling', 'Young']
selected_attrs_att = ['Blond_Hair', 'Bald', 'Young']
data_loader = get_loader(f'{typ}/images', f'{typ}/attr.txt', selected_attrs)

############################################# STARGAN #############################################
from stargan_generator import ResidualBlock, GeneratorSTAR, generate_star_image

stargan = GeneratorSTAR(64,4)
stargan.load_state_dict(torch.load("StarGAN/stargan/models/200000-G.ckpt", map_location=lambda storage, loc: storage))

generate_star_image(stargan, data_loader, selected_attrs, typ)

############################################# ATTENTIONGAN #############################################
from attentiongan_generator import GeneratorATTENTION, generate_attention_image
attentiongan = GeneratorATTENTION(64,4)
attentiongan.load_state_dict(torch.load("AttentionGAN/stargan/models/200000-G.ckpt", map_location=lambda storage, loc: storage))

generate_attention_image(attentiongan, data_loader, selected_attrs, typ)
############################################# ATTGAN #############################################
from attgan_generator import *


parser = argparse.ArgumentParser()
parser.add_argument('--img_size', dest='img_size', type=int, default=128)
parser.add_argument('--shortcut_layers', dest='shortcut_layers', type=int, default=1)
parser.add_argument('--inject_layers', dest='inject_layers', type=int, default=1)
parser.add_argument('--enc_dim', dest='enc_dim', type=int, default=64)
parser.add_argument('--dec_dim', dest='dec_dim', type=int, default=64)
parser.add_argument('--dis_dim', dest='dis_dim', type=int, default=64)
parser.add_argument('--dis_fc_dim', dest='dis_fc_dim', type=int, default=1024)
parser.add_argument('--enc_layers', dest='enc_layers', type=int, default=5)
parser.add_argument('--dec_layers', dest='dec_layers', type=int, default=5)
parser.add_argument('--dis_layers', dest='dis_layers', type=int, default=5)
parser.add_argument('--enc_norm', dest='enc_norm', type=str, default='batchnorm')
parser.add_argument('--dec_norm', dest='dec_norm', type=str, default='batchnorm')
parser.add_argument('--dis_norm', dest='dis_norm', type=str, default='instancenorm')
parser.add_argument('--dis_fc_norm', dest='dis_fc_norm', type=str, default='none')
parser.add_argument('--enc_acti', dest='enc_acti', type=str, default='lrelu')
parser.add_argument('--dec_acti', dest='dec_acti', type=str, default='relu')
parser.add_argument('--dis_acti', dest='dis_acti', type=str, default='lrelu')
parser.add_argument('--dis_fc_acti', dest='dis_fc_acti', type=str, default='relu')
parser.add_argument('--lambda_1', dest='lambda_1', type=float, default=100.0)
parser.add_argument('--lambda_2', dest='lambda_2', type=float, default=10.0)
parser.add_argument('--lambda_3', dest='lambda_3', type=float, default=1.0)
parser.add_argument('--lambda_gp', dest='lambda_gp', type=float, default=10.0)
parser.add_argument('--mode', dest='mode', default='wgan', choices=['wgan', 'lsgan', 'dcgan'])
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5)
parser.add_argument('--beta2', dest='beta2', type=float, default=0.999)
parser.add_argument('--gpu', action='store_true')
parser.add_argument('--type', type=str, default="testData")
args = parser.parse_args()
args.n_attrs = 13
args.betas = (args.beta1, args.beta2)

attgan = AttGAN(args)

def find_model(path, epoch='latest'):
    if epoch == 'latest':
        files = glob(os.path.join(path, '*.pth'))
        file = sorted(files, key=lambda x: int(x.rsplit('.', 2)[1]))[-1]
    else:
        file = os.path.join(path, 'weights.{:d}.pth'.format(int(epoch)))
    assert os.path.exists(file), 'File not found: ' + file
    print('Find model of {} epoch: {}'.format(epoch, file))
    return file

attgan.load(find_model("AttGAN/output/128_shortcut1_inject1_none/checkpoint", 49))
attgan = attgan.G

## Attributes over which ATTGAN was trained
attrs = ["Bald", "Bangs", "Black_Hair", "Blond_Hair", "Brown_Hair", "Bushy_Eyebrows", "Eyeglasses", "Male", "Mouth_Slightly_Open", "Mustache", "No_Beard", "Pale_Skin", "Young"]
att_testData = Custom(f"{typ}/images", f"{typ}/attr.txt", (128, 128), attrs)
att_dataloader = data.DataLoader(att_testData, batch_size=1, num_workers=0, shuffle=False, drop_last=False)

generate_att_image(attgan, att_dataloader, [0, 3, 12], attrs, typ)