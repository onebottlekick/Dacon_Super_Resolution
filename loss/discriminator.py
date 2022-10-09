import torch
import torch.nn as nn


def calc_shape(input_size, kernel_size=3, stride=2, padding=1, iter_num=4):
    def _calc_shape(input_size, kernel_size, stride, padding):
        h, w = input_size
        h = (h + 2*padding - kernel_size)//stride + 1
        w = (w + 2*padding - kernel_size)//stride + 1
        return h, w
    for _ in range(iter_num):
        input_size = _calc_shape(input_size, kernel_size, stride, padding)
    return input_size


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation=None, batchnorm=False):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.batchnorm = nn.BatchNorm2d(out_channels) if batchnorm else nn.Identity()
        self.activation = activation if activation is not None else nn.Identity()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.activation(x)
        
        return x
    
    
class Discriminator(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.conv1 = ConvBlock(3, 64, kernel_size=3, batchnorm=False, activation=nn.LeakyReLU(0.2))
        self.conv2 = ConvBlock(64, 64, stride=2, activation=nn.LeakyReLU(0.2), batchnorm=True)
        self.conv3 = ConvBlock(64, 128, activation=nn.LeakyReLU(0.2), batchnorm=True)
        self.conv4 = ConvBlock(128, 128, stride=2, activation=nn.LeakyReLU(0.2), batchnorm=True)
        self.conv5 = ConvBlock(128, 256, activation=nn.LeakyReLU(0.2), batchnorm=True)
        self.conv6 = ConvBlock(256, 256, stride=2, activation=nn.LeakyReLU(0.2), batchnorm=True)
        self.conv7 = ConvBlock(256, 512, activation=nn.LeakyReLU(0.2), batchnorm=True)
        self.conv8 = ConvBlock(512, 512, stride=2, activation=nn.LeakyReLU(0.2), batchnorm=True)
        h, w = calc_shape((args.hr_size, args.hr_size))
        self.fc1 = nn.Linear(512*h*w, 1024)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(1024, 1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.leakyrelu(x)
        x = self.fc2(x)
        
        return x