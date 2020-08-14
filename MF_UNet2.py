# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 23:07:22 2020

@author: zll
"""


import torch
import torch.nn as nn


class Conv_bn_ac(nn.Module):
    def __init__(self, in_ch, out_ch, group,
                 kernel_size=3, padding=1, stride=1):
        '''
        nn.Conv2d(input_channel, output_channel, kernel_size,
                  padding_for_each_edge)
        for a 3 * 3 kernel, padding=1 means zero-padding.
        '''
        super(Conv_bn_ac, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size,
                              stride, padding, groups=group)
        self.bn = nn.BatchNorm2d(out_ch)  # batch normolization
        self.ac = nn.ReLU(inplace=True)  # activate function

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.ac(x)
        return x


class MF_Unit(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(MF_Unit, self).__init__()
        mid_ch = min(in_ch, out_ch)
        if in_ch % 4 == 0:
            self.conv = nn.Sequential(
                Conv_bn_ac(in_ch, in_ch//4, 1, 1, padding=0, stride=1),
                Conv_bn_ac(in_ch//4, in_ch, 1, 1, padding=0, stride=1),
                Conv_bn_ac(in_ch, mid_ch, 8),
                Conv_bn_ac(mid_ch, out_ch, 8)
            )
        else:
            self.conv = nn.Sequential(
                Conv_bn_ac(in_ch, out_ch, 1),
                Conv_bn_ac(out_ch, out_ch, 1)
            )
        self.short_connect = nn.Conv2d(in_ch, out_ch, 1, padding=0)

    def forward(self, x):
        x1 = self.short_connect(x)
        x2 = self.conv(x)
        return x1 + x2  # residual connection


class MF_U_Net(nn.Module):
    def __init__(self):
        super(MF_U_Net, self).__init__()
        '''
        nn.ConvTranspose2d(input_channel, output_channel, kernel_size,
                  stride)
        '''

        self.conv1 = MF_Unit(1, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = MF_Unit(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = MF_Unit(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bridge = MF_Unit(256, 512)
        self.conv_trans1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv4 = MF_Unit(512, 256)
        self.conv_trans2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv5 = MF_Unit(256, 128)
        self.conv_trans3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv6 = MF_Unit(128, 64)
        self.conv7 = nn.Conv2d(64, 64, 3, padding=1)
        self.out = nn.Conv2d(64, 2, 1, padding=0)

    def forward(self, x):
        x1 = self.conv1(x)
        x = self.pool1(x1)
        x2 = self.conv2(x)
        x = self.pool2(x2)
        x3 = self.conv3(x)
        x = self.pool3(x3)
        x = self.bridge(x)
        x = self.conv_trans1(x)
        x = self.conv4(torch.cat((x, x3), dim=1))
        x = self.conv_trans2(x)
        x = self.conv5(torch.cat((x, x2), dim=1))
        x = self.conv_trans3(x)
        x = self.conv6(torch.cat((x, x1), dim=1))
        x = self.conv7(x)
        x = self.out(x)
        return x


if __name__ == "__main__":

    device = torch.device('cpu')
    x = torch.rand((1, 1, 256, 256), device=device)
    print("x size: {}".format(x.size()))
    model = MF_U_Net().to(device)
    out = model(x)
    print("out size:", out.shape)
