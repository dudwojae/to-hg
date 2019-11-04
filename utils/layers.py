import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
from utils import arglist


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def weight_init_fn(model):
    """
    General function for model weight initialization.
        Currently only implemented for simple layers.
    """
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight)
            nn.init.zeros_(m.bias)
        else:
            pass


def conv2d_init(in_, out_, stride, kernel_size, padding):
    relu_gain = nn.init.calculate_gain('relu')
    conv = nn.Conv2d(in_, out_, stride=stride,
                     kernel_size=kernel_size, padding=padding)
    conv.weight.data.mul_(relu_gain)
    return nn.DataParallel(conv)


def linear_init(in_, out_):
    relu_gain = nn.init.calculate_gain('relu')
    linear = nn.Linear(in_, out_)
    linear.weight.data.mul_(relu_gain)
    return nn.DataParallel(linear)
