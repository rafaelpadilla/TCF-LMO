import math

import numpy as np
import PIL.Image as Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter


class StructuringElementSigmoid(Module):
    def __init__(self, radius=None, gamma=1., max_radius=(20, 20), device=None):
        super(StructuringElementSigmoid, self).__init__()
        ########################
        # initialize variables #
        ########################
        if gamma is None:
            self.gamma = 1.
        else:
            self.gamma = gamma
        # radius
        if radius is None:
            self.radius = Parameter(torch.tensor(0.5))
        else:
            self.radius = Parameter(torch.tensor(radius))
        # device
        if device is None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # min and max radius
        xs = torch.arange(-max_radius[1], max_radius[1] + 1, dtype=torch.float32)
        ys = torch.arange(-max_radius[0], max_radius[0] + 1, dtype=torch.float32)
        X, Y = torch.meshgrid(xs, ys)
        self.convolution = F.conv2d
        self.one_padding = nn.ConstantPad2d(max_radius[0], 1.)

    def get_kernel(self):
        # kernel = 1. / (1. + torch.exp(self.gamma * (self.X**2 + self.Y**2 - self.radius**2)))
        # OU
        kernel = torch.sigmoid(-self.gamma * (self.X**2 + self.Y**2 - self.radius**2))
        return kernel

    def forward(self, x):
        # generate kernel
        kernel = self.get_kernel()
        # Apply convolution using the kernel
        padded = self.one_padding(x)
        out = self.convolution(padded, kernel)
        return out
