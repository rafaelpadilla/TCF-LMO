from collections import OrderedDict
from enum import Enum

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module

from .modules_functions import InvertPixels, SigmoidThresholdByVolume
from .se_sigmoid import StructuringElementSigmoid


class Morphology_Operation(Enum):
    Opening = 1
    Closing = 2
    Erosion = 3
    Dilate = 4


class DifferentialMorphology(Module):
    def __init__(self,
                 se_gamma=4,
                 se_radius=0,
                 kwargs_thresh_params=None,
                 epsilon=0.1,
                 operation=Morphology_Operation.Opening,
                 max_radius_kernel=(20, 20)):
        super().__init__()

        # Define initial variables
        self.operation = operation
        ############################
        # Kernel with sigmoid      #
        ############################
        self.se_sigmoid = StructuringElementSigmoid(radius=se_radius,
                                                    gamma=se_gamma,
                                                    max_radius=max_radius_kernel)
        ################################
        # Thresholding with Sigmoid    #
        ################################
        self.automatic_volume = True
        if operation == Morphology_Operation.Dilate:
            self.thresh_by_volume_dilate = SigmoidThresholdByVolume(
                gamma=kwargs_thresh_params['gamma'], volume=0., epsilon=epsilon)
        elif operation == Morphology_Operation.Erosion:
            self.thresh_by_volume_erosion = SigmoidThresholdByVolume(
                gamma=kwargs_thresh_params['gamma'], volume=0., epsilon=epsilon)
        elif operation == Morphology_Operation.Opening or operation == Morphology_Operation.Closing:
            self.thresh_by_volume_dilate = SigmoidThresholdByVolume(
                gamma=kwargs_thresh_params['gamma'], volume=0., epsilon=epsilon)
            self.thresh_by_volume_erosion = SigmoidThresholdByVolume(
                gamma=kwargs_thresh_params['gamma'], volume=0., epsilon=epsilon)
        #############################
        # Invert pixels operation   #
        #############################
        if operation in [
                Morphology_Operation.Dilate, Morphology_Operation.Opening,
                Morphology_Operation.Closing
        ]:
            self.invert_pixels_in = InvertPixels()
            self.invert_pixels_out = InvertPixels()
        #############################
        # Morphological operations  #
        #############################
        if operation == Morphology_Operation.Erosion:
            self.pipeline = nn.Sequential(
                OrderedDict([('conv_erosion', self.se_sigmoid),
                             ('thresh_erosion', self.thresh_by_volume_erosion)]))
        elif operation == Morphology_Operation.Dilate:
            self.pipeline = nn.Sequential(
                OrderedDict([('invert_in', self.invert_pixels_in), ('conv_dilate', self.se_sigmoid),
                             ('thresh_dilate', self.thresh_by_volume_dilate),
                             ('invert_out', self.invert_pixels_out)]))
        elif operation == Morphology_Operation.Opening:
            self.pipeline = nn.Sequential(  # erosion -> dilate
                OrderedDict([
                    # Erosion
                    ('conv_erosion', self.se_sigmoid),
                    ('thresh_erosion', self.thresh_by_volume_erosion),
                    # Dilate
                    ('invert_in', self.invert_pixels_in),
                    ('conv_dilate', self.se_sigmoid),
                    ('thresh_dilate', self.thresh_by_volume_dilate),
                    ('invert_out', self.invert_pixels_out),
                ]))
        elif operation == Morphology_Operation.Closing:
            self.pipeline = nn.Sequential(  # dilate -> erosion
                OrderedDict([
                    # Dilate
                    ('invert_in', self.invert_pixels_in),
                    ('conv_dilate', self.se_sigmoid),
                    ('thresh_dilate', self.thresh_by_volume_dilate),
                    ('invert_out', self.invert_pixels_out),
                    # Erosion
                    ('conv_erosion', self.se_sigmoid),
                    ('thresh_erosion', self.thresh_by_volume_erosion),
                ]))

    def forward(self, x):
        # Define threshold by volume
        if self.automatic_volume:
            volume = self.se_sigmoid.get_kernel().sum().item()
            if self.operation == Morphology_Operation.Dilate:
                self.thresh_by_volume_dilate.set_volume_threshold(volume)
            elif self.operation == Morphology_Operation.Erosion:
                self.thresh_by_volume_erosion.set_volume_threshold(volume)
            elif self.operation == Morphology_Operation.Opening or self.operation == Morphology_Operation.Closing:
                self.thresh_by_volume_erosion.set_volume_threshold(volume)
                self.thresh_by_volume_dilate.set_volume_threshold(volume)
        # If numpy, change it to torch
        if type(x) == np.ndarray:
            x = torch.tensor(x, dtype=torch.float)
        # Making x float (between 0 and 1)
        if x.dtype is not torch.float:
            x = x.type(dtype=torch.float)
        # Squeeze to make it with 4 dimensions: batch, channels, height, width
        # if it has 2 dimensions only (height, with)
        if x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        # if if has 3 dimensions (batch, height, width)
        elif x.ndim == 3:
            x = x.unsqueeze(1)
        return x

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
            p.grad = None

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True


class DifferentialErosion(DifferentialMorphology):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, operation=Morphology_Operation.Erosion)

    def forward(self, x):
        x = super().forward(x)
        return self.pipeline(x)


class DifferentialDilate(DifferentialMorphology):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, operation=Morphology_Operation.Dilate)

    def forward(self, x):
        x = super().forward(x)
        return self.pipeline(x)


class DifferentialOpening(DifferentialMorphology):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, operation=Morphology_Operation.Opening)

    def forward(self, x):
        x = super().forward(x)
        return self.pipeline(x)


class DifferentialClosing(DifferentialMorphology):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, operation=Morphology_Operation.Closing)

    def forward(self, x):
        x = super().forward(x)
        return self.pipeline(x)
