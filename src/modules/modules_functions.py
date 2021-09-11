import torch
import torch.nn as nn
from torch.nn import Module, Parameter


class LearnableSigmoid(Module):
    '''
    sigmoid  = 1/(1+e^(-gamma*(x-t)))
    t: learnable parameter
    '''
    def __init__(self, threshold=None, gamma=1.):
        super(LearnableSigmoid, self).__init__()

        ########################
        # initialize variables #
        ########################
        if gamma is None:
            self.gamma = 1.
        else:
            self.gamma = gamma
        # threshold
        if threshold is None:
            self.threshold = Parameter(torch.tensor(0.5))
        else:
            self.threshold = Parameter(torch.tensor(threshold))

    def forward(self, x):
        # out = 1. / (1. + torch.exp(-self.gamma * (x - self.threshold)))
        out = torch.sigmoid(self.gamma * (x - self.threshold))
        return out

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
            p.grad = None

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True


class SumAllPixels(Module):
    def __init__(self):
        super(SumAllPixels, self).__init__()

    def forward(self, x):
        return x.sum(axis=(-1, -2))


class ChangeDynamicRange(Module):
    def __init__(self, min_in, max_in, min_out=0., max_out=1.):
        super(ChangeDynamicRange, self).__init__()
        self.min_in = min_in
        self.max_in = max_in
        self.min_out = min_out
        self.max_out = max_out

    def forward(self, x):
        return ((x - self.min_in) * (self.max_out - self.min_out) /
                (self.max_in - self.min_in)) + self.min_out


class InvertPixels(Module):
    def __init__(self):
        super(InvertPixels, self).__init__()

    def forward(self, x):
        return 1. - x


class FunctionThresholdByVolume(Module):
    def __init__(self, **kwargs):
        super().__init__()
        # Set default values
        if 'learnable_thresh' not in kwargs:
            kwargs['learnable_thresh'] = False
        if 'volume' not in kwargs:
            kwargs['volume'] = 0.
        if 'epsilon' not in kwargs:
            kwargs['epsilon'] = 0.

        self.learnable_thresh = kwargs['learnable_thresh']
        self.volume = kwargs['volume']
        self.epsilon = kwargs['epsilon']


class SigmoidThresholdByVolume(FunctionThresholdByVolume):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set default values
        if 'gamma' not in kwargs:
            kwargs['gamma'] = 1.
        self.gamma = kwargs['gamma']
        if self.learnable_thresh:
            self.volume = nn.Parameter(torch.tensor(self.volume))
        else:
            self.volume = torch.tensor(self.volume, requires_grad=False)

    def set_volume_threshold(self, volume):
        assert self.learnable_thresh is False  # make sure we're not overwriting a learnable parameter
        self.volume = volume * (1 - self.epsilon)

    def forward(self, x):
        '''Aplica threshold com base no volume'''
        out = torch.sigmoid(self.gamma * (x - self.volume))
        return out
