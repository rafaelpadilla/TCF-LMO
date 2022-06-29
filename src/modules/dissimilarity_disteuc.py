import torch
from torch.nn import Module


class DistEucModule(Module):
    def __init__(self, gamma_sigmoid=10):
        super(DistEucModule, self).__init__()
        self.combination_bias = torch.nn.Parameter(torch.tensor(-2.15))
        self.gamma_sigmoid = gamma_sigmoid

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
            p.grad = None

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True

    def forward(self, ref_tensor, tar_tensor):
        out = (tar_tensor - ref_tensor)**2
        out = out.sum(axis=1)
        out = torch.sqrt(out)

        out = torch.sigmoid(self.gamma_sigmoid * (out + self.combination_bias))
        return out
