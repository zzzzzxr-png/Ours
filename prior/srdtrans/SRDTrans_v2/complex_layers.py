"""Small glue required around ComplexTorch for phase-preserving dropout."""

import torch
from torch import nn


class SharedDropout(nn.Module):
    """Elementwise dropout with one real mask shared by real and imaginary parts."""

    def __init__(self, p=0.5):
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError("dropout probability must be in [0,1), got {}".format(p))
        self.p = float(p)

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        mask = torch.empty_like(x.real).bernoulli_(1.0 - self.p).div_(1.0 - self.p)
        return x * mask

    def extra_repr(self):
        return "p={}".format(self.p)
