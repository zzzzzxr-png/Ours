"""Small glue required around ComplexTorch for stable complex activations."""

import torch
from torch import nn


class ComplexRMSNorm3d(nn.Module):
    """Per-sample, per-channel RMS normalization that preserves complex phase."""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = float(eps)

    def forward(self, x):
        if x.ndim != 5 or not x.is_complex():
            raise ValueError("expected complex [B,C,T,H,W] input")
        rms = x.abs().square().mean(dim=(2, 3, 4), keepdim=True)
        return x * torch.rsqrt(rms + self.eps)


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
