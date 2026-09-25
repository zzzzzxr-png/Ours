"""Complex Restormer-style spatial block.

The MDTA/GDFN layout follows the official Restormer implementation; only the
operators are complex-valued so it can be used without discarding Fourier
phase.  This module is opt-in and does not alter the Swin baseline.
"""

import torch
from torch import nn
import torch.nn.functional as F
import complextorch.nn as cvnn


class ComplexMDTA(nn.Module):
    def __init__(self, dim, heads, bias=True):
        super().__init__()
        if dim % heads:
            raise ValueError('ComplexMDTA requires dim divisible by heads')
        self.heads = int(heads)
        self.temperature = nn.Parameter(torch.ones(self.heads, 1, 1))
        self.qkv = cvnn.Conv2d(dim, 3 * dim, kernel_size=1, bias=bias)
        self.qkv_dwconv = cvnn.Conv2d(
            3 * dim, 3 * dim, kernel_size=3, padding=1,
            groups=3 * dim, bias=bias,
        )
        self.project_out = cvnn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)
        d = c // self.heads
        q = q.reshape(b, self.heads, d, h * w)
        k = k.reshape(b, self.heads, d, h * w)
        v = v.reshape(b, self.heads, d, h * w)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.conj().transpose(-2, -1)).real * self.temperature
        out = (attn.softmax(dim=-1).to(v.dtype) @ v)
        return self.project_out(out.reshape(b, c, h, w))


class ComplexGDFN(nn.Module):
    def __init__(self, dim, expansion=2.66, bias=True):
        super().__init__()
        hidden = int(dim * expansion)
        self.project_in = cvnn.Conv2d(dim, 2 * hidden, kernel_size=1, bias=bias)
        self.dwconv = cvnn.Conv2d(
            2 * hidden, 2 * hidden, kernel_size=3, padding=1,
            groups=2 * hidden, bias=bias,
        )
        self.project_out = cvnn.Conv2d(hidden, dim, kernel_size=1, bias=bias)
        self.activation = cvnn.modReLU()

    def forward(self, x):
        x1, x2 = self.dwconv(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(self.activation(x1) * x2)


class ComplexRestormerBlock(nn.Module):
    def __init__(self, dim, heads, hidden_dim, dropout=0.0):
        super().__init__()
        self.norm1 = ComplexChannelNorm(dim)
        self.attn = ComplexMDTA(dim, heads)
        self.norm2 = ComplexChannelNorm(dim)
        self.ffn = ComplexGDFN(dim, expansion=hidden_dim / dim)
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()

    def forward(self, x):
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class ComplexChannelNorm(nn.Module):
    """Restormer LayerNorm over channels, while retaining NCHW layout."""

    def __init__(self, dim):
        super().__init__()
        self.norm = cvnn.LayerNorm(dim)

    def forward(self, x):
        b, c, h, w = x.shape
        y = x.permute(0, 2, 3, 1).reshape(-1, c)
        y = self.norm(y)
        return y.reshape(b, h, w, c).permute(0, 3, 1, 2)


class ComplexVideoChannelNorm(nn.Module):
    """Apply the same channel-last normalization independently per frame."""

    def __init__(self, dim):
        super().__init__()
        self.norm = ComplexChannelNorm(dim)

    def forward(self, x):
        b, c, t, h, w = x.shape
        y = self.norm(x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w))
        return y.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
