from dataclasses import dataclass

import torch
from pytorch_wavelets import DTCWTForward, DTCWTInverse

from .base import Representation


@dataclass
class DTCWTCoefficients:
    low: torch.Tensor
    highs: tuple
    spatial_size: tuple
    image_channels: int


def unpack_low(yl):
    """Convert four interleaved real low trees to two complex maps."""
    y = yl / (2 ** 0.5)
    a, b = y[..., 0::2, 0::2], y[..., 0::2, 1::2]
    c, d = y[..., 1::2, 0::2], y[..., 1::2, 1::2]
    return torch.stack((torch.complex(a - d, b + c),
                        torch.complex(a + d, b - c)), dim=2)


def pack_low(low):
    """Differentiable inverse of unpack_low (library c2q uses in-place writes)."""
    z1, z2 = low.unbind(dim=2)
    top = torch.stack((z1.real + z2.real, z1.imag + z2.imag), dim=-1).flatten(-2)
    bottom = torch.stack((z1.imag - z2.imag, -z1.real + z2.real), dim=-1).flatten(-2)
    return torch.stack((top, bottom), dim=-2).flatten(-3, -2) / (2 ** 0.5)


class DTCWT2D(Representation):
    """Frame-wise official pytorch_wavelets DTCWT for [B,C,T,H,W]."""

    def __init__(self, levels=3, biort='near_sym_b', qshift='qshift_b'):
        super().__init__()
        self.levels = int(levels)
        if self.levels < 1:
            raise ValueError('DTCWT levels must be positive')
        self.analysis = DTCWTForward(J=self.levels, biort=biort, qshift=qshift)
        self.synthesis = DTCWTInverse(biort=biort, qshift=qshift)

    @staticmethod
    def _to_video(value, batch, time):
        _, channels, bands, height, width = value.shape
        return value.reshape(batch, time, channels, bands, height, width).permute(
            0, 2, 3, 1, 4, 5
        ).reshape(batch, channels * bands, time, height, width)

    @staticmethod
    def _to_frames(value, batch, time, channels, bands):
        height, width = value.shape[-2:]
        return value.reshape(batch, channels, bands, time, height, width).permute(
            0, 3, 1, 2, 4, 5
        ).reshape(batch * time, channels, bands, height, width)

    def forward(self, x):
        if x.ndim != 5 or x.is_complex() or not x.is_floating_point():
            raise ValueError('expected real floating [B,C,T,H,W], got {}'.format(
                (tuple(x.shape), x.dtype)))
        batch, channels, time, height, width = x.shape
        divisor = 2 ** self.levels
        if height % divisor or width % divisor:
            raise ValueError('H and W must be divisible by 2**levels={} for learned adapters, got {}x{}'.format(
                divisor, height, width))
        frames = x.permute(0, 2, 1, 3, 4).reshape(batch * time, channels, height, width)
        yl, yh = self.analysis(frames)
        low = self._to_video(unpack_low(yl), batch, time)
        highs = tuple(
            self._to_video(torch.view_as_complex(value.contiguous()), batch, time)
            for value in yh
        )
        return DTCWTCoefficients(low, highs, (height, width), channels)

    def inverse(self, coefficients):
        low, highs = coefficients.low, coefficients.highs
        batch, _, time = low.shape[:3]
        channels = coefficients.image_channels
        low_frames = self._to_frames(low, batch, time, channels, 2)
        yl = pack_low(low_frames)
        yh = [
            torch.view_as_real(
                self._to_frames(value, batch, time, channels, 6).contiguous()
            )
            for value in highs
        ]
        frames = self.synthesis((yl, yh))
        height, width = coefficients.spatial_size
        frames = frames[..., :height, :width]
        return frames.reshape(batch, time, channels, height, width).permute(0, 2, 1, 3, 4)
