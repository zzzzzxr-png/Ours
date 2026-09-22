"""GPU steerable Fourier pyramid for frame-wise video representations."""

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import torch
from torch import fft
from torch import nn

try:
    import plenoptic as po
except ImportError:  # Keep the existing DTCWT baseline usable without the optional package.
    po = None


if po is not None:
    _SteerablePyramidBase = po.process.SteerablePyramidFreq
else:
    _SteerablePyramidBase = nn.Module


class _BatchSafeSteerablePyramidFreq(_SteerablePyramidBase):
    """Fix plenoptic 2.1 reconstruction's batch-dimension fftshift."""

    def _recon_levels(self, pyr_coeffs, recon_levels, recon_bands, scale):
        if scale == self.num_scales:
            if 'residual_lowpass' in recon_levels:
                lodft = fft.fft2(
                    pyr_coeffs['residual_lowpass'], dim=(-2, -1), norm=self.fft_norm
                )
                return fft.fftshift(lodft, dim=(-2, -1))
            lodft = fft.fft2(
                torch.zeros_like(pyr_coeffs['residual_lowpass']),
                dim=(-2, -1), norm=self.fft_norm
            )
            return fft.fftshift(lodft, dim=(-2, -1))

        if scale in recon_levels:
            himask = getattr(self, f'_himasks_scale_{scale}')
            mask = getattr(self, f'_anglemasks_recon_scale_{scale}') * himask
            coeffs = pyr_coeffs[scale]
            if not isinstance(recon_bands, str):
                coeffs = coeffs[:, :, recon_bands]
                mask = mask[recon_bands]
            if self.tight_frame and self.is_complex:
                coeffs = coeffs * np.sqrt(2)
            orientdft = fft.fft2(coeffs, dim=(-2, -1), norm=self.fft_norm)
            orientdft = fft.fftshift(orientdft, dim=(-2, -1))
            orientdft = self._complex_const_recon * orientdft * mask
            orientdft = orientdft.sum(2)
        else:
            orientdft = torch.zeros_like(pyr_coeffs[scale][:, :, 0])

        lostart, loend = self._loindices[scale]
        lomask = getattr(self, f'_lomasks_scale_{scale}')
        reslevdft = self._recon_levels(pyr_coeffs, recon_levels, recon_bands, scale + 1)
        if (not self.tight_frame) and (not self.downsample):
            reslevdft = reslevdft / 2
        resdft = torch.zeros_like(pyr_coeffs[scale][:, :, 0], dtype=torch.complex64)
        resdft[..., lostart[0]:loend[0], lostart[1]:loend[1]] = reslevdft * lomask
        return resdft + orientdft


@dataclass
class FourierPyramidCoefficients:
    highpass: torch.Tensor
    bands: tuple
    lowpass: torch.Tensor
    spatial_size: tuple
    image_channels: int


class FourierPyramid2D(nn.Module):
    """Plenoptic frequency steerable pyramid on [B,C,T,H,W] videos."""

    def __init__(self, image_size, height=3, order=5, image_channels=1):
        super().__init__()
        if po is None:
            raise ImportError(
                'steerable_fourier representation requires plenoptic; install it with '
                '`python -m pip install plenoptic`'
            )
        if int(image_channels) != 1:
            raise NotImplementedError('Fourier pyramid currently supports one image channel')
        self.image_size = (int(image_size), int(image_size))
        self.height = int(height)
        self.order = int(order)
        self.image_channels = int(image_channels)
        self.pyramid = _BatchSafeSteerablePyramidFreq(
            self.image_size,
            height=self.height,
            order=self.order,
            is_complex=True,
            downsample=True,
            tight_frame=True,
        )

    @staticmethod
    def _to_video(value, batch, time, bands=None):
        if bands is None:
            channels, height, width = value.shape[1:]
            return value.reshape(batch, time, channels, height, width).permute(
                0, 2, 1, 3, 4
            )
        channels, _, height, width = value.shape[1:]
        return value.reshape(batch, time, channels, bands, height, width).permute(
            0, 2, 3, 1, 4, 5
        ).reshape(batch, channels * bands, time, height, width)

    @staticmethod
    def _to_frames(value, batch, time, bands=None):
        if bands is None:
            return value.permute(0, 2, 1, 3, 4).reshape(
                batch * time, value.shape[1], value.shape[-2], value.shape[-1]
            )
        channels = value.shape[1] // bands
        frames = value.reshape(
            batch, channels, bands, time, value.shape[-2], value.shape[-1]
        ).permute(0, 3, 1, 2, 4, 5)
        return frames.reshape(batch * time, channels, bands, value.shape[-2], value.shape[-1])

    @staticmethod
    def _as_complex(value):
        return torch.complex(value, torch.zeros_like(value)) if not value.is_complex() else value

    def forward(self, x):
        if x.ndim != 5 or x.shape[1] != self.image_channels or x.is_complex():
            raise ValueError('expected real [B,1,T,H,W], got {}'.format(
                (tuple(x.shape), x.dtype)
            ))
        if tuple(x.shape[-2:]) != self.image_size:
            raise ValueError('expected spatial size {}, got {}'.format(
                self.image_size, tuple(x.shape[-2:])
            ))
        batch, channels, time, height, width = x.shape
        frames = x.permute(0, 2, 1, 3, 4).reshape(batch * time, channels, height, width)
        coeffs = self.pyramid(frames)
        bands = tuple(self._to_video(coeffs[level], batch, time, coeffs[level].shape[2])
                      for level in range(self.height))
        return FourierPyramidCoefficients(
            highpass=self._as_complex(self._to_video(coeffs['residual_highpass'], batch, time)),
            bands=bands,
            lowpass=self._as_complex(self._to_video(coeffs['residual_lowpass'], batch, time)),
            spatial_size=(height, width),
            image_channels=channels,
        )

    def inverse(self, coefficients):
        batch, _, time = coefficients.highpass.shape[:3]
        pyramid_coeffs = OrderedDict()
        pyramid_coeffs['residual_highpass'] = self._to_frames(
            coefficients.highpass.real, batch, time
        )
        for level, value in enumerate(coefficients.bands):
            pyramid_coeffs[level] = self._to_frames(
                value, batch, time, value.shape[1] // coefficients.image_channels
            )
        pyramid_coeffs['residual_lowpass'] = self._to_frames(
            coefficients.lowpass.real, batch, time
        )
        frames = self.pyramid.recon_pyr(pyramid_coeffs)
        height, width = coefficients.spatial_size
        frames = frames[..., :height, :width]
        return frames.reshape(batch, time, coefficients.image_channels, height, width).permute(
            0, 2, 1, 3, 4
        )
