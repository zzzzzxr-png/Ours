"""Learned branch alignment for the plenoptic Fourier pyramid."""

import torch
import torch.nn.functional as F

from .fourier_pyramid import FourierPyramidCoefficients
from .dtcwt_adapter import _learned_head


class LearnedFourierPyramidAdapter(torch.nn.Module):
    """Align Fourier pyramid branches to the SRDTrans coefficient grid."""

    def __init__(self, image_channels=1, height=3, orientations=6):
        super().__init__()
        if int(image_channels) != 1 or int(height) != 3 or int(orientations) != 6:
            raise ValueError('current SRDTrans Fourier adapter expects 1x3x6 branches')
        # Keep all branches on the H/2 grid. The full-resolution highpass is
        # rearranged losslessly into four polyphase channels first.
        self.branch_channels = [4] + [orientations] * height + [1]
        self.analysis_heads = torch.nn.ModuleList([
            _learned_head(4, 1, up=True),       # highpass H -> 4 x H/2
            _learned_head(orientations, 2, up=False),  # scale 0 H -> H/2
            _learned_head(orientations, 1, up=True),  # scale 1 H/2
            _learned_head(orientations, 2, up=True),  # scale 2 H/4 -> H/2
            _learned_head(1, 4, up=True),         # lowpass H/8 -> H/2
        ])
        self.synthesis_heads = torch.nn.ModuleList([
            _learned_head(4, 1, up=True),
            _learned_head(orientations, 2, up=True),
            _learned_head(orientations, 1, up=True),
            _learned_head(orientations, 2, up=False),
            _learned_head(1, 4, up=False),
        ])

    @property
    def feature_channels(self):
        return sum(self.branch_channels)

    def encode(self, coefficients):
        highpass = coefficients.highpass
        batch, channels, time, height, width = highpass.shape
        if channels != 1 or height % 2 or width % 2:
            raise ValueError('highpass must be [B,1,T,even H,even W]')
        highpass = F.pixel_unshuffle(
            highpass.permute(0, 2, 1, 3, 4).reshape(batch * time, channels, height, width), 2
        ).reshape(batch, time, 4 * channels, height // 2, width // 2).permute(0, 2, 1, 3, 4)
        branches = (highpass,) + coefficients.bands + (coefficients.lowpass,)
        aligned = tuple(head(value) for head, value in zip(self.analysis_heads, branches))
        target_size = aligned[0].shape[-2:]
        if any(value.shape[-2:] != target_size for value in aligned):
            raise RuntimeError('Fourier adapter produced inconsistent feature sizes')
        return torch.cat(aligned, dim=1)

    def decode(self, features, reference):
        if features.shape[1] != self.feature_channels:
            raise ValueError('expected Fourier features with C={}, got {}'.format(
                self.feature_channels, features.shape[1]
            ))
        branches = torch.split(features, self.branch_channels, dim=1)
        decoded = list(head(value) for head, value in zip(self.synthesis_heads, branches))
        highpass = decoded[0]
        batch, channels, time, height, width = highpass.shape
        highpass = F.pixel_shuffle(
            highpass.permute(0, 2, 1, 3, 4).reshape(batch * time, channels, height, width), 2
        ).reshape(batch, time, channels // 4, height * 2, width * 2).permute(0, 2, 1, 3, 4)
        decoded[0] = highpass
        return FourierPyramidCoefficients(
            highpass=decoded[0],
            bands=tuple(decoded[1:-1]),
            lowpass=decoded[-1],
            spatial_size=reference.spatial_size,
            image_channels=reference.image_channels,
        )
