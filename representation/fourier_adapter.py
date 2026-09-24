"""Learned branch alignment for the plenoptic Fourier pyramid."""

import torch
import complextorch.nn as cvnn

from .fourier_pyramid import FourierPyramidCoefficients
from .dtcwt_adapter import _learned_head


class LegacyFourierPyramidAdapter(torch.nn.Module):
    """Original learned 20-channel alignment used by the patch-64 experiment."""

    branch_channels = [1, 6, 6, 6, 1]

    @property
    def feature_channels(self):
        return sum(self.branch_channels)

    def __init__(self):
        super().__init__()
        factors = [1, 1, 2, 4, 8]
        self.analysis_heads = torch.nn.ModuleList([
            _learned_head(channels, factor, up=True)
            for channels, factor in zip(self.branch_channels, factors)
        ])
        self.synthesis_heads = torch.nn.ModuleList([
            _learned_head(channels, factor, up=False)
            for channels, factor in zip(self.branch_channels, factors)
        ])

    def encode(self, coefficients):
        branches = (coefficients.highpass,) + coefficients.bands + (coefficients.lowpass,)
        aligned = tuple(head(value) for head, value in zip(self.analysis_heads, branches))
        return torch.cat(aligned, dim=1)

    def decode(self, features, reference):
        if features.shape[1] != self.feature_channels:
            raise ValueError('expected legacy Fourier features with C=20')
        branches = torch.split(features, self.branch_channels, dim=1)
        decoded = tuple(head(value) for head, value in zip(self.synthesis_heads, branches))
        return FourierPyramidCoefficients(
            decoded[0], tuple(decoded[1:-1]), decoded[-1],
            reference.spatial_size, reference.image_channels,
        )


class LearnedFourierPyramidAdapter(torch.nn.Module):
    """Align Fourier pyramid branches to the SRDTrans coefficient grid."""

    def __init__(self, image_channels=1, height=3, orientations=6):
        super().__init__()
        if int(image_channels) != 1 or int(height) != 3 or int(orientations) != 6:
            raise ValueError('current SRDTrans Fourier adapter expects 1x3x6 branches')
        # Learned overlapping spatial reduction; not an invertible transform.
        self.branch_channels = [3, 20, orientations, 2, 1]
        self.analysis_heads = torch.nn.ModuleList([
            cvnn.Conv3d(1, 3, (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            cvnn.Conv3d(orientations, 20, (1, 3, 3),
                        stride=(1, 2, 2), padding=(0, 1, 1)),
            _learned_head(orientations, 1, up=True),  # scale 1 H/2
            torch.nn.Sequential(
                _learned_head(orientations, 2, up=True),
                cvnn.Conv3d(orientations, 2, (1, 1, 1)),
            ),  # scale 2 H/4 -> H/2, then 6 -> 2 channels
            _learned_head(1, 4, up=True),         # lowpass H/8 -> H/2
        ])
        self.synthesis_heads = torch.nn.ModuleList([
            cvnn.ConvTranspose3d(3, 1, (1, 3, 3), stride=(1, 2, 2),
                                 padding=(0, 1, 1), output_padding=(0, 1, 1)),
            cvnn.ConvTranspose3d(20, orientations, (1, 3, 3),
                                 stride=(1, 2, 2), padding=(0, 1, 1),
                                 output_padding=(0, 1, 1)),
            _learned_head(orientations, 1, up=True),
            torch.nn.Sequential(
                cvnn.Conv3d(2, orientations, (1, 1, 1)),
                _learned_head(orientations, 2, up=False),
            ),
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
        targets = (reference.highpass,) + reference.bands + (reference.lowpass,)
        if any(value.shape != target.shape for value, target in zip(decoded, targets)):
            raise ValueError('Fourier decoded coefficient shapes do not match reference')
        return FourierPyramidCoefficients(
            highpass=decoded[0],
            bands=tuple(decoded[1:-1]),
            lowpass=decoded[-1],
            spatial_size=reference.spatial_size,
            image_channels=reference.image_channels,
        )
