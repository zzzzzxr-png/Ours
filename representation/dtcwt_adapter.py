import torch
import complextorch.nn as cvnn

from .dtcwt import DTCWTCoefficients


def _repeat_to(value, spatial_size):
    height, width = spatial_size
    if height % value.shape[-2] or width % value.shape[-1]:
        raise ValueError('target size must be an integer multiple of coefficient size')
    return value.repeat_interleave(height // value.shape[-2], dim=-2).repeat_interleave(
        width // value.shape[-1], dim=-1)


def _block_mean(value, spatial_size):
    height, width = spatial_size
    factor_h = value.shape[-2] // height
    factor_w = value.shape[-1] // width
    if factor_h * height != value.shape[-2] or factor_w * width != value.shape[-1]:
        raise ValueError('coefficient size must divide the aligned feature size')
    batch, channels, time = value.shape[:3]
    return value.reshape(batch, channels, time, height, factor_h, width, factor_w).mean(
        dim=(4, 6))


class FullResolutionDTCWTAdapter(torch.nn.Module):
    """Fixed scale alignment plus learned complex channel projections."""

    def __init__(self, image_channels=1, levels=3, feature_channels=8):
        super().__init__()
        self.levels = int(levels)
        self._feature_channels = int(feature_channels)
        if self._feature_channels < 1:
            raise ValueError('DTCWT adapter feature_channels must be positive')
        self.branch_channels = [2 * image_channels] + [6 * image_channels] * self.levels
        self.input_projections = torch.nn.ModuleList([
            cvnn.Conv3d(channels, self._feature_channels, kernel_size=1, bias=False)
            for channels in self.branch_channels
        ])
        self.output_projections = torch.nn.ModuleList([
            cvnn.Conv3d(self._feature_channels, channels, kernel_size=1)
            for channels in self.branch_channels
        ])
        for projection in self.output_projections:
            for parameter in projection.parameters():
                torch.nn.init.zeros_(parameter)

    @property
    def feature_channels(self):
        return self._feature_channels

    def encode(self, coefficients):
        branches = (coefficients.low,) + coefficients.highs
        return sum(
            _repeat_to(projection(value), coefficients.spatial_size)
            for projection, value in zip(self.input_projections, branches)
        )

    def decode_residual(self, features, reference):
        targets = (reference.low,) + reference.highs
        residuals = [
            _block_mean(projection(features), target.shape[-2:])
            for projection, target in zip(self.output_projections, targets)
        ]
        if residuals[0].shape != reference.low.shape or any(
                value.shape != target.shape
                for value, target in zip(residuals[1:], reference.highs)):
            raise RuntimeError('DTCWT adapter produced inconsistent coefficient shapes')
        return DTCWTCoefficients(
            reference.low + residuals[0],
            tuple(target + value for target, value in zip(reference.highs, residuals[1:])),
            reference.spatial_size,
            reference.image_channels,
        )
