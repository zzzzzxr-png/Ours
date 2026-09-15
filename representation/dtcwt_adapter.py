import torch

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


class DTCWTScaleAdapter(torch.nn.Module):
    """Align the DTCWT pyramid on the finest coefficient grid."""

    def __init__(self, image_channels=1, levels=3):
        super().__init__()
        self.levels = int(levels)
        self.branch_channels = [2 * image_channels] + [6 * image_channels] * self.levels

    @property
    def feature_channels(self):
        return sum(self.branch_channels)

    def encode(self, coefficients):
        branches = (coefficients.low,) + coefficients.highs
        if len(branches) != len(self.branch_channels):
            raise ValueError('DTCWT coefficient level count does not match adapter')
        spatial_size = coefficients.highs[0].shape[-2:]
        return torch.cat(
            [_repeat_to(value, spatial_size) for value in branches], dim=1
        )

    def decode(self, features, reference):
        targets = (reference.low,) + reference.highs
        expected_size = reference.highs[0].shape[-2:]
        if features.shape[1] != self.feature_channels or features.shape[-2:] != expected_size:
            raise ValueError(
                'expected aligned features with C={} and spatial size {}, got {}'.format(
                    self.feature_channels, expected_size, tuple(features.shape)
                )
            )
        branches = torch.split(features, self.branch_channels, dim=1)
        predictions = [
            _block_mean(value, target.shape[-2:])
            for value, target in zip(branches, targets)
        ]
        if predictions[0].shape != reference.low.shape or any(
                value.shape != target.shape
                for value, target in zip(predictions[1:], reference.highs)):
            raise RuntimeError('DTCWT adapter produced inconsistent coefficient shapes')
        return DTCWTCoefficients(
            predictions[0],
            tuple(predictions[1:]),
            reference.spatial_size,
            reference.image_channels,
        )
