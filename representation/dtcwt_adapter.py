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


class _LearnedResize(torch.nn.Module):
    """Learnable complex 2x resize, initialized as repeat/block mean."""

    def __init__(self, channels, up):
        super().__init__()
        layer = (
            cvnn.ConvTranspose3d(
                channels, channels, kernel_size=(1, 2, 2),
                stride=(1, 2, 2), padding=0,
            )
            if up else
            cvnn.Conv3d(
                channels, channels, kernel_size=(1, 2, 2),
                stride=(1, 2, 2), padding=0,
            )
        )
        self.layer = layer
        convolution = next(layer.children())
        with torch.no_grad():
            convolution.weight.zero_()
            value = 1.0 if up else 0.25
            for channel in range(channels):
                convolution.weight[channel, channel].fill_(value)
            if convolution.bias is not None:
                convolution.bias.zero_()

    def forward(self, value):
        return self.layer(value)


class _LearnedChannelMix(torch.nn.Module):
    """Identity-initialized complex channel mixing without resizing."""

    def __init__(self, channels):
        super().__init__()
        self.layer = cvnn.Conv3d(channels, channels, kernel_size=1)
        convolution = next(self.layer.children())
        with torch.no_grad():
            convolution.weight.zero_()
            for channel in range(channels):
                convolution.weight[channel, channel].fill_(1.0)
            if convolution.bias is not None:
                convolution.bias.zero_()

    def forward(self, value):
        return self.layer(value)


def _learned_head(channels, factor, up):
    resize_count = factor.bit_length() - 1
    layers = [_LearnedResize(channels, up=up) for _ in range(resize_count)]
    return torch.nn.Sequential(*(layers or [_LearnedChannelMix(channels)]))


class LearnedDTCWTScaleAdapter(torch.nn.Module):
    """Lightweight learned analysis/synthesis heads around SRDTrans."""

    def __init__(self, image_channels=1, levels=3):
        super().__init__()
        self.levels = int(levels)
        self.branch_channels = [2 * image_channels] + [6 * image_channels] * self.levels
        factors = [2 ** (self.levels - 1)] + [2 ** index for index in range(self.levels)]
        self.analysis_heads = torch.nn.ModuleList([
            _learned_head(channels, factor, up=True)
            for channels, factor in zip(self.branch_channels, factors)
        ])
        self.synthesis_heads = torch.nn.ModuleList([
            _learned_head(channels, factor, up=False)
            for channels, factor in zip(self.branch_channels, factors)
        ])

    @property
    def feature_channels(self):
        return sum(self.branch_channels)

    def encode(self, coefficients):
        branches = (coefficients.low,) + coefficients.highs
        target_size = coefficients.highs[0].shape[-2:]
        aligned = tuple(head(value) for head, value in zip(self.analysis_heads, branches))
        if any(value.shape[-2:] != target_size for value in aligned):
            raise RuntimeError('learned DTCWT analysis heads produced inconsistent sizes')
        return torch.cat(aligned, dim=1)

    def decode(self, features, reference):
        expected_size = reference.highs[0].shape[-2:]
        if features.shape[1] != self.feature_channels or features.shape[-2:] != expected_size:
            raise ValueError(
                'expected aligned features with C={} and spatial size {}, got {}'.format(
                    self.feature_channels, expected_size, tuple(features.shape)
                )
            )
        aligned = torch.split(features, self.branch_channels, dim=1)
        predictions = tuple(
            head(value) for head, value in zip(self.synthesis_heads, aligned)
        )
        targets = (reference.low,) + reference.highs
        if any(value.shape != target.shape for value, target in zip(predictions, targets)):
            raise RuntimeError('learned DTCWT synthesis heads produced inconsistent sizes')
        return DTCWTCoefficients(
            predictions[0], tuple(predictions[1:]),
            reference.spatial_size, reference.image_channels,
        )
