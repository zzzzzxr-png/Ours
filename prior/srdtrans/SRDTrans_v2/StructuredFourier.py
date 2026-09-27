"""Native-pyramid wrapper around the unchanged complex SRDTrans core."""

import inspect

import torch
from einops import rearrange
from torch import nn
import complextorch.nn as cvnn

from SRDTrans_v2.complex_layers import ComplexRMSNorm3d
from representation.fourier_pyramid import FourierPyramid2D, FourierPyramidCoefficients


def _complex_block(cin, cout):
    return nn.Sequential(cvnn.Conv3d(cin, cout, 1), ComplexRMSNorm3d(), cvnn.modReLU())


def _require_masked_complex_attention():
    if ('residual_norm' not in inspect.signature(cvnn.MultiheadAttention).parameters
            or 'attn_mask' not in inspect.signature(cvnn.MultiheadAttention.forward).parameters):
        raise RuntimeError(
            'steerable_fourier_structured requires ComplexTorch main (2.2.0 API) '
            'with MultiheadAttention residual_norm and attn_mask support; '
            'the installed package is too old')


class SharedDirectionalEmbedding(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.proj = cvnn.Conv3d(1, channels, 1)

    def forward(self, x):
        b, o, t, h, w = x.shape
        x = rearrange(x, 'b o t h w -> (b o) 1 t h w')
        x = self.proj(x)
        return rearrange(x, '(b o) c t h w -> b o c t h w', b=b, o=o)


class OrientationInteraction(nn.Module):
    def __init__(self, channels, heads=4, dropout=0.0):
        super().__init__()
        if channels % heads:
            raise ValueError('orientation channels {} must divide orientation_heads {}'.format(
                channels, heads))
        self.attn = cvnn.MultiheadAttention(
            n_heads=heads, d_model=channels, d_k=channels // heads,
            d_v=channels // heads, dropout=dropout, softmax_on='real',
            residual_norm=False)
        self.gate = nn.Parameter(torch.full((), 1e-3))
        self.register_buffer('relative_orientation_index', self._make_relative_index())
        self.orientation_bias = nn.Parameter(torch.zeros(4))

    @staticmethod
    def _make_relative_index():
        i = torch.arange(6)[:, None]
        d = (i - i.T).abs()
        return torch.minimum(d, 6 - d)

    def forward(self, x):
        b, o, c, t, h, w = x.shape
        tokens = rearrange(x, 'b o c t h w -> (b t h w) o c')
        bias = self.orientation_bias[self.relative_orientation_index]
        y = self.attn(tokens, tokens, tokens, attn_mask=bias)
        y = rearrange(y, '(b t h w) o c -> b o c t h w', b=b, t=t, h=h, w=w)
        return x + self.gate.to(x.real.dtype) * y


class TemporalBranchAlign(nn.Module):
    def __init__(self, channels, strides):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                cvnn.Conv3d(channels, channels, (3, 1, 1), stride=(int(stride), 1, 1), padding=(1, 0, 0)),
                ComplexRMSNorm3d(), cvnn.modReLU()
            ) for stride in strides
        ])

    def forward(self, x):
        six = x.ndim == 6
        if six:
            b, o, c, t, h, w = x.shape
            x = rearrange(x, 'b o c t h w -> (b o) c t h w')
        for block in self.blocks:
            x = block(x)
        if six:
            x = rearrange(x, '(b o) c t h w -> b o c t h w', b=b, o=o)
        return x


class CrossScaleInjection(nn.Module):
    def __init__(self, channels, ratio, heads=4):
        super().__init__()
        if channels % heads:
            raise ValueError('stage channels {} must divide orientation_heads {}'.format(
                channels, heads))
        self.ratio = int(ratio)
        self.attn = cvnn.MultiheadAttention(
            n_heads=heads, d_model=channels, d_k=channels // heads,
            d_v=channels // heads, dropout=0.0, softmax_on='real',
            residual_norm=False)
        self.gate = nn.Parameter(torch.full((), 1e-3))

    def forward(self, trunk, branch):
        b, c, t, h, w = trunk.shape
        r = self.ratio
        if branch.ndim != 6 or branch.shape[2] != c or branch.shape[3] != t:
            raise ValueError('branch must be [B,O,C,T,H,W] aligned to trunk T/C; '
                             'got trunk={} branch={}'.format(tuple(trunk.shape), tuple(branch.shape)))
        if h % r or w % r or branch.shape[-2:] != (h // r, w // r):
            raise ValueError('native parent-child shapes do not match: trunk={} branch={} r={}'
                             .format(tuple(trunk.shape), tuple(branch.shape), r))
        q = rearrange(trunk, 'b c t (h rh) (w rw) -> (b t h w) (rh rw) c', rh=r, rw=r)
        kv = rearrange(branch, 'b o c t h w -> (b t h w) o c')
        message = self.attn(q, kv, kv)
        message = rearrange(message, '(b t h w) (rh rw) c -> b c t (h rh) (w rw)',
                            b=b, t=t, h=h // r, w=w // r, rh=r, rw=r)
        return trunk + self.gate.to(trunk.real.dtype) * message


class StructuredCoefficientHead(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.highpass = cvnn.Conv3d(channels, 1, 1)
        self.scale0 = cvnn.Conv3d(channels, 6, 1)
        self.scale1 = cvnn.Conv3d(channels * 4, 6, 1)
        self.scale2 = cvnn.Conv3d(channels * 16, 6, 1)
        self.lowpass = cvnn.Conv3d(channels * 64, 1, 1)
        for head in (self.highpass, self.scale0, self.scale1, self.scale2, self.lowpass):
            with torch.no_grad():
                head.conv.weight.mul_(1e-2)
                if head.conv.bias is not None:
                    nn.init.zeros_(head.conv.bias)

    @staticmethod
    def _pack(x, r):
        if x.shape[-2] % r or x.shape[-1] % r:
            raise ValueError('spatial size {} is not divisible by {}'.format(x.shape[-2:], r))
        return rearrange(x, 'b c t (h r1) (w r2) -> b (c r1 r2) t h w', r1=r, r2=r)

    def forward(self, x):
        return (self.highpass(x).real, self.scale0(x), self.scale1(self._pack(x, 2)),
                self.scale2(self._pack(x, 4)), self.lowpass(self._pack(x, 8)).real)


class StructuredFourierComplexBackbone(nn.Module):
    """Native Fourier branches around an unchanged SRDTrans temporal/Swin core."""
    def __init__(self, backbone, image_size, f_maps, orientation_heads=4,
                 channel_normalize=True, channel_scales=None):
        super().__init__()
        _require_masked_complex_attention()
        self.backbone = backbone
        self.representation = FourierPyramid2D(image_size=image_size, height=3, order=5)
        c0, c1, c2, c3 = map(int, f_maps)
        self.highpass = _complex_block(1, c0)
        self.directional = nn.ModuleList([
            SharedDirectionalEmbedding(c0), SharedDirectionalEmbedding(c1),
            SharedDirectionalEmbedding(c2), SharedDirectionalEmbedding(c3)])
        self.orientation = nn.ModuleList([
            OrientationInteraction(c, orientation_heads) for c in (c0, c1, c2)])
        strides = [int(encoder.down_sample.conv.stride[0]) for encoder in backbone.encoders]
        self.temporal = nn.ModuleList([
            TemporalBranchAlign(c, strides[:i]) for i, c in enumerate((c0, c1, c2, c3))])
        self.inject = nn.ModuleList([
            CrossScaleInjection(c0, 1, orientation_heads),
            CrossScaleInjection(c1, 2, orientation_heads),
            CrossScaleInjection(c2, 4, orientation_heads),
            CrossScaleInjection(c3, 8, orientation_heads)])
        self.head = StructuredCoefficientHead(c0)
        self.channel_normalize = bool(channel_normalize)
        if self.channel_normalize:
            if channel_scales is None or len(channel_scales) != 20:
                raise ValueError('structured Fourier normalization needs 20 scales')
            scales = torch.as_tensor(channel_scales, dtype=torch.float32)
        else:
            scales = torch.ones(20, dtype=torch.float32)
        self.register_buffer('channel_scales', scales)
        self.orientation_heads = int(orientation_heads)

    def _scale(self, coeffs, inverse=False):
        values = (coeffs.highpass,) + coeffs.bands + (coeffs.lowpass,)
        out, offset = [], 0
        for value in values:
            n = value.shape[1]
            s = self.channel_scales[offset:offset + n].to(value.dtype).reshape(1, n, 1, 1, 1)
            out.append(value * s if inverse else value / s)
            offset += n
        return FourierPyramidCoefficients(out[0], tuple(out[1:-1]), out[-1],
                                           coeffs.spatial_size, coeffs.image_channels)

    def _run_stage(self, index, trunk, branch):
        encoder = self.backbone.encoders[index]
        trunk = encoder.conv_net(trunk)
        if index < 3:
            branch = self.directional[index](branch)
            branch = self.orientation[index](branch)
        else:
            branch = self.directional[index](branch)
        branch = self.temporal[index](branch)
        if branch.shape[3] != trunk.shape[2]:
            raise RuntimeError('structured Fourier branch T={} does not match stage {} T={}'
                               .format(branch.shape[3], index, trunk.shape[2]))
        trunk = self.inject[index](trunk, branch)
        before = trunk
        trunk = encoder.down_norm(encoder.down_sample(trunk))
        return before, trunk

    def forward(self, x):
        coeffs = self.representation(x)
        if self.channel_normalize:
            coeffs = self._scale(coeffs)
        branches = (coeffs.highpass, coeffs.bands[0], coeffs.bands[1],
                    coeffs.bands[2], coeffs.lowpass)
        trunk = self.highpass(branches[0])
        skips = []
        for i, encoder in enumerate(self.backbone.encoders):
            branch = branches[i + 1]
            if self.backbone.gradient_checkpointing and self.training and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint
                before, trunk = checkpoint(
                    lambda a, b, stage=i: self._run_stage(stage, a, b),
                    trunk, branch, use_reentrant=False)
            else:
                before, trunk = self._run_stage(i, trunk, branch)
            skips.insert(0, before)
        if self.backbone.gradient_checkpointing and self.training and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            trunk = checkpoint(self.backbone.process_by_trans, trunk, use_reentrant=False)
        else:
            trunk = self.backbone.process_by_trans(trunk)
        for decoder, skip in zip(self.backbone.decoders, skips):
            if self.backbone.gradient_checkpointing and self.training and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint
                trunk = checkpoint(decoder, trunk, skip, use_reentrant=False)
            else:
                trunk = decoder(trunk, skip)
        deltas = self.head(trunk)
        predicted = FourierPyramidCoefficients(
            branches[0] + deltas[0],
            tuple(branches[i + 1] + deltas[i + 1] for i in range(3)),
            branches[-1] + deltas[-1], coeffs.spatial_size, coeffs.image_channels)
        if self.channel_normalize:
            predicted = self._scale(predicted, inverse=True)
        return self.representation.inverse(predicted)
