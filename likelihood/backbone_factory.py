"""Model factory for the SRDTrans-aligned (likelihood) training protocol."""

import math
import os
import sys

import torch
import torch.nn as nn

from prior.deepcadrt import Network_3D_Unet
from representation import (
    DTCWT2D,
    FourierPyramid2D,
    LearnedDTCWTScaleAdapter,
    LearnedFourierPyramidAdapter,
)

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEFAULT_SRDTRANS_ROOT = os.path.join(_PROJECT_ROOT, 'prior', 'srdtrans', 'SRDTrans_v2')


def raw_kappa_bias_from_init(kappa_init: float, kappa_min: float) -> float:
    """Bias init so exp(bias) + kappa_min ≈ kappa_init."""
    x = float(kappa_init) - float(kappa_min)
    if x <= 0:
        raise ValueError(
            'kappa_init must exceed kappa_min, got init={} min={}'.format(
                kappa_init, kappa_min
            )
        )
    return math.log(x)


class LearnedKappaHead(nn.Module):
    """Wrap a 1-channel backbone to emit [mu_x_centered, raw_kappa]."""

    def __init__(self, backbone: nn.Module, kappa_bias: float):
        super().__init__()
        self.backbone = backbone
        self.kappa_head = nn.Conv3d(1, 1, kernel_size=1)
        nn.init.zeros_(self.kappa_head.weight)
        nn.init.constant_(self.kappa_head.bias, float(kappa_bias))

    def forward(self, x):
        mu = self.backbone(x)
        if mu.shape[1] != 1:
            raise RuntimeError(
                'LearnedKappaHead expects backbone output [B,1,...], got {}'.format(
                    tuple(mu.shape)
                )
            )
        raw_kappa = self.kappa_head(mu)
        return torch.cat([mu, raw_kappa], dim=1)


class DTCWTComplexBackbone(nn.Module):
    """Real video -> learned DTCWT adapter -> complex SRDTrans -> real video."""

    def __init__(self, backbone: nn.Module, levels=3, image_channels=1,
                 channel_normalize=False, channel_scales=None,
                 biort='near_sym_b', qshift='qshift_b'):
        super().__init__()
        self.backbone = backbone
        self.representation = DTCWT2D(levels=levels, biort=biort, qshift=qshift)
        self.adapter = LearnedDTCWTScaleAdapter(
            image_channels=image_channels, levels=levels
        )
        self.channel_normalize = bool(channel_normalize)
        if self.channel_normalize:
            if channel_scales is None or len(channel_scales) != levels + 1:
                raise ValueError('fixed DTCWT normalization needs {} branch scales'.format(
                    levels + 1))
            scales = torch.as_tensor(channel_scales, dtype=torch.float32)
            if not bool(torch.isfinite(scales).all()) or bool((scales <= 0).any()):
                raise ValueError('fixed DTCWT branch scales must be finite and positive')
        else:
            scales = torch.ones(levels + 1, dtype=torch.float32)
        self.register_buffer('channel_scales', scales)

    def forward(self, x):
        if x.ndim != 5 or x.shape[1] != 1 or x.is_complex():
            raise ValueError(
                'expected real [B,1,T,H,W], got shape={} dtype={}'.format(
                    tuple(x.shape), x.dtype
                )
            )
        coefficients = self.representation(x)
        branches = (coefficients.low,) + coefficients.highs
        if self.channel_normalize:
            branches = tuple(
                value / self.channel_scales[index].to(value.dtype)
                for index, value in enumerate(branches)
            )
            coefficients = type(coefficients)(
                branches[0], tuple(branches[1:]), coefficients.spatial_size,
                coefficients.image_channels,
            )
        features = self.adapter.encode(coefficients)
        predicted = self.backbone(features)
        if predicted.shape != features.shape or not predicted.is_complex():
            raise RuntimeError('complex SRDTrans changed aligned DTCWT feature shape')
        predicted = self.adapter.decode(predicted, coefficients)
        if self.channel_normalize:
            predicted = type(predicted)(
                predicted.low * self.channel_scales[0].to(predicted.low.dtype),
                tuple(value * self.channel_scales[index + 1].to(value.dtype)
                      for index, value in enumerate(predicted.highs)),
                predicted.spatial_size, predicted.image_channels,
            )
        return self.representation.inverse(predicted)


class FourierComplexBackbone(nn.Module):
    """Real video -> complex steerable Fourier pyramid -> complex SRDTrans."""

    def __init__(self, backbone: nn.Module, image_size, image_channels=1,
                 channel_normalize=True, channel_scales=None):
        super().__init__()
        self.backbone = backbone
        self.representation = FourierPyramid2D(
            image_size=image_size, height=3, order=5,
            image_channels=image_channels,
        )
        self.adapter = LearnedFourierPyramidAdapter(
            image_channels=image_channels, height=3, orientations=6,
        )
        self.channel_normalize = bool(channel_normalize)
        if self.channel_normalize:
            if channel_scales is None or len(channel_scales) != 20:
                raise ValueError('Fourier normalization needs 20 coefficient scales')
            scales = torch.as_tensor(channel_scales, dtype=torch.float32)
            if not bool(torch.isfinite(scales).all()) or bool((scales <= 0).any()):
                raise ValueError('Fourier channel scales must be finite and positive')
        else:
            scales = torch.ones(20, dtype=torch.float32)
        self.register_buffer('channel_scales', scales)

    def _scale_branches(self, coefficients, inverse=False):
        branches = (coefficients.highpass,) + coefficients.bands + (coefficients.lowpass,)
        scaled = []
        offset = 0
        for value in branches:
            channels = value.shape[1]
            scale = self.channel_scales[offset:offset + channels].to(value.dtype)
            scale = scale.reshape(1, channels, 1, 1, 1)
            scaled.append(value * scale if inverse else value / scale)
            offset += channels
        return type(coefficients)(
            scaled[0], tuple(scaled[1:-1]), scaled[-1],
            coefficients.spatial_size, coefficients.image_channels,
        )

    def forward(self, x):
        if x.ndim != 5 or x.shape[1] != 1 or x.is_complex():
            raise ValueError(
                'expected real [B,1,T,H,W], got shape={} dtype={}'.format(
                    tuple(x.shape), x.dtype
                )
            )
        coefficients = self.representation(x)
        if self.channel_normalize:
            coefficients = self._scale_branches(coefficients)
        features = self.adapter.encode(coefficients)
        predicted = self.backbone(features)
        if predicted.shape != features.shape or not predicted.is_complex():
            raise RuntimeError('complex SRDTrans changed aligned Fourier feature shape')
        predicted = self.adapter.decode(predicted, coefficients)
        if self.channel_normalize:
            predicted = self._scale_branches(predicted, inverse=True)
        return self.representation.inverse(predicted)


def ensure_srdtrans_repo_on_path(srdtrans_root=None):
    root = os.path.abspath(srdtrans_root or DEFAULT_SRDTRANS_ROOT)
    if not os.path.isdir(root):
        raise ImportError(
            'SRDTrans path does not exist: {!r}. Expected {!r}.'.format(
                root, DEFAULT_SRDTRANS_ROOT)
        )
    if (os.path.basename(root) == 'SRDTrans_v2'
            and os.path.isfile(os.path.join(root, '__init__.py'))):
        sys_path_root = os.path.dirname(root)
    else:
        sys_path_root = root
    if sys_path_root not in sys.path:
        sys.path.insert(0, sys_path_root)
    return root, sys_path_root


def import_srdtrans_v2_class(srdtrans_root=None):
    ensure_srdtrans_repo_on_path(srdtrans_root)
    from SRDTrans_v2 import SRDTrans_v2
    return SRDTrans_v2


def _build_transformer_protocol_model(cfg, ModelClass, label):
    f_maps = getattr(cfg, 'srdtrans_f_maps', None) or [8, 16, 32, 64]
    kwargs = dict(
        img_dim=int(cfg.patch_x),
        img_time=int(cfg.patch_t),
        in_channel=1,
        embedding_dim=int(getattr(cfg, 'embedding_dim', 128)),
        num_heads=int(getattr(cfg, 'num_heads', 8)),
        hidden_dim=int(getattr(cfg, 'hidden_dim', 128 * 4)),
        window_size=int(getattr(cfg, 'window_size', 7)),
        num_transBlock=int(getattr(cfg, 'num_transBlock', 1)),
        attn_dropout_rate=float(getattr(cfg, 'attn_dropout_rate', 0.1)),
        f_maps=list(f_maps),
        input_dropout_rate=float(getattr(cfg, 'input_dropout_rate', 0.0)),
    )
    temporal_strides = getattr(cfg, 'temporal_strides', None)
    if temporal_strides is not None:
        kwargs['temporal_strides'] = list(temporal_strides)
    last_squeeze_op = getattr(cfg, 'last_squeeze_op', 'conv')
    if last_squeeze_op is not None:
        kwargs['last_squeeze_op'] = last_squeeze_op
    if label == 'SRDTrans':
        enc_d2 = getattr(cfg, 'enc_d2', None) or [1, 1, 1]
        kwargs['enc_d2'] = [int(x) for x in enc_d2]
        kwargs['trans_order'] = getattr(cfg, 'trans_order', 'ts') or 'ts'
        kwargs['upsample_mode'] = getattr(cfg, 'upsample_mode', 'convt') or 'convt'
        kwargs['freq_aware'] = bool(getattr(cfg, 'freq_aware', False))
        kwargs['ftvsr_enc1'] = bool(getattr(cfg, 'ftvsr_enc1', False))
    model = ModelClass(**kwargs)
    param_num = sum(p.numel() for p in model.parameters())
    print('\033[1;31mSRDTrans protocol / {} img_dim={} img_time={} '
          'params={:.2f}M\033[0m'.format(
              label, int(cfg.patch_x), int(cfg.patch_t), param_num / 1e6))
    return model


def _build_srdtrans_v2_protocol_model(cfg, ModelClass):
    f_maps = list(getattr(cfg, 'srdtrans_f_maps', None) or [8, 16, 32, 64])
    trans_order = getattr(cfg, 'trans_order', 'ts')
    space_post_norm = bool(getattr(cfg, 'space_post_norm', False))
    space_dropout_rate = float(getattr(cfg, 'space_dropout_rate', 0.0))
    use_msconv_before_trans = bool(getattr(cfg, 'use_msconv_before_trans', False))
    levels = int(getattr(cfg, 'dtcwt_levels', 3))
    coefficient_dim = int(cfg.patch_x) // 2
    coefficient_channels = 23 if getattr(cfg, 'representation', 'dtcwt') == 'steerable_fourier' else 2 + 6 * levels
    f_maps[0] = max(coefficient_channels, f_maps[0])
    for index in range(1, len(f_maps)):
        f_maps[index] = max(f_maps[index - 1], f_maps[index])

    model = ModelClass(
        img_dim=coefficient_dim,
        img_time=int(cfg.patch_t),
        in_channel=coefficient_channels,
        embedding_dim=int(getattr(cfg, 'embedding_dim', 128)),
        num_heads=int(getattr(cfg, 'num_heads', 8)),
        hidden_dim=int(getattr(cfg, 'hidden_dim', 128 * 4)),
        window_size=int(getattr(cfg, 'window_size', 7)),
        num_transBlock=int(getattr(cfg, 'num_transBlock', 1)),
        attn_dropout_rate=float(getattr(cfg, 'attn_dropout_rate', 0.1)),
        f_maps=list(f_maps),
        input_dropout_rate=float(getattr(cfg, 'input_dropout_rate', 0.0)),
        trans_order=trans_order,
        space_post_norm=space_post_norm,
        space_dropout_rate=space_dropout_rate,
        use_msconv_before_trans=use_msconv_before_trans,
        skip_fusion=getattr(cfg, 'skip_fusion', 'add'),
        interleaved_transformer=bool(getattr(cfg, 'interleaved_transformer', False)),
    )
    model.gradient_checkpointing = bool(
        getattr(cfg, 'gradient_checkpointing', True)
    )
    param_num = sum(p.numel() for p in model.parameters())
    print(
        '\033[1;31mSRDTrans protocol / SRDTrans_v2 img_dim={} img_time={} '
        'trans_order={} msconv={} space_post_norm={} checkpoint={} '
        'params={:.2f}M\033[0m'.format(
            coefficient_dim,
            int(cfg.patch_t),
            trans_order,
            use_msconv_before_trans,
            space_post_norm,
            model.gradient_checkpointing,
            param_num / 1e6,
        )
    )
    return model


def build_denoise_network_srdtrans(cfg):
    """Build a 3D U-Net or SRDTrans_v2 with img_dim=patch_x, img_time=patch_t."""
    if isinstance(cfg, dict):
        class _Cfg:
            pass
        obj = _Cfg()
        for key, value in cfg.items():
            setattr(obj, key, value)
        cfg = obj

    backbone = getattr(cfg, 'backbone', 'unet')
    if backbone in (None, '', 'unet', '3dunet', '3DUNet'):
        model = Network_3D_Unet(
            in_channels=1,
            out_channels=1,
            f_maps=int(getattr(cfg, 'fmap', 16)),
            final_sigmoid=True,
        )
        param_num = sum(p.numel() for p in model.parameters())
        print('\033[1;31mSRDTrans protocol / 3D U-Net params={:.2f}M\033[0m'.format(
            param_num / 1e6))
    elif backbone in ('srdtrans_v2', 'SRDTrans_v2'):
        SRDTrans_v2 = import_srdtrans_v2_class(getattr(cfg, 'srdtrans_root', None))
        representation = getattr(cfg, 'representation', 'dtcwt')
        dtcwt_dim = int(getattr(cfg, 'dtcwt_dim', 2))
        if representation not in ('dtcwt', 'steerable_fourier'):
            raise ValueError('unknown SRDTrans_v2 representation {!r}'.format(
                representation))
        # ponytail: 2D only; add a published differentiable 3D backend when requested.
        if dtcwt_dim != 2:
            raise NotImplementedError('Only --dtcwt_dim 2 is implemented')
        levels = int(getattr(cfg, 'dtcwt_levels', 3))
        backbone_model = _build_srdtrans_v2_protocol_model(cfg, SRDTrans_v2)
        if representation == 'dtcwt':
            model = DTCWTComplexBackbone(
                backbone_model,
                levels=levels,
                channel_normalize=bool(
                    getattr(cfg, 'dtcwt_channel_normalize', False)
                ),
                channel_scales=getattr(cfg, 'dtcwt_channel_scales', None),
                biort=getattr(cfg, 'dtcwt_biort', 'near_sym_b'),
                qshift=getattr(cfg, 'dtcwt_qshift', 'qshift_b'),
            )
        else:
            if levels != 3:
                raise ValueError('steerable_fourier currently requires --dtcwt_levels 3')
            model = FourierComplexBackbone(
                backbone_model, image_size=int(cfg.patch_x), image_channels=1,
                channel_normalize=bool(getattr(cfg, 'fourier_channel_normalize', True)),
                channel_scales=getattr(cfg, 'fourier_channel_scales', None),
            )
    else:
        raise ValueError(
            'SRDTrans protocol supports backbone unet or srdtrans_v2, '
            'got {!r}'.format(backbone)
        )

    kappa_mode = getattr(cfg, 'kappa_mode', 'fixed')
    if kappa_mode == 'learned_map':
        kappa_init = float(getattr(cfg, 'mpgn_kappa_init', 50.0))
        kappa_min = float(getattr(cfg, 'mpgn_kappa_min', 1e-4))
        kappa_bias = raw_kappa_bias_from_init(kappa_init, kappa_min)
        model = LearnedKappaHead(model, kappa_bias=kappa_bias)
        print(
            '\033[1;31mLearnedKappaHead enabled: kappa_init={}, kappa_min={}, '
            'raw_kappa_bias={:.6f} (kappa=exp(raw)+min)\033[0m'.format(
                kappa_init, kappa_min, kappa_bias
            )
        )
    elif kappa_mode not in ('fixed', None, ''):
        raise ValueError('Unknown kappa_mode: {!r}'.format(kappa_mode))

    return model
