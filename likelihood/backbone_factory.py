"""Model factory for the SRDTrans-aligned (likelihood) training protocol."""

import math
import os
import sys

import torch
import torch.nn as nn

from prior.deepcadrt import Network_3D_Unet
from posterior.analytic_representation import analytic_representation, inverse_candidates

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


class AnalyticComplexBackbone(nn.Module):
    """Axis-Hilbert input -> complex latent -> structured analytic readout."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        # Keep the decoder's real/imaginary semantics intact: no learned
        # projection may mix the six latent fields.
        last_complex_conv = None
        for module in self.backbone.modules():
            if (module.__class__.__name__ == 'Conv3d'
                    and module.__class__.__module__.startswith('complextorch')):
                last_complex_conv = module
        if last_complex_conv is not None:
            with torch.no_grad():
                torch.nn.init.normal_(last_complex_conv.conv.weight, mean=0.0, std=0.01)
                if last_complex_conv.conv.bias is not None:
                    last_complex_conv.conv.bias.zero_()

    def forward_with_complex(self, x):
        if x.ndim != 5 or x.shape[1] != 1 or x.is_complex():
            raise ValueError(
                'expected real [B,1,T,H,W], got shape={} dtype={}'.format(
                    tuple(x.shape), x.dtype
                )
            )
        latent_complex = self.backbone(analytic_representation(x))
        if latent_complex.shape[1] != 3 or not latent_complex.is_complex():
            raise RuntimeError(
                'complex SRDTrans must return complex [B,3,T,H,W], got '
                'shape={} dtype={}'.format(
                    tuple(latent_complex.shape), latent_complex.dtype
                )
            )
        shared_real = latent_complex.real.mean(dim=1, keepdim=True)
        structured_output = torch.cat(
            [
                torch.complex(shared_real, latent_complex.imag[:, index:index + 1])
                for index in range(3)
            ],
            dim=1,
        )
        return inverse_candidates(structured_output), structured_output

    def forward(self, x, return_complex=False):
        candidates, complex_output = self.forward_with_complex(x)
        if return_complex:
            return candidates, complex_output
        return candidates


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
    f_maps = getattr(cfg, 'srdtrans_f_maps', None) or [8, 16, 32, 64]
    trans_order = getattr(cfg, 'trans_order', 'ts')
    space_post_norm = bool(getattr(cfg, 'space_post_norm', False))
    space_dropout_rate = float(getattr(cfg, 'space_dropout_rate', 0.0))
    use_msconv_before_trans = bool(getattr(cfg, 'use_msconv_before_trans', False))

    model = ModelClass(
        img_dim=int(cfg.patch_x),
        img_time=int(cfg.patch_t),
        in_channel=3,
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
    )
    param_num = sum(p.numel() for p in model.parameters())
    print(
        '\033[1;31mSRDTrans protocol / SRDTrans_v2 img_dim={} img_time={} '
        'trans_order={} msconv={} space_post_norm={} params={:.2f}M\033[0m'.format(
            int(cfg.patch_x),
            int(cfg.patch_t),
            trans_order,
            use_msconv_before_trans,
            space_post_norm,
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
        model = AnalyticComplexBackbone(
            _build_srdtrans_v2_protocol_model(cfg, SRDTrans_v2)
        )
    else:
        raise ValueError(
            'SRDTrans protocol supports backbone unet or srdtrans_v2, '
            'got {!r}'.format(backbone)
        )

    kappa_mode = getattr(cfg, 'kappa_mode', 'fixed')
    if isinstance(model, AnalyticComplexBackbone) and kappa_mode == 'learned_map':
        raise ValueError(
            'analytic complex SRDTrans currently requires fixed kappa; '
            'reuse the selected --mpgn_kappa value'
        )
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
