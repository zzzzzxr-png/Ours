"""Build denoise backbones for the posterior (unroll-transformer) pipeline."""

import os
import sys

from .unroll_network import (
    Network_SRDTrans_Unroll_Transformer,
)
from prior.deepcadrt.unet3d import Network_3D_Unet

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEFAULT_SRDTRANS_ROOT = os.path.join(_PROJECT_ROOT, 'prior', 'srdtrans', 'SRDTrans_v2')

TRANSFORMER_BACKBONES = frozenset({
    'srdtrans_v2', 'SRDTrans_v2',
    'srdtrans_unroll_transformer',
})


def _srdtrans_sys_path(root):
    """Directory to prepend to sys.path for importing SRDTrans_v2."""
    root = os.path.abspath(root)
    if (os.path.basename(root) == 'SRDTrans_v2'
            and os.path.isfile(os.path.join(root, '__init__.py'))):
        return os.path.dirname(root)
    return root


def import_srdtrans_v2_class(srdtrans_root=None):
    """Import SRDTrans_v2 from the configured checkout."""
    return _import_srdtrans_module(
        srdtrans_root,
        module_path='SRDTrans_v2',
        class_name='SRDTrans_v2',
    )


def _import_srdtrans_module(srdtrans_root, module_path, class_name):
    root = os.path.abspath(srdtrans_root or DEFAULT_SRDTRANS_ROOT)
    if not os.path.isdir(root):
        raise ImportError(
            'SRDTrans path does not exist: {!r}. '
            'Expected {!r} or its parent repo root.'.format(
                root, DEFAULT_SRDTRANS_ROOT)
        )

    sys_path_root = _srdtrans_sys_path(root)
    if sys_path_root not in sys.path:
        sys.path.insert(0, sys_path_root)
    try:
        module = __import__(module_path, fromlist=[class_name])
        model_cls = getattr(module, class_name)
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            'Cannot import {} from {!r} (sys.path root: {!r}). '
            'Last error: {!r}'.format(
                class_name, root, sys_path_root, exc)
        ) from exc
    return model_cls, root


def get_srdtrans_input_shape(sampling_mode, patch_x, patch_y, patch_t):
    """Return (img_dim, img_time) expected by transformer backbones."""
    if int(patch_x) != int(patch_y):
        raise ValueError(
            'Transformer backbone requires patch_x == patch_y, got {} vs {}'.format(
                patch_x, patch_y)
        )

    mode = sampling_mode or 'temporal'
    patch_x = int(patch_x)
    patch_t = int(patch_t)

    if mode == 'spatial_ori':
        return patch_x // 2, patch_t
    return patch_x, patch_t


def _build_transformer_unet(cfg, ModelClass, label, srdtrans_root):
    img_dim, img_time = get_srdtrans_input_shape(
        getattr(cfg, 'sampling_mode', 'temporal'),
        cfg.patch_x,
        cfg.patch_y,
        cfg.patch_t,
    )
    f_maps = getattr(cfg, 'srdtrans_f_maps', None)
    if f_maps is None:
        f_maps = [8, 16, 32, 64]

    model = ModelClass(
        img_dim=img_dim,
        img_time=img_time,
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
    param_num = sum(p.numel() for p in model.parameters())
    print('\033[1;31m{} backbone ({}) img_dim={} img_time={} root={} '
          'params={:.2f}M\033[0m'.format(
              label,
              getattr(cfg, 'sampling_mode', 'temporal'),
              img_dim, img_time, srdtrans_root, param_num / 1e6))
    return model


def build_denoise_network(cfg):
    """
    Build a denoise network from a training/testing config object or dict.

    Supported cfg.backbone:
      - 'unet' (default): 3D U-Net wrapper
      - 'unet_unroll': K-step unrolled 3D U-Net + shifted-Poisson correction
      - 'srdtrans_v2': SRDTrans_v2 transformer U-Net hybrid
    """
    if isinstance(cfg, dict):
        class _Cfg:
            pass
        obj = _Cfg()
        for k, v in cfg.items():
            setattr(obj, k, v)
        cfg = obj

    backbone = getattr(cfg, 'backbone', 'unet')

    if backbone == 'srdtrans_unroll_transformer':
        return Network_SRDTrans_Unroll_Transformer(
            prior_backbone=getattr(
                cfg,
                'prior_backbone',
                'srdtrans_v2',
            ),
            unet_f_maps=int(
                getattr(cfg, 'fmap', 16)
            ),

            srdtrans_root=cfg.srdtrans_root,

            img_dim=cfg.patch_x,
            img_time=cfg.patch_t,

            embedding_dim=cfg.embedding_dim,
            num_heads=cfg.num_heads,
            hidden_dim=cfg.hidden_dim,
            window_size=cfg.window_size,
            num_transBlock=cfg.num_transBlock,
            attn_dropout_rate=cfg.attn_dropout_rate,
            f_maps=cfg.srdtrans_f_maps,
            input_dropout_rate=cfg.input_dropout_rate,

            unroll_steps=cfg.unroll_steps,

            mpgn_alpha=cfg.mpgn_alpha,
            mpgn_beta=cfg.mpgn_beta,
            mpgn_offset=cfg.mpgn_offset,
            mpgn_kmax=cfg.mpgn_kmax,

            unroll_rho_sched_min=getattr(
                cfg,
                'unroll_rho_sched_min',
                getattr(cfg, 'unroll_mu_min', 0.1),
            ),
            unroll_rho_sched_max=getattr(
                cfg,
                'unroll_rho_sched_max',
                getattr(cfg, 'unroll_mu_max', 100.0),
            ),

            unroll_rho_min=cfg.unroll_rho_min,
            unroll_eps=cfg.unroll_eps,
            unroll_corr_n_iter=getattr(cfg, 'unroll_corr_n_iter', 2),
            unroll_corr_chunk_t=getattr(
                cfg, 'unroll_corr_chunk_t', cfg.mpgn_nll_chunk_t
            ),
            unroll_corr_step_cap_sigma=getattr(
                cfg, 'unroll_corr_step_cap_sigma', 1.0
            ),
            unroll_gradient_checkpointing=(
                cfg.unroll_gradient_checkpointing
            ),
        )

    if backbone in (None, '', 'unet', '3dunet', '3DUNet'):
        return Network_3D_Unet(
            in_channels=1,
            out_channels=1,
            f_maps=cfg.fmap,
            final_sigmoid=True,
        )

    if backbone in ('unet_unroll', '3dunet_unroll', '3DUNet_unroll'):
        from .model_3DUnet_unroll import Network_3D_Unet_Unroll
        model = Network_3D_Unet_Unroll(
            UNet_type='3DUNet',
            in_channels=1,
            out_channels=1,
            f_maps=cfg.fmap,
            final_sigmoid=True,
            unroll_steps=int(getattr(cfg, 'unroll_steps', 8)),
            mpgn_alpha=float(getattr(cfg, 'mpgn_alpha', 5000.0)),
            mpgn_beta=float(getattr(cfg, 'mpgn_beta', 1600.0)),
            mpgn_offset=float(getattr(cfg, 'mpgn_offset', 0.0)),
            unroll_data_weight_scale=float(
                getattr(cfg, 'unroll_data_weight_scale', 0.0)
            ),
            unroll_data_multiplier_init=float(
                getattr(cfg, 'unroll_data_multiplier_init', 1.0)
            ),
            unroll_prior_multiplier_init=float(
                getattr(cfg, 'unroll_prior_multiplier_init', 1.0)
            ),
            unroll_hypa_hidden=int(getattr(cfg, 'unroll_hypa_hidden', 64)),
            unroll_rho_min=float(getattr(cfg, 'unroll_rho_min', 1e-6)),
            unroll_eps=float(getattr(cfg, 'unroll_eps', 1e-6)),
        )
        param_num = sum(p.numel() for p in model.parameters())
        print('\033[1;31mUnrolled 3D U-Net (K={}) params={:.2f}M\033[0m'.format(
            int(getattr(cfg, 'unroll_steps', 8)), param_num / 1e6))
        return model

    if backbone in ('srdtrans_v2', 'SRDTrans_v2'):
        SRDTrans_v2, srdtrans_root = import_srdtrans_v2_class(
            getattr(cfg, 'srdtrans_root', None)
        )
        return _build_transformer_unet(
            cfg, SRDTrans_v2, 'SRDTrans_v2', srdtrans_root)

    raise ValueError('Unknown backbone: {!r}'.format(backbone))
