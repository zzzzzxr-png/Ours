"""CPU smoke/regression checks for the structured Fourier representation."""

import torch

from likelihood.backbone_factory import build_denoise_network_srdtrans
from likelihood.backbone_factory import FourierComplexBackbone
from representation.fourier_pyramid import FourierPyramid2D


def config(representation, checkpoint=False, patch_x=64):
    return dict(
        backbone='srdtrans_v2', representation=representation, dtcwt_dim=2,
        dtcwt_levels=3, patch_x=patch_x, patch_t=16, srdtrans_f_maps=[16] * 4,
        embedding_dim=32, num_heads=4, hidden_dim=64, window_size=7,
        num_transBlock=1, attn_dropout_rate=0.0, input_dropout_rate=0.0,
        trans_order='st', space_post_norm=False, space_dropout_rate=0.0,
        use_msconv_before_trans=False, skip_fusion='add',
        interleaved_transformer=True, space_attention='swin',
        gradient_checkpointing=checkpoint, checkpoint_transformer_only=False,
        fourier_channel_normalize=False, orientation_heads=4,
        srdtrans_root='prior/srdtrans/SRDTrans_v2',
    )


def run_structured(checkpoint):
    model = build_denoise_network_srdtrans(
        config('steerable_fourier_structured', checkpoint=checkpoint)
    ).train()
    x = torch.randn(1, 1, 16, 64, 64, requires_grad=True)
    y = model(x)
    assert y.shape == x.shape and not y.is_complex()
    assert torch.isfinite(y).all()
    y.square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all()
               for p in model.parameters())
    params = dict(model.named_parameters())
    required = (
        'directional.0.proj.conv.weight',
        'orientation.0.attn.w_q.linear.weight',
        'orientation.0.orientation_bias',
        'inject.1.attn.w_q.linear.weight',
        'backbone.layers.0.space_regular.transformer.blocks.0.attn.qkv.linear.weight',
        'head.scale0.conv.weight',
    )
    for name in required:
        assert params[name].grad is not None, name
    return model


def main():
    torch.set_num_threads(2)
    coeffs = FourierPyramid2D(64)(torch.randn(1, 1, 2, 64, 64))
    assert coeffs.highpass.shape == (1, 1, 2, 64, 64)
    assert [tuple(x.shape) for x in coeffs.bands] == [
        (1, 6, 2, 64, 64), (1, 6, 2, 32, 32), (1, 6, 2, 16, 16)]
    assert coeffs.lowpass.shape == (1, 1, 2, 8, 8)

    structured = run_structured(False)
    checkpointed = run_structured(True)
    assert structured.backbone.gradient_checkpointing is False
    assert checkpointed.backbone.gradient_checkpointing is True

    normalized_cfg = config('steerable_fourier_structured')
    normalized_cfg.update(fourier_channel_normalize=True, fourier_channel_scales=[1.0] * 20)
    normalized = build_denoise_network_srdtrans(normalized_cfg).eval()
    with torch.no_grad():
        y = normalized(torch.randn(1, 1, 16, 64, 64))
    assert y.shape == (1, 1, 16, 64, 64) and torch.isfinite(y).all()

    # A second legal pyramid size guards against patch-size-specific reshapes.
    size96 = build_denoise_network_srdtrans(
        config('steerable_fourier_structured', patch_x=96)
    ).eval()
    with torch.no_grad():
        y96 = size96(torch.randn(1, 1, 16, 96, 96))
    assert y96.shape == (1, 1, 16, 96, 96) and torch.isfinite(y96).all()

    # The old mode still builds the old wrapper and does not acquire structured keys.
    baseline = build_denoise_network_srdtrans(config('steerable_fourier'))
    assert isinstance(baseline, FourierComplexBackbone)
    assert not any(key.startswith(('directional.', 'orientation.', 'inject.'))
                   for key in baseline.state_dict())
    print('structured Fourier forward/backward, checkpoint on/off, and old-path checks: PASS')


if __name__ == '__main__':
    main()
