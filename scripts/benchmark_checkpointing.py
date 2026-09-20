"""Measure SRDTrans training-step time and peak memory with/without checkpointing."""

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from likelihood.backbone_factory import build_denoise_network_srdtrans  # noqa: E402


def run(model, sample, enabled, warmup=2, iterations=10):
    model.backbone.gradient_checkpointing = enabled
    model.train()

    def step():
        model.zero_grad(set_to_none=True)
        output = model(sample)
        output.square().mean().backward()

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(iterations):
        step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed / iterations, torch.cuda.max_memory_allocated() / 2**30


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    torch.manual_seed(1024)
    cfg = SimpleNamespace(
        backbone='srdtrans_v2', patch_x=128, patch_t=128,
        srdtrans_root=str(ROOT / 'prior' / 'srdtrans' / 'SRDTrans_v2'),
        srdtrans_f_maps=[32, 64, 128, 256], embedding_dim=128, num_heads=8,
        hidden_dim=512, window_size=7, num_transBlock=1,
        attn_dropout_rate=0.1, input_dropout_rate=0.0,
        trans_order='st', space_post_norm=False, space_dropout_rate=0.0,
        use_msconv_before_trans=False, kappa_mode='fixed', mpgn_kappa=10.0,
        representation='dtcwt', dtcwt_dim=2, dtcwt_levels=3,
        dtcwt_channel_normalize=False,
    )
    model = build_denoise_network_srdtrans(cfg).cuda()
    sample = torch.randn(1, 1, 128, 128, 128, device='cuda') * 3000
    for enabled in (True, False):
        seconds, gib = run(model, sample, enabled)
        print('checkpoint={} seconds_per_step={:.4f} peak_memory_gib={:.3f}'.format(
            enabled, seconds, gib))


if __name__ == '__main__':
    main()
