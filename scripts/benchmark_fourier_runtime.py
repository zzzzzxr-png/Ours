"""Benchmark native cuDNN algorithm selection on the current Fourier model."""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from likelihood.backbone_factory import build_denoise_network_srdtrans
from likelihood.losses import masked_l1_l2_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--tf32-matmul', action='store_true')
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error('iterations must be positive')
    with open(args.config) as stream:
        cfg = yaml.safe_load(stream)
    torch.manual_seed(1024)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32_matmul
    torch.set_float32_matmul_precision('high' if args.tf32_matmul else 'highest')
    model = build_denoise_network_srdtrans(cfg).cuda()
    sample = torch.randn(1, 1, cfg['patch_t'], cfg['patch_x'], cfg['patch_y'], device='cuda') * 3000
    target = torch.randn_like(sample)
    mask = torch.zeros_like(sample)
    mask[..., ::20] = 1
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], betas=(cfg['b1'], cfg['b2']))
    print(json.dumps({'threads': torch.get_num_threads(), 'torch': torch.__version__,
                      'gpu': torch.cuda.get_device_name(),
                      'cudnn_tf32': torch.backends.cudnn.allow_tf32,
                      'matmul_tf32': torch.backends.cuda.matmul.allow_tf32}), flush=True)

    def step():
        optimizer.zero_grad(set_to_none=True)
        loss = masked_l1_l2_loss(model(sample), target, mask, l1_weight=0.5, l2_weight=0.5)
        loss.backward()
        optimizer.step()

    # Check outputs and parameter gradients before either timed training run.
    model.eval()
    results = []
    for enabled in (False, True):
        torch.backends.cudnn.benchmark = enabled
        optimizer.zero_grad(set_to_none=True)
        output = model(sample)
        output.square().mean().backward()
        results.append((output.detach().cpu(), [p.grad.detach().cpu().clone()
                       for p in model.parameters() if p.grad is not None]))
        del output
    def check(original, tuned):
        relative_error = (original - tuned).norm() / original.norm().clamp_min(1e-12)
        assert torch.isfinite(tuned).all()
        return float(relative_error)

    output_error = check(results[0][0], results[1][0])
    gradient_error = 0.0
    for original, tuned in zip(results[0][1], results[1][1]):
        gradient_error = max(gradient_error, check(original, tuned))
    del results
    print(f'Output/gradient relative L2 error: {output_error:.3g}/{gradient_error:.3g}', flush=True)
    model.train()
    for enabled in (False, True, False):
        torch.backends.cudnn.benchmark = enabled
        for _ in range(5):
            step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(args.iterations):
            step()
        torch.cuda.synchronize()
        print(json.dumps({'cudnn_benchmark': enabled, 'patches': args.iterations,
                          'train_seconds_per_patch': (time.perf_counter() - start) / args.iterations,
                          'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30}), flush=True)
    if args.profile:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as prof:
            step()
            torch.cuda.synchronize()
        print(prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=20))


if __name__ == '__main__':
    main()
