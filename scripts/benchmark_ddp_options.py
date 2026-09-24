"""Measure DDP reducer options on the real Fourier SRDTrans path."""

import os
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from likelihood.backbone_factory import build_denoise_network_srdtrans
from likelihood.losses import masked_l1_l2_loss


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.distributed.init_process_group('nccl', init_method='env://')
    torch.cuda.set_device(local_rank)
    with open(os.environ['DDP_TEST_CONFIG']) as stream:
        cfg = yaml.safe_load(stream)
    if os.environ.get('DDP_HIDDEN_DIM'):
        cfg['hidden_dim'] = int(os.environ['DDP_HIDDEN_DIM'])
    model = build_denoise_network_srdtrans(cfg).cuda(local_rank)
    if os.environ.get('DDP_COMPILE', '0') == '1':
        model.backbone = torch.compile(model.backbone, mode='reduce-overhead')
    tuned = os.environ.get('DDP_TUNED', '0') == '1'
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        gradient_as_bucket_view=tuned,
        static_graph=tuned,
        bucket_cap_mb=(float(os.environ['DDP_BUCKET_MB'])
                       if os.environ.get('DDP_BUCKET_MB') else None),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    generator = torch.Generator(device='cuda').manual_seed(1024 + rank)
    x = torch.randn((1, 1, 128, 128, 128), device='cuda', generator=generator)
    target = torch.randn_like(x)
    mask = torch.zeros_like(x)
    mask[..., ::20] = 1

    def step():
        optimizer.zero_grad(set_to_none=True)
        output = model(x)
        loss = masked_l1_l2_loss(output, target, mask)
        loss.backward()
        optimizer.step()
        return loss.detach()

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(20):
        step()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / 20
    loss = step().item()
    if rank == 0:
        print('tuned={} seconds_per_global_step={:.6f} peak_gib={:.3f} loss={:.6g}'.format(
            tuned, elapsed, torch.cuda.max_memory_allocated() / 2**30, loss), flush=True)
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
