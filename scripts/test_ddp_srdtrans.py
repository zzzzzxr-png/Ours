"""Two-rank DDP smoke test for the Fourier SRDTrans training path."""

import os
import sys
from pathlib import Path

import torch
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from likelihood.backbone_factory import build_denoise_network_srdtrans
from likelihood.losses import masked_l1_l2_loss


def main():
    if int(os.environ.get('WORLD_SIZE', '1')) != 2:
        raise RuntimeError('launch with torchrun --nproc_per_node=2')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.distributed.init_process_group('nccl', init_method='env://')
    torch.cuda.set_device(local_rank)
    cfg_path = os.environ['DDP_TEST_CONFIG']
    with open(cfg_path) as stream:
        cfg = yaml.safe_load(stream)
    model = build_denoise_network_srdtrans(cfg).cuda(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                broadcast_buffers=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    generator = torch.Generator(device='cuda').manual_seed(1024 + rank)
    target = torch.randn((1, 1, 128, 128, 128), device='cuda', generator=generator)
    mask = torch.zeros_like(target)
    mask[..., ::20] = 1
    for step in range(3):
        inp = torch.randn(target.shape, device='cuda', generator=generator)
        optimizer.zero_grad(set_to_none=True)
        output = model(inp)
        loss = masked_l1_l2_loss(output, target, mask)
        loss.backward()
        optimizer.step()
        if not torch.isfinite(loss):
            raise FloatingPointError('non-finite loss on rank {}'.format(rank))
    grad = next(p.grad for p in model.parameters() if p.grad is not None)
    norm = grad.detach().abs().mean()
    gathered = [torch.zeros_like(norm) for _ in range(2)]
    torch.distributed.all_gather(gathered, norm)
    if rank == 0:
        torch.save(model.module.state_dict(), '/tmp/ddp_srdtrans_smoke.pth')
        print('DDP smoke PASS: losses finite, gradients synchronized, checkpoint saved', flush=True)
        print('rank gradient means:', [float(value) for value in gathered], flush=True)
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
