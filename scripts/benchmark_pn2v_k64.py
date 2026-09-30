"""Minimal 0925-v2 versus PN2V-K64 CUDA benchmark."""
from types import SimpleNamespace
import time

import torch

from likelihood.backbone_factory import build_denoise_network_srdtrans
from likelihood.losses import mpgn_pn2v_marginal_nll


def config(mode):
    return SimpleNamespace(
        backbone='srdtrans_v2', representation='steerable_fourier',
        legacy_fourier_adapter=True, dtcwt_dim=2, dtcwt_levels=3,
        patch_x=64, patch_t=128, embedding_dim=128, num_heads=8,
        hidden_dim=384, window_size=7, num_transBlock=1,
        attn_dropout_rate=.1, input_dropout_rate=0.,
        srdtrans_f_maps=[24, 36, 48, 64], trans_order='st',
        space_post_norm=False, space_dropout_rate=0.,
        use_msconv_before_trans=False, skip_fusion='add',
        interleaved_transformer=True, space_attention='swin',
        compression_axis='time', gradient_checkpointing=False,
        prior_samples=64, mask_loss=mode,
        fourier_channel_normalize=True, fourier_channel_scales=[1.] * 20,
    )


def main():
    device = torch.device('cuda')
    for mode in ('l1l2', 'pn2v_pg'):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        model = build_denoise_network_srdtrans(config(mode)).to(device).train()
        optimizer = torch.optim.Adam(model.parameters(), 1e-4)
        x = torch.randn(1, 1, 128, 64, 64, device=device)
        y = torch.randn_like(x)
        mask = torch.zeros_like(x)
        mask[:, :, ::20] = 1

        for _ in range(20):
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = (((out - y) ** 2) * mask).sum() / mask.sum() if mode == 'l1l2' else \
                mpgn_pn2v_marginal_nll(out, y, mask, 0., 0., 5000., 1600.,
                                        kmax=512, quant_step=1.0)
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize(device)
        times = []
        for _ in range(100):
            optimizer.zero_grad(set_to_none=True)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = model(x)
            loss = (((out - y) ** 2) * mask).sum() / mask.sum() if mode == 'l1l2' else \
                mpgn_pn2v_marginal_nll(out, y, mask, 0., 0., 5000., 1600.,
                                        kmax=512, quant_step=1.0)
            loss.backward()
            optimizer.step()
            end.record()
            torch.cuda.synchronize(device)
            times.append(start.elapsed_time(end) / 1000.)
        print(mode, 'mean_s', sum(times) / len(times),
              'median_s', sorted(times)[len(times) // 2],
              'max_alloc_GiB', torch.cuda.max_memory_allocated(device) / 2**30,
              'max_reserved_GiB', torch.cuda.max_memory_reserved(device) / 2**30,
              'output_shape', tuple(out.shape))
        del model, optimizer, out, loss


if __name__ == '__main__':
    main()
