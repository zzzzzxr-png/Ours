"""Oracle 1/5/20 steerable-Fourier gain ceilings for frozen 0925 TIFFs."""
import argparse
import glob
import os

import numpy as np
import torch
from skimage import io

from representation.fourier_pyramid import FourierPyramid2D, FourierPyramidCoefficients


def component_images(pyr, coeffs):
    branches = (coeffs.highpass,) + tuple(coeffs.bands) + (coeffs.lowpass,)
    out = []
    for branch_index, branch in enumerate(branches):
        for channel in range(branch.shape[1]):
            selected = [torch.zeros_like(x) for x in branches]
            selected[branch_index][:, channel:channel + 1] = branch[:, channel:channel + 1]
            out.append(pyr.inverse(FourierPyramidCoefficients(
                highpass=selected[0], bands=tuple(selected[1:-1]),
                lowpass=selected[-1], spatial_size=coeffs.spatial_size,
                image_channels=coeffs.image_channels,
            )))
    return out


def stats(z, target, selector=None):
    if selector is not None:
        z, target = z[selector], target[selector]
    return z.sum(0), torch.einsum('nt,n->t', z, target)


def run(args):
    pattern = os.path.join(args.experiment, f'{args.freq}Hz', '*', '*E_50_Iter_6048.tif')
    candidates = glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError(pattern)
    pred_path = candidates[0]
    gt_path = os.path.join(args.dataset_root, f'{args.freq}Hz', f'{args.freq}Hz_GT',
                           f'clean_{args.freq}Hz_1000frames.tif')
    pred = io.imread(pred_path).astype(np.float32)
    gt = io.imread(gt_path).astype(np.float32)[50:350]
    if pred.shape != gt.shape:
        raise ValueError(f'shape mismatch: {pred.shape} vs {gt.shape}')
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    pyr = FourierPyramid2D(image_size=args.padded_size, height=3, order=5, image_channels=1).to(device)
    h, w = pred.shape[1:]
    pad_h, pad_w = args.padded_size - h, args.padded_size - w
    if pad_h < 0 or pad_w < 0:
        raise ValueError('padded_size is smaller than input')
    sums = {k: torch.zeros(k, k, device=device) for k in (1, 5, 20)}
    cross = {k: torch.zeros(k, device=device) for k in (1, 5, 20)}
    sums_even = {k: torch.zeros(k, k, device=device) for k in (1, 5, 20)}
    cross_even = {k: torch.zeros(k, device=device) for k in (1, 5, 20)}
    sums_odd = {k: torch.zeros(k, k, device=device) for k in (1, 5, 20)}
    cross_odd = {k: torch.zeros(k, device=device) for k in (1, 5, 20)}
    components = []
    with torch.no_grad():
        for start in range(0, pred.shape[0], args.chunk):
            stop = min(start + args.chunk, pred.shape[0])
            x = torch.from_numpy(pred[start:stop]).to(device).unsqueeze(0).unsqueeze(0)
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h, 0, 0), mode='reflect')
            coeffs = pyr(x)
            parts = [v[:, 0, :, :h, :w] for v in component_images(pyr, coeffs)]
            z20 = torch.stack(parts, dim=-1).reshape(-1, 20)
            y = torch.from_numpy(gt[start:stop]).to(device).reshape(-1)
            groups = [[0], list(range(1, 7)), list(range(7, 13)), list(range(13, 19)), [19]]
            for k, ids in ((1, None), (5, groups), (20, list(range(20)))):
                if k == 1:
                    zz = z20.sum(dim=1, keepdim=True)
                elif k == 5:
                    zz = torch.stack([z20[:, group].sum(dim=1) for group in ids], dim=1)
                else:
                    zz = z20[:, ids]
                sums[k] += zz.T @ zz
                cross[k] += zz.T @ y
                even = (torch.arange(stop - start, device=device)[:, None, None]
                        .expand(stop - start, h, w).reshape(-1) + start) % 2 == 0
                sums_even[k] += zz[even].T @ zz[even]
                cross_even[k] += zz[even].T @ y[even]
                sums_odd[k] += zz[~even].T @ zz[~even]
                cross_odd[k] += zz[~even].T @ y[~even]
            components.append((start, stop, z20.cpu()))
    gains = {k: torch.linalg.lstsq(sums[k], cross[k]).solution for k in (1, 5, 20)}
    ge = {k: torch.linalg.lstsq(sums_even[k], cross_even[k]).solution for k in (1, 5, 20)}
    go = {k: torch.linalg.lstsq(sums_odd[k], cross_odd[k]).solution for k in (1, 5, 20)}
    gt_flat = torch.from_numpy(gt.reshape(-1)).to(device)
    pred_flat = torch.from_numpy(pred.reshape(-1)).to(device)
    out = []
    gt_mean = gt_flat.mean()
    pred_mean = pred_flat.mean()
    base_mse = torch.mean((pred_flat - gt_flat) ** 2)
    out.append(f'{args.freq} Hz pred={os.path.basename(pred_path)} base_snr={10*torch.log10(torch.mean(gt_flat**2)/base_mse).item():.5f} centered={args.center}')
    groups = [[0], list(range(1, 7)), list(range(7, 13)), list(range(13, 19)), [19]]
    for k, ids in ((1, None), (5, groups), (20, list(range(20)))):
        z = torch.cat([v for _, _, v in components], dim=0).to(device)
        if k == 1:
            zz = z.sum(dim=1, keepdim=True)
        elif k == 5:
            zz = torch.stack([z[:, group].sum(dim=1) for group in ids], dim=1)
        else:
            zz = z[:, ids]
        if args.center:
            zz = zz - zz.mean(dim=0, keepdim=True)
            gains[k] = torch.linalg.lstsq(zz.T @ zz, zz.T @ (gt_flat - gt_mean)).solution
            corrected = gt_mean + zz @ gains[k]
        else:
            corrected = zz @ gains[k]
        mse = torch.mean((corrected - gt_flat) ** 2)
        snr = 10 * torch.log10(torch.mean(gt_flat ** 2) / mse)
        if args.center:
            cross_snr = torch.tensor(float('nan'), device=device)
        else:
            ce = (zz[::2] @ ge[k]); co = (zz[1::2] @ go[k])
            cross_mse = 0.5 * (torch.mean((ce - gt_flat[::2]) ** 2) + torch.mean((co - gt_flat[1::2]) ** 2))
            cross_snr = 10 * torch.log10(torch.mean(gt_flat ** 2) / cross_mse)
        out.append(f'  K={k:2d} gains={gains[k].detach().cpu().numpy().round(6).tolist()} full_snr={snr.item():.5f} cross_snr={cross_snr.item():.5f}')
    print('\n'.join(out), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--freq', required=True)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--chunk', type=int, default=8)
    p.add_argument('--padded-size', type=int, default=256)
    p.add_argument('--center', action='store_true', help='fit gains after centering every component')
    p.add_argument('--experiment', default='experiments/260925_steerable_fourier_add_interleaved_width24_36_48_64_head8_hidden384_patch64_pure_l1l2_lr1e-4_v2')
    p.add_argument('--dataset-root', default='/data/zhouxirou/All_Datasets/dataset_CI_syn_Q004_a5000_b1600/dataset_zxr_all_frequency_Q004_beta1600-1-T1000H245W245')
    run(p.parse_args())
