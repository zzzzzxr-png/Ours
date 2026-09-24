#!/usr/bin/env python3
"""Compare GT/noisy/SRDTrans/Ours in common Fourier subbands."""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage import io

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from representation import FourierPyramid2D


DATA = Path('/data/zhouxirou/All_Datasets/dataset_CI_syn_Q004_a5000_b1600/'
            'dataset_zxr_all_frequency_Q004_beta1600-1-T1000H245W245/30Hz')
PATHS = {
    'GT': DATA / '30Hz_GT/clean_30Hz_1000frames.tif',
    'Noisy': DATA / '30Hz_Noisy/noise_30Hz_4RPN_-0.32dBSNR.tif',
    'SRDTrans': Path('/data/zhouxirou/All_Related_Work/DeepCAD-RT/DeepCAD_RT_pytorch/'
                    '260620_experiments_srdprotocol_srdtrans_T1000H245W245/30Hz_Noisy_srdtrans_spatial_srdtrans/'
                    'noise_30Hz_4RPN_-0.32dBSNR_E_20_Iter_3492.tif'),
    'Ours': Path('/data/zhouxirou/Ours_260911/experiments/'
                 '260920_steerable_fourier_complex_srdtrans_fullres_norm_ckpt_patch64_pure_l1l2_lr5e-5/'
                 '30Hz/30Hz_Noisy_srdtrans_temporal_mask_srdtrans_v2/'
                 'noise_30Hz_4RPN_-0.32dBSNR_E_62_Iter_6048.tif'),
}
OUT = Path('/data/zhouxirou/Ours_260911/experiments/representation_export/'
           '30Hz_subband_magnitude_comparison_245')
# Full-volume screening selected scale0/3 and scale1/3; scale2/2 is an honest control.
SELECTED = [('highpass', 0), ('scale0', 3), ('scale1', 3), ('scale2', 2)]
FRAME = 50  # 51st frame, zero-based index


def load_patch(path, frame=0):
    image = np.asarray(io.imread(path), dtype=np.float32)
    if image.ndim != 3 or frame >= image.shape[0]:
        raise ValueError(f'invalid frame range for {path}: {image.shape}')
    image = image[frame]
    if image.shape != (245, 245):
        raise ValueError(f'expected full 245x245 frame, got {image.shape}')
    return image


def extract(transform, image, device):
    x = torch.from_numpy(image)[None, None, None].to(device)
    with torch.inference_mode():
        c = transform(x)
    values = {'highpass': c.highpass[0, 0, 0], 'lowpass': c.lowpass[0, 0, 0]}
    for scale, band in enumerate(c.bands):
        for channel in range(6):
            values[f'scale{scale}_{channel}'] = band[0, channel, 0]
    return values


def main():
    device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
    OUT.mkdir(parents=True, exist_ok=True)
    images = {name: load_patch(path, frame=FRAME) for name, path in PATHS.items()}
    transform = FourierPyramid2D(245, height=3, order=5).to(device).eval()
    coeffs = {name: extract(transform, image, device) for name, image in images.items()}
    metrics = {}
    fig, axes = plt.subplots(len(SELECTED), 6, figsize=(18, 15), squeeze=False)
    columns = ['GT', 'Noisy', 'SRDTrans', 'Ours', '|SRDTrans−GT|', '|Ours−GT|']
    for col, title in enumerate(columns):
        axes[0, col].set_title(title, fontsize=12, weight='bold')
    for row, (branch, channel) in enumerate(SELECTED):
        key = branch if branch in ('highpass', 'lowpass') else f'{branch}_{channel}'
        gt = coeffs['GT'][key]
        noisy = coeffs['Noisy'][key]
        srd = coeffs['SRDTrans'][key]
        ours = coeffs['Ours'][key]
        magnitudes = [z.abs().cpu().numpy() for z in (gt, noisy, srd, ours)]
        vmax = max(float(np.percentile(z, 99.5)) for z in magnitudes)
        gt_energy = gt.abs().square().mean().sqrt().clamp_min(1e-12)
        noisy_error_rms = (noisy.abs() - gt.abs()).abs().square().mean().sqrt().clamp_min(1e-12)
        errors = [((srd.abs() - gt.abs()).abs() / noisy_error_rms * 100).cpu().numpy(),
                  ((ours.abs() - gt.abs()).abs() / noisy_error_rms * 100).cpu().numpy()]
        err_vmax = max(float(np.percentile(z, 99.5)) for z in errors)
        rel_srd = float((srd.abs() - gt.abs()).abs().square().mean().sqrt() / noisy_error_rms * 100)
        rel_ours = float((ours.abs() - gt.abs()).abs().square().mean().sqrt() / noisy_error_rms * 100)
        metrics[key] = {'noisy_baseline_percent': 100.0,
                        'gt_percent': 0.0,
                        'normalized_magnitude_error_percent_srdtrans': rel_srd,
                        'normalized_magnitude_error_percent_ours': rel_ours,
                        'energy_ratio_srdtrans': float(srd.abs().square().mean().sqrt() / gt_energy),
                        'energy_ratio_ours': float(ours.abs().square().mean().sqrt() / gt_energy),
                        'ours_better': rel_ours < rel_srd}
        for col, image in enumerate(magnitudes):
            axes[row, col].imshow(image, cmap='magma', vmin=0, vmax=max(vmax, 1e-12))
        for col, image in enumerate(errors, start=4):
            axes[row, col].imshow(image, cmap='magma', vmin=0, vmax=max(err_vmax, 1e-12))
        axes[row, 0].set_ylabel(
            f'{branch} c{channel}\nNormErr S/O: {rel_srd:.1f}%/{rel_ours:.1f}%', fontsize=9)
        for col in range(6):
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
    fig.suptitle(f'30 Hz Fourier subband magnitude comparison | frame {FRAME + 1} | full 245×245 | Noisy=100%, GT=0%', fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / 'subband_comparison_frame051.png', dpi=180)
    plt.close(fig)
    (OUT / 'subband_metrics_frame051.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps({'output': str(OUT), 'metrics': metrics}, indent=2))


if __name__ == '__main__':
    main()
