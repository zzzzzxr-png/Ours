#!/usr/bin/env python3
"""Export raw complex steerable-Fourier coefficients for a 30 Hz patch."""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from skimage import io

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from representation import FourierPyramid2D


DATA = Path('/data/zhouxirou/All_Datasets/dataset_CI_syn_Q004_a5000_b1600/'
            'dataset_zxr_all_frequency_Q004_beta1600-1-T1000H245W245/30Hz')
OUT = Path('/data/zhouxirou/Ours_260911/experiments/representation_export/'
           '30Hz_fourier_51frames_patch64')


def load_patch(path):
    frames = np.asarray(io.imread(path), dtype=np.float32)[:51]
    top = (frames.shape[1] - 64) // 2
    left = (frames.shape[2] - 64) // 2
    patch = frames[:, top:top + 64, left:left + 64]
    if patch.shape != (51, 64, 64):
        raise ValueError(f'expected (51,64,64), got {patch.shape}')
    return torch.from_numpy(patch)[None, None]


def export_one(name, path, transform, device):
    x = load_patch(path).to(device)
    with torch.inference_mode():
        coeffs = transform(x)
    branches = [('highpass', coeffs.highpass), *
                [(f'scale{idx}', value) for idx, value in enumerate(coeffs.bands)],
                ('lowpass', coeffs.lowpass)]
    manifest = {'input': str(path), 'frames': 51, 'spatial_size': [64, 64],
                'normalization': 'none', 'branches': {}}
    for branch, value in branches:
        value = value[0].permute(1, 0, 2, 3).cpu().numpy()  # [T,C,H,W]
        branch_info = {'shape': list(value.shape), 'channels': {}}
        for channel in range(value.shape[1]):
            stem = f'{name}_{branch}_c{channel:02d}'
            np.save(OUT / f'{stem}_real.npy', value[:, channel].real.astype(np.float32))
            np.save(OUT / f'{stem}_imag.npy', value[:, channel].imag.astype(np.float32))
            branch_info['channels'][str(channel)] = {
                'real': f'{stem}_real.npy', 'imag': f'{stem}_imag.npy'
            }
        manifest['branches'][branch] = branch_info
    return manifest


def main():
    device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
    OUT.mkdir(parents=True, exist_ok=True)
    transform = FourierPyramid2D(image_size=64, height=3, order=5).to(device).eval()
    manifests = {}
    for name, path in (
        ('gt', DATA / '30Hz_GT/clean_30Hz_1000frames.tif'),
        ('noisy', DATA / '30Hz_Noisy/noise_30Hz_4RPN_-0.32dBSNR.tif'),
    ):
        manifests[name] = export_one(name, path, transform, device)
    (OUT / 'manifest.json').write_text(json.dumps(manifests, indent=2))
    print(json.dumps({'output': str(OUT), 'device': str(device), 'manifest': 'manifest.json'}, indent=2))


if __name__ == '__main__':
    main()
