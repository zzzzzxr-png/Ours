"""Compare DTCWT and complex steerable-pyramid coefficients on X and X+N."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from skimage import io

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from representation import DTCWT2D, FourierPyramid2D


def load_patch(clean_path, noisy_path, frames, size):
    clean = np.asarray(io.imread(clean_path), dtype=np.float32)
    noisy = np.asarray(io.imread(noisy_path), dtype=np.float32)
    if clean.shape != noisy.shape or clean.ndim != 3:
        raise ValueError(f'clean/noisy shape mismatch: {clean.shape} vs {noisy.shape}')
    frames = min(int(frames), clean.shape[0])
    top = max((clean.shape[1] - size) // 2, 0)
    left = max((clean.shape[2] - size) // 2, 0)
    clean = clean[:frames, top:top + size, left:left + size]
    noisy = noisy[:frames, top:top + size, left:left + size]
    if clean.shape[1:] != (size, size):
        raise ValueError(f'patch is {clean.shape}, expected spatial size {size}')
    return (torch.from_numpy(clean).permute(1, 0, 2).unsqueeze(0).unsqueeze(0),
            torch.from_numpy(noisy).permute(1, 0, 2).unsqueeze(0).unsqueeze(0))


def branches(coeffs, kind):
    if kind == 'dtcwt':
        return [('low', coeffs.low)] + [(f'band{idx}', value) for idx, value in enumerate(coeffs.highs)]
    return [('highpass', coeffs.highpass)] + [
        (f'band{idx}', value) for idx, value in enumerate(coeffs.bands)
    ] + [('lowpass', coeffs.lowpass)]


def scalar_metrics(clean, noisy, energy_percentages):
    signal = clean.abs().square()
    noise = (noisy - clean).abs().square()
    snr = 10.0 * torch.log10(signal.sum() / noise.sum().clamp_min(1e-30))
    flat = signal.flatten()
    sorted_energy = torch.sort(flat, descending=True).values
    total = sorted_energy.sum().clamp_min(1e-30)
    concentration = {}
    for percentage in energy_percentages:
        count = max(1, int(np.ceil(float(percentage) * flat.numel())))
        concentration[str(percentage)] = float(sorted_energy[:count].sum() / total)
    result = {'snr_db': float(snr), 'energy_concentration': concentration}
    if clean.is_complex():
        magnitude = clean.abs()
        threshold = torch.quantile(magnitude.flatten(), 0.5)
        valid = magnitude >= threshold
        phase_delta = torch.angle(noisy * clean.conj())[valid]
        if phase_delta.numel():
            resultant = torch.abs(torch.exp(1j * phase_delta).mean())
            result['phase_circular_variance'] = float(1.0 - resultant)
            result['phase_circular_std_rad'] = float(
                torch.sqrt((-2.0 * torch.log(resultant.clamp_min(1e-12))).clamp_min(0.0))
            )
            result['phase_valid_fraction'] = float(valid.float().mean())
            result['phase_threshold'] = float(threshold)
    return result


def channel_metrics(clean, noisy, energy_percentages):
    return [
        scalar_metrics(clean[:, channel:channel + 1], noisy[:, channel:channel + 1], energy_percentages)
        for channel in range(clean.shape[1])
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean', required=True)
    parser.add_argument('--noisy', required=True)
    parser.add_argument('--device', default='cuda:4')
    parser.add_argument('--frames', type=int, default=128)
    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    clean, noisy = load_patch(args.clean, args.noisy, args.frames, args.size)
    device = torch.device(args.device)
    clean, noisy = clean.to(device), noisy.to(device)
    transforms = {
        'dtcwt': DTCWT2D(levels=3).to(device),
        'csp': FourierPyramid2D(image_size=args.size, height=3, order=5).to(device),
    }
    result = {
        'clean': args.clean, 'noisy': args.noisy, 'frames': clean.shape[2],
        'spatial_size': list(clean.shape[-2:]), 'normalization': 'none',
        'energy_percentages': [0.01, 0.05, 0.10], 'transforms': {},
    }
    with torch.inference_mode():
        for name, transform in transforms.items():
            clean_coeffs = transform(clean)
            noisy_coeffs = transform(noisy)
            result['transforms'][name] = {}
            for branch_name, clean_value in branches(clean_coeffs, name):
                noisy_value = dict(branches(noisy_coeffs, name))[branch_name]
                result['transforms'][name][branch_name] = {
                    'shape': list(clean_value.shape),
                    **scalar_metrics(clean_value, noisy_value, result['energy_percentages']),
                    'per_channel': channel_metrics(
                        clean_value, noisy_value, result['energy_percentages']
                    ),
                }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
