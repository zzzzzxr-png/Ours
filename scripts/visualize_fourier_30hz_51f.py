#!/usr/bin/env python3
"""Create labeled PNG visualizations for the 30 Hz Fourier representation."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage import io
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from representation import FourierPyramid2D


ROOT = Path('/data/zhouxirou/Ours_260911/experiments/representation_export/'
            '30Hz_fourier_51frames_patch64')
DATA = Path('/data/zhouxirou/All_Datasets/dataset_CI_syn_Q004_a5000_b1600/'
            'dataset_zxr_all_frequency_Q004_beta1600-1-T1000H245W245/30Hz')

BRANCHES = [('highpass', 1), ('scale0', 6), ('scale1', 6),
            ('scale2', 6), ('lowpass', 1)]


def load_source(path):
    value = np.asarray(io.imread(path), dtype=np.float32)[:51]
    top = (value.shape[1] - 64) // 2
    left = (value.shape[2] - 64) // 2
    return value[:, top:top + 64, left:left + 64]


def load_coeff(name, branch, channel, part, frame=None):
    value = np.load(ROOT / f'{name}_{branch}_c{channel:02d}_{part}.npy')
    return value if frame is None else value[frame]


def channel_rows():
    return [(branch, channel) for branch, count in BRANCHES
            for channel in range(count)]


def signed_limits(gt, noisy):
    limit = max(float(np.max(np.abs(gt))), float(np.max(np.abs(noisy))), 1e-12)
    return -limit, limit


def save_real_imag_montage(name, frame, out_dir, limits):
    rows = channel_rows()
    fig, axes = plt.subplots(len(rows), 2, figsize=(7.5, 42), squeeze=False)
    fig.suptitle(f'{name.upper()} | frame {frame:03d} | complex Fourier coefficients', y=0.995)
    for row, (branch, channel) in enumerate(rows):
        for col, part in enumerate(('real', 'imag')):
            image = load_coeff(name, branch, channel, part, frame)
            lo, hi = limits[(branch, channel, part)]
            ax = axes[row, col]
            ax.imshow(image, cmap='RdBu_r', vmin=lo, vmax=hi, interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f'{branch} c{channel:02d} {part}', fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(out_dir / f'{name}_frame{frame:03d}_real_imag.png', dpi=120)
    plt.close(fig)


def save_band_maps(name, frame, out_dir, mode):
    bands = [(f'scale{s}', c) for s in range(3) for c in range(6)]
    fig, axes = plt.subplots(3, 6, figsize=(15, 8), squeeze=False)
    fig.suptitle(f'{name.upper()} | frame {frame:03d} | band {mode}', y=0.995)
    for index, (branch, channel) in enumerate(bands):
        image = load_coeff(name, branch, channel, 'real', frame)
        imag = load_coeff(name, branch, channel, 'imag', frame)
        image = np.hypot(image, imag) if mode == 'magnitude' else np.arctan2(imag, image)
        ax = axes[index // 6, index % 6]
        ax.imshow(image, cmap='gray' if mode == 'magnitude' else 'twilight', interpolation='nearest')
        ax.set_title(f'{branch}, ori {channel}', fontsize=9)
        ax.axis('off')
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_dir / f'{name}_frame{frame:03d}_bands_{mode}.png', dpi=150)
    plt.close(fig)


def save_input_fft(name, frame, out_dir, source):
    image = source[frame]
    fft_image = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(image))))
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle(f'{name.upper()} | frame {frame:03d}')
    axes[0].imshow(image, cmap='gray'); axes[0].set_title('input patch'); axes[0].axis('off')
    axes[1].imshow(fft_image, cmap='magma'); axes[1].set_title('log FFT magnitude'); axes[1].axis('off')
    fig.tight_layout()
    fig.savefig(out_dir / f'{name}_frame{frame:03d}_input_fft.png', dpi=180)
    plt.close(fig)


def save_complex_3d(name, frame, out_dir):
    rows = channel_rows()
    fig = plt.figure(figsize=(13, 10))
    ax = fig.add_subplot(111, projection='3d')
    colors = {'highpass': '#c6dbef', 'scale0': '#6baed6',
              'scale1': '#3182bd', 'scale2': '#08519c', 'lowpass': '#08306b'}
    rng = np.random.default_rng(0)
    for z, (branch, channel) in enumerate(rows):
        real = load_coeff(name, branch, channel, 'real', frame).ravel()
        imag = load_coeff(name, branch, channel, 'imag', frame).ravel()
        scale = max(float(np.sqrt(np.mean(real ** 2 + imag ** 2))), 1e-12)
        real, imag = real / scale, imag / scale
        keep = rng.choice(real.size, size=min(800, real.size), replace=False)
        ax.scatter(real[keep], imag[keep], np.full(keep.size, z),
                   s=2, alpha=0.22, color=colors[branch])
    ax.set_xlabel('Re(c) / per-channel RMS')
    ax.set_ylabel('Im(c) / per-channel RMS')
    ax.set_zlabel('partition / channel')
    ax.set_zticks([0, 1, 7, 13, 19])
    ax.set_zticklabels(['highpass', 'scale0', 'scale1', 'scale2', 'lowpass'])
    ax.legend(handles=[plt.Line2D([], [], marker='o', linestyle='', color=color, label=branch)
                       for branch, color in colors.items()], loc='upper left')
    ax.set_title(f'{name.upper()} | frame {frame:03d} | x=Re, y=Im, z=partition channel')
    ax.view_init(elev=24, azim=-62)
    fig.tight_layout()
    fig.savefig(out_dir / f'{name}_frame{frame:03d}_complex_3d.png', dpi=180)
    plt.close(fig)


def save_frequency_response(transform, device, out_dir):
    impulse = torch.zeros(1, 1, 1, 64, 64, device=device)
    impulse[..., 32, 32] = 1
    with torch.inference_mode():
        coeffs = transform(impulse)
    values = [('highpass', coeffs.highpass), *
              [(f'scale{s}', x) for s, x in enumerate(coeffs.bands)],
              ('lowpass', coeffs.lowpass)]
    panels = []
    for branch, value in values:
        count = value.shape[1]
        for channel in range(count):
            x = value[0, channel, 0].cpu().numpy()
            response = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(x))))
            panels.append((f'{branch} c{channel:02d}', response))
    fig, axes = plt.subplots(4, 5, figsize=(13, 11), squeeze=False)
    fig.suptitle('Approximate Fourier-plane responses (impulse probe)')
    for ax, (title, image) in zip(axes.flat, panels):
        ax.imshow(image, cmap='magma', interpolation='nearest')
        ax.set_title(title, fontsize=8); ax.axis('off')
    for ax in axes.flat[len(panels):]: ax.axis('off')
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_dir / 'frequency_responses_impulse.png', dpi=180)
    plt.close(fig)

    pyramid = transform.pyramid
    full = int(transform.image_size[0])
    colors = {'highpass': '#c6dbef', 'scale0': '#6baed6',
              'scale1': '#3182bd', 'scale2': '#08519c', 'lowpass': '#08306b'}
    hi = pyramid.hi0mask[0].detach().cpu().numpy()

    # 3-D partition view: x/y are common Fourier coordinates, z is channel layer.
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(np.arange(full), np.arange(full))
    layer = 0
    ax.contour(X, Y, hi, levels=[0.35], zdir='z', offset=layer,
               colors=[colors['highpass']], linewidths=1.4)
    layer += 1
    offset, size = 0, full
    current = pyramid.lo0mask[0, 0, 0].detach().cpu().numpy()
    for scale in range(3):
        angle = getattr(pyramid, f'_anglemasks_scale_{scale}')[0, 0].detach().cpu().numpy()
        radial = getattr(pyramid, f'_himasks_scale_{scale}')[0].detach().cpu().numpy()
        for orientation in range(6):
            embedded = np.zeros((full, full), dtype=np.float32)
            embedded[offset:offset + size, offset:offset + size] = np.abs(
                current * angle[orientation] * radial)
            ax.contour(X, Y, embedded, levels=[0.35], zdir='z', offset=layer,
                       colors=[colors[f'scale{scale}']], linewidths=1.1)
            layer += 1
        lostart, loend = pyramid._loindices[scale]
        low_mask = getattr(pyramid, f'_lomasks_scale_{scale}')[0].detach().cpu().numpy()
        current = current[lostart[0]:loend[0], lostart[1]:loend[1]] * low_mask
        offset += int(lostart[0]); size = int(loend[0] - lostart[0])
    low = np.zeros((full, full), dtype=np.float32)
    low[offset:offset + size, offset:offset + size] = current
    ax.contour(X, Y, low, levels=[0.5], zdir='z', offset=layer,
               colors=[colors['lowpass']], linewidths=1.4)
    ax.set_xlim(0, full); ax.set_ylim(full, 0); ax.set_zlim(0, layer)
    ax.set_xlabel('Fourier kx'); ax.set_ylabel('Fourier ky'); ax.set_zlabel('partition layer')
    ax.set_zticks([0, 1, 7, 13, 19]); ax.set_zticklabels(
        ['HP', 'scale0', 'scale1', 'scale2', 'LP'])
    ax.set_title('3-D Fourier partition layers')
    ax.view_init(elev=28, azim=-58)
    fig.tight_layout()
    fig.savefig(out_dir / 'frequency_partition_3d_layers.png', dpi=200)
    plt.close(fig)

    # Clean schematic overview: ideal concentric radial bands plus axes.
    full = int(transform.image_size[0])
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    colors = {'highpass': '#c6dbef', 'scale0': '#6baed6',
              'scale1': '#3182bd', 'scale2': '#08519c', 'lowpass': '#08306b'}
    center = 0.0
    radius = full * 0.47
    # Draw only one representative center line for each radial band.
    from matplotlib.patches import Circle
    ring_lines = [(0.10, colors['lowpass'], 'low'),
                  (0.23, colors['scale2'], 'low-medium'),
                  (0.365, colors['scale1'], 'medium'),
                  (0.565, colors['scale0'], 'high'),
                  (0.85, colors['highpass'], 'highpass')]
    for fraction, color, _ in ring_lines:
        ax.add_patch(Circle((center, center), radius * fraction, fill=False,
                            edgecolor=color, linewidth=2.4))
    for orientation, theta in enumerate(np.linspace(0, np.pi, 6, endpoint=False)):
        dx, dy = radius * np.cos(theta), radius * np.sin(theta)
        ax.plot([center - dx, center + dx], [center - dy, center + dy],
                color='#555555', linewidth=1.1, alpha=0.35 + 0.10 * orientation)
    ax.set_title('Schematic Fourier partition: low / medium / high bands and six directions')
    ax.set_xlim(-radius, radius); ax.set_ylim(-radius, radius)
    ax.set_xlabel('frequency kx', color='black'); ax.set_ylabel('frequency ky', color='black')
    ax.tick_params(colors='black')
    handles = [plt.Line2D([], [], color=colors['lowpass'], lw=2.4, label='low frequency'),
               plt.Line2D([], [], color=colors['scale1'], lw=2.4, label='medium frequency'),
               plt.Line2D([], [], color=colors['scale0'], lw=2.4, label='high frequency'),
               plt.Line2D([], [], color='#555555', lw=1.2, label='orientation axes')]
    ax.legend(handles=handles, loc='upper right', facecolor='white', labelcolor='black', edgecolor='#555555')
    fig.tight_layout()
    fig.savefig(out_dir / 'frequency_partition_masks.png', dpi=220, facecolor='white')
    plt.close(fig)

    # One Fourier-plane overview: retain only approximate support edges.
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor('black')
    colors = {'highpass': '#ffffff', 'scale0': '#ff595e',
              'scale1': '#ffca3a', 'scale2': '#8ac926', 'lowpass': '#4d9de0'}
    line_styles = {'highpass': '-', 'scale0': '-', 'scale1': '--',
                   'scale2': ':', 'lowpass': '-'}
    for branch, value in values:
        count = value.shape[1]
        for channel in range(count):
            x = value[0, channel, 0].cpu().numpy()
            response = np.abs(np.fft.fftshift(np.fft.fft2(x)))
            response /= max(float(response.max()), 1e-12)
            contour = ax.contour(response, levels=[0.35], colors=[colors[branch]],
                                  linewidths=1.2, linestyles=line_styles[branch])
            paths = contour.get_paths()
            if paths:
                center = paths[0].vertices.mean(axis=0)
                ax.text(center[0], center[1],
                        'HP' if branch == 'highpass' else
                        ('LP' if branch == 'lowpass' else f'{branch[-1]}:{channel}'),
                        color=colors[branch], fontsize=7, ha='center', va='center')
    ax.set_title('Approximate Fourier-plane support edges\nwhite=highpass, blue=lowpass; colors/linestyles=scales')
    ax.set_xlim(0, 64); ax.set_ylim(64, 0)
    ax.set_xlabel('frequency kx'); ax.set_ylabel('frequency ky')
    fig.tight_layout()
    fig.savefig(out_dir / 'frequency_responses_overlay_edges.png', dpi=220,
                facecolor='black')
    plt.close(fig)


def main():
    out_dir = ROOT / 'png'
    out_dir.mkdir(exist_ok=True)
    gt_source = load_source(DATA / '30Hz_GT/clean_30Hz_1000frames.tif')
    noisy_source = load_source(DATA / '30Hz_Noisy/noise_30Hz_4RPN_-0.32dBSNR.tif')
    limits = {}
    for branch, channel in channel_rows():
        for part in ('real', 'imag'):
            limits[(branch, channel, part)] = signed_limits(
                load_coeff('gt', branch, channel, part),
                load_coeff('noisy', branch, channel, part),
            )
    for name, source in (('gt', gt_source), ('noisy', noisy_source)):
        save_input_fft(name, 0, out_dir, source)
        save_band_maps(name, 0, out_dir, 'magnitude')
        save_band_maps(name, 0, out_dir, 'phase')
        save_complex_3d(name, 0, out_dir)
        for frame in range(51):
            save_real_imag_montage(name, frame, out_dir, limits)
    device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
    save_frequency_response(FourierPyramid2D(64, height=3, order=5).to(device).eval(), device, out_dir)
    print(f'created PNGs in {out_dir}')


if __name__ == '__main__':
    main()
