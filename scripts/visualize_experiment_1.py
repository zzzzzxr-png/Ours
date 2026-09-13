"""Visualize Identity, FFT, DTCWT, Riesz, and reconstruction residuals."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch
from pytorch_wavelets import DTCWTForward, DTCWTInverse

from experiment_1_invertibility import riesz_quaternion


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def add_image(ax, image, title, cmap="gray", **kwargs):
    shown = ax.imshow(image, cmap=cmap, **kwargs)
    ax.set_title(title)
    ax.set_axis_off()
    return shown


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input fluorescence TIFF stack [T,H,W]")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame", type=int, help="Frame to show; defaults to T//2")
    parser.add_argument("--max-frames", type=int, help="Optional prefix to visualize")
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    array = tifffile.imread(args.input)
    if array.ndim != 3:
        raise ValueError(f"Expected [T,H,W], got shape {array.shape}.")
    if args.max_frames is not None:
        if args.max_frames < 1:
            raise ValueError("--max-frames must be positive.")
        array = array[: args.max_frames]
    frame_index = len(array) // 2 if args.frame is None else args.frame
    if not 0 <= frame_index < len(array):
        raise ValueError(f"--frame must be in [0, {len(array) - 1}].")
    if args.levels < 1:
        raise ValueError("--levels must be positive.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    x = torch.as_tensor(array, dtype=torch.float32, device=args.device)
    frame = x[frame_index].cpu().numpy()
    low, high = np.percentile(frame, (1, 99))
    center_y, center_x = frame.shape[0] // 2, frame.shape[1] // 2

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    add_image(axes[0], frame, f"Raw frame t={frame_index}", vmin=low, vmax=high)
    add_image(axes[1], array[:, center_y], f"x-t slice at y={center_y}", aspect="auto")
    radius = 8
    roi = array[:, center_y - radius:center_y + radius, center_x - radius:center_x + radius]
    axes[2].plot(roi.mean(axis=(1, 2)))
    axes[2].set(title="Center ROI mean", xlabel="Frame", ylabel="Fluorescence")
    save_figure(fig, args.output_dir / "identity.png")

    spectrum = torch.fft.fftn(x, norm="ortho")
    fft_rec = torch.fft.ifftn(spectrum, norm="ortho").real[frame_index]
    magnitude = spectrum.abs()
    spatial_magnitude = torch.fft.fftshift(magnitude.mean(0)).cpu().numpy()
    tx_magnitude = torch.fft.fftshift(magnitude.mean(1)).cpu().numpy()
    phase_plane = torch.fft.fftshift(torch.angle(spectrum[0])).cpu().numpy()
    phase_amplitude = torch.fft.fftshift(magnitude[0]).cpu().numpy()
    phase_plane = np.ma.masked_where(phase_amplitude < np.percentile(phase_amplitude, 75), phase_plane)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    add_image(axes[0], np.log1p(spatial_magnitude), "Mean FFT amplitude over temporal frequency")
    add_image(axes[1], np.log1p(tx_magnitude), "Temporal-spatial FFT amplitude", aspect="auto")
    shown = add_image(axes[2], phase_plane, "Temporal-DC phase (low amplitude masked)", "twilight", vmin=-np.pi, vmax=np.pi)
    fig.colorbar(shown, ax=axes[2], fraction=0.046)
    save_figure(fig, args.output_dir / "fft.png")
    del spectrum, magnitude

    dtcwt_forward = DTCWTForward(J=args.levels, mode="symmetric").to(args.device)
    dtcwt_inverse = DTCWTInverse(mode="symmetric").to(args.device)
    selected = x[frame_index][None, None]
    lowpass, highpasses = dtcwt_forward(selected)
    dtcwt_rec = dtcwt_inverse((lowpass, highpasses))[0, 0, : frame.shape[0], : frame.shape[1]]
    angles = (15, 45, 75, -75, -45, -15)
    fig_amp, axes_amp = plt.subplots(args.levels, 6, figsize=(15, 2.5 * args.levels), squeeze=False)
    fig_phase, axes_phase = plt.subplots(args.levels, 6, figsize=(15, 2.5 * args.levels), squeeze=False)
    for level, coefficients in enumerate(highpasses):
        coefficients = coefficients[0, 0].cpu().numpy()
        amplitude = np.linalg.norm(coefficients, axis=-1)
        phase = np.arctan2(coefficients[..., 1], coefficients[..., 0])
        vmax = np.percentile(amplitude, 99)
        for orientation, angle in enumerate(angles):
            add_image(axes_amp[level, orientation], amplitude[orientation], f"L{level + 1}, {angle}°", "magma", vmin=0, vmax=vmax)
            masked_phase = np.ma.masked_where(amplitude[orientation] < 0.05 * vmax, phase[orientation])
            add_image(axes_phase[level, orientation], masked_phase, f"L{level + 1}, {angle}°", "twilight", vmin=-np.pi, vmax=np.pi)
    save_figure(fig_amp, args.output_dir / "dtcwt_amplitude.png")
    save_figure(fig_phase, args.output_dir / "dtcwt_phase.png")

    quaternion = riesz_quaternion(x)
    _, r_t, r_y, r_x = (component[frame_index].cpu().numpy() for component in quaternion)
    vector_amplitude = np.sqrt(r_t**2 + r_y**2 + r_x**2)
    local_amplitude = np.sqrt(frame**2 + vector_amplitude**2)
    local_phase = np.arctan2(vector_amplitude, frame)
    component_limit = np.percentile(np.abs(np.stack((r_t, r_y, r_x))), 99)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    add_image(axes[0, 0], frame, "X", vmin=low, vmax=high)
    for ax, component, name in zip(axes.flat[1:4], (r_t, r_y, r_x), ("R_t", "R_y", "R_x")):
        add_image(ax, component, name, "coolwarm", vmin=-component_limit, vmax=component_limit)
    add_image(axes[1, 1], local_amplitude, "Local amplitude", "magma")
    shown = add_image(axes[1, 2], local_phase, "Local phase", "twilight", vmin=0, vmax=np.pi)
    fig.colorbar(shown, ax=axes[1, 2], fraction=0.046)
    save_figure(fig, args.output_dir / "riesz.png")

    residuals = (
        np.zeros_like(frame),
        np.abs(frame - fft_rec.cpu().numpy()),
        np.abs(frame - dtcwt_rec.cpu().numpy()),
        np.abs(frame - quaternion[0][frame_index].cpu().numpy()),
    )
    residual_limit = max(np.percentile(residual, 99) for residual in residuals)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, residual, name in zip(axes, residuals, ("Identity", "FFT", "DTCWT", "Riesz")):
        add_image(ax, residual, f"{name} |X-X_rec|", "magma", vmin=0, vmax=residual_limit)
    save_figure(fig, args.output_dir / "reconstruction_residuals.png")

    print(f"Saved visualizations to {args.output_dir}")


if __name__ == "__main__":
    main()
