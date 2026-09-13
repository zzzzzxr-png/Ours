"""Experiment 1: reconstruction error of candidate physics transforms."""

import argparse
import math
from pathlib import Path

import tifffile
import torch
from pytorch_wavelets import DTCWTForward, DTCWTInverse


def relative_error(x, x_rec):
    denominator = torch.linalg.vector_norm(x)
    if denominator == 0:
        raise ValueError("Relative error is undefined for an all-zero input.")
    return (torch.linalg.vector_norm(x - x_rec) / denominator).item()


def dtcwt_error(x, levels, batch_size):
    forward = DTCWTForward(J=levels, mode="symmetric").to(x.device)
    inverse = DTCWTInverse(mode="symmetric").to(x.device)
    squared_error = 0.0
    squared_norm = 0.0
    height, width = x.shape[-2:]

    for frames in x.split(batch_size):
        frames = frames[:, None]
        lowpass, highpasses = forward(frames)
        reconstructed = inverse((lowpass, highpasses))[..., :height, :width]
        squared_error += (frames - reconstructed).double().square().sum().item()
        squared_norm += frames.double().square().sum().item()

    return math.sqrt(squared_error / squared_norm)


def riesz_quaternion(x):
    """Return (X, R_t, R_y, R_x) for a real [T, H, W] volume."""
    spectrum = torch.fft.fftn(x, norm="ortho")
    frequencies = torch.meshgrid(
        *(torch.fft.fftfreq(n, device=x.device) for n in x.shape), indexing="ij"
    )
    radius = torch.sqrt(sum(frequency.square() for frequency in frequencies))
    radius = torch.where(radius == 0, torch.ones_like(radius), radius)
    riesz = tuple(
        torch.fft.ifftn((-1j * frequency / radius) * spectrum, norm="ortho").real
        for frequency in frequencies
    )
    return (x, *riesz)


def markdown_table(results):
    lines = [
        "| Transform | Physics meaning | Invertibility error | Implementation |",
        "|---|---|---:|---|",
    ]
    for name, meaning, error, implementation in results:
        lines.append(f"| {name} | {meaning} | {error:.3e} | {implementation} |")
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input fluorescence TIFF stack [T,H,W]")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--levels", type=int, default=3, help="DTCWT decomposition levels")
    parser.add_argument("--dtcwt-batch-size", type=int, default=16)
    parser.add_argument("--max-frames", type=int, help="Optional prefix for a quick check")
    parser.add_argument("--threshold", type=float, default=1e-5)
    parser.add_argument("--output", type=Path, help="Optional Markdown result path")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    if args.levels < 1 or args.dtcwt_batch_size < 1:
        raise ValueError("--levels and --dtcwt-batch-size must be positive.")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("--max-frames must be positive.")
    if args.input.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError("Input must be a .tif or .tiff stack.")

    array = tifffile.imread(args.input)
    if array.ndim != 3:
        raise ValueError(f"Expected [T,H,W], got shape {array.shape}.")
    x = torch.as_tensor(array[: args.max_frames], dtype=torch.float32, device=args.device)

    identity_error = relative_error(x, x)
    fft_error = relative_error(
        x, torch.fft.ifftn(torch.fft.fftn(x, norm="ortho"), norm="ortho").real
    )
    dtcwt_rec_error = dtcwt_error(x, args.levels, args.dtcwt_batch_size)
    quaternion = riesz_quaternion(x)
    riesz_error = relative_error(x, quaternion[0])

    results = [
        ("Identity", "reference representation", identity_error, "PyTorch identity"),
        ("FFT", "global frequency amplitude/phase", fft_error, "torch.fft"),
        ("DTCWT", "local scale/orientation/phase", dtcwt_rec_error, "pytorch_wavelets"),
        ("Riesz", "local geometry/quadrature", riesz_error, "torch.fft"),
    ]
    table = markdown_table(results)
    report = (
        f"# Experiment 1: transform invertibility\n\n"
        f"- Input: `{args.input}`\n"
        f"- Shape: `{tuple(x.shape)}`\n"
        f"- Dtype/device: `{x.dtype}` / `{x.device}`\n"
        f"- Acceptance threshold: `{args.threshold:.1e}`\n\n"
        f"{table}\n"
    )
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")

    failed = [name for name, _, error, _ in results if error > args.threshold]
    if failed:
        raise SystemExit(f"Rejected (error > {args.threshold:.1e}): {', '.join(failed)}")


if __name__ == "__main__":
    main()
