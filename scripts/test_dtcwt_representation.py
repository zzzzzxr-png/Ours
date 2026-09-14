"""Minimal DTCWT reconstruction, adapter packing, and gradient checks."""

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from representation.dtcwt import DTCWT2D, pack_low, unpack_low  # noqa: E402
from representation.dtcwt_adapter import _block_mean, _repeat_to  # noqa: E402


def main():
    torch.manual_seed(260914)
    x = torch.randn(1, 1, 4, 32, 32, requires_grad=True)
    transform = DTCWT2D(levels=3)

    coefficients = transform(x)
    reconstructed = transform.inverse(coefficients)
    relative_error = (x - reconstructed).norm() / x.norm()
    assert relative_error.item() < 1e-6, relative_error.item()

    frames = x.detach().permute(0, 2, 1, 3, 4).reshape(4, 1, 32, 32)
    yl, _ = transform.analysis(frames)
    pack_error = (yl - pack_low(unpack_low(yl))).abs().max()
    assert pack_error.item() < 1e-6, pack_error.item()
    aligned = _repeat_to(coefficients.low, coefficients.spatial_size)
    alignment_error = (
        _block_mean(aligned, coefficients.low.shape[-2:]) - coefficients.low
    ).abs().max()
    assert alignment_error.item() < 1e-6, alignment_error.item()

    reconstructed.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    print('DTCWT checks passed: rel_error={:.3g}, pack_error={:.3g}'.format(
        relative_error.item(), pack_error.item()))


if __name__ == '__main__':
    main()
