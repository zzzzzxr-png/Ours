"""Minimal Fourier pyramid analysis/synthesis and adapter checks."""

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from representation import FourierPyramid2D, LearnedFourierPyramidAdapter  # noqa: E402


def main():
    torch.manual_seed(260920)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = FourierPyramid2D(128).to(device)
    adapter = LearnedFourierPyramidAdapter().to(device)
    x = torch.randn(1, 1, 2, 128, 128, device=device)
    coefficients = transform(x)
    assert len(coefficients.bands) == 3
    assert all(value.shape[1] == 6 for value in coefficients.bands)
    reconstruction = transform.inverse(coefficients)
    assert torch.allclose(reconstruction, x, atol=2e-4, rtol=2e-4)
    features = adapter.encode(coefficients)
    assert features.shape == (1, 32, 2, 64, 64)
    assert features.is_complex()
    decoded = adapter.decode(features, coefficients)
    output = transform.inverse(decoded)
    assert output.shape == x.shape and torch.isfinite(output).all()
    output.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in adapter.parameters())
    print('Fourier pyramid checks passed')


if __name__ == '__main__':
    main()
