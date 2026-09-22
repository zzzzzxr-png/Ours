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
    transform = FourierPyramid2D(64).to(device)
    adapter = LearnedFourierPyramidAdapter().to(device)
    x = torch.randn(1, 1, 2, 64, 64, device=device)
    coefficients = transform(x)
    assert len(coefficients.bands) == 3
    assert all(value.shape[1] == 6 for value in coefficients.bands)
    reconstruction = transform.inverse(coefficients)
    assert torch.allclose(reconstruction, x, atol=2e-4, rtol=2e-4)
    features = adapter.encode(coefficients)
    assert features.shape == (1, 23, 2, 32, 32)
    assert features.is_complex()
    print('Fourier pyramid checks passed')


if __name__ == '__main__':
    main()
