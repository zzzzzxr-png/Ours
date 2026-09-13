"""Minimal complex SRDTrans forward/backward and dropout checks."""

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "prior" / "srdtrans"))

from SRDTrans_v2 import SRDTrans_v2  # noqa: E402
from SRDTrans_v2.complex_layers import SharedDropout  # noqa: E402


def main():
    torch.manual_seed(260911)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SRDTrans_v2(
        img_dim=8,
        img_time=4,
        in_channel=3,
        embedding_dim=8,
        window_size=4,
        num_heads=2,
        hidden_dim=16,
        num_transBlock=1,
        attn_dropout_rate=0.0,
        f_maps=[4, 8],
        input_dropout_rate=0.0,
    ).to(device)
    x = torch.randn(1, 3, 4, 8, 8, dtype=torch.complex64, device=device, requires_grad=True)
    output = model(x)
    assert output.shape == x.shape
    assert output.dtype == torch.complex64
    output.abs().square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()

    dropout = SharedDropout(0.5).train()
    ones = torch.ones(4096, dtype=torch.complex64, device=device) * (1 + 2j)
    dropped = dropout(ones)
    kept = dropped != 0
    assert kept.any() and (~kept).any()
    assert torch.allclose(dropped[kept].imag, 2 * dropped[kept].real)
    print("complex SRDTrans checks passed")


if __name__ == "__main__":
    main()
