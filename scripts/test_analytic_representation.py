"""Minimal runnable checks for the fixed analytic operators."""

import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from posterior.analytic_representation import (  # noqa: E402
    analytic_channel,
    analytic_inverse_axis,
    analytic_projection,
    analytic_pseudoinverse,
    analytic_representation,
    analytic_residual_ratio,
    quadrature,
    quadrature_contribution,
)


def relative_error(actual, expected):
    return ((actual - expected).norm() / expected.norm().clamp_min(1e-12)).item()


def check_shape(shape):
    torch.manual_seed(260911)
    x = torch.randn(shape, dtype=torch.float32)
    representation = analytic_representation(x)
    assert representation.shape == (shape[0], 3, *shape[2:])
    assert representation.dtype == torch.complex64

    for index, direction in enumerate(("x", "y", "t")):
        z = representation[:, index:index + 1]
        reconstructed = analytic_pseudoinverse(z, direction)
        assert relative_error(reconstructed, x) < 1e-5

        projected = analytic_projection(z, direction)
        assert relative_error(projected, z) < 1e-5
        assert analytic_residual_ratio(z, direction).max().item() < 1e-5
        assert quadrature_contribution(z, direction).isfinite().all()

        arbitrary = torch.complex(torch.randn_like(x), torch.randn_like(x))
        arbitrary_projected = analytic_projection(arbitrary, direction)
        assert relative_error(
            analytic_projection(arbitrary_projected, direction), arbitrary_projected
        ) < 1e-5
        assert relative_error(
            analytic_pseudoinverse(arbitrary_projected, direction),
            analytic_pseudoinverse(arbitrary, direction),
        ) < 1e-5

        u = torch.randn_like(x)
        v = torch.randn_like(x)
        lhs = (quadrature(u, direction) * v).sum()
        rhs = -(u * quadrature(v, direction)).sum()
        assert torch.allclose(lhs, rhs, rtol=1e-5, atol=1e-5)

        noise = torch.complex(torch.randn_like(x), torch.randn_like(x)) * 1e-3
        amplification = (
            analytic_pseudoinverse(z + noise, direction) - reconstructed
        ).norm() / noise.abs().norm()
        assert amplification.item() <= 1.0 + 1e-5


def check_gradients():
    torch.manual_seed(260911)
    u = torch.randn((1, 1, 5, 7, 9), requires_grad=True)
    v = torch.randn_like(u, requires_grad=True)
    z = torch.complex(u, v)
    loss = sum(analytic_pseudoinverse(z, direction).square().mean()
               for direction in ("x", "y", "t"))
    loss.backward()
    assert u.grad is not None and torch.isfinite(u.grad).all() and u.grad.abs().sum() > 0
    assert v.grad is not None and torch.isfinite(v.grad).all() and v.grad.abs().sum() > 0

    shared_real = torch.randn((1, 1, 5, 7, 9), requires_grad=True)
    quadratures = [torch.randn_like(shared_real, requires_grad=True) for _ in range(3)]
    loss = sum(
        analytic_inverse_axis(shared_real, value, dim).square().mean()
        for value, dim in zip(quadratures, (-1, -2, -3))
    )
    loss.backward()
    assert shared_real.grad is not None and shared_real.grad.abs().sum() > 0
    for value in quadratures:
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert value.grad.abs().sum() > 0


def main():
    check_shape((2, 1, 6, 8, 10))
    check_shape((1, 1, 5, 7, 9))
    constant = torch.ones((1, 1, 6, 8, 10))
    for direction in ("x", "y", "t"):
        assert quadrature(constant, direction).abs().max().item() < 1e-6
        assert torch.allclose(
            analytic_pseudoinverse(analytic_channel(constant, direction), direction),
            constant,
            rtol=1e-6,
            atol=1e-6,
        )
    check_gradients()
    print("analytic representation checks passed")


if __name__ == "__main__":
    main()
