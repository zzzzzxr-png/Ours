"""Minimal checks for the Gamma-mixture MPGN loss."""

import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from posterior.losses_gamma import (  # noqa: E402
    gamma_mixture_nb_predictive_and_posterior,
    gamma_mixture_nb_nll_from_ab,
    gamma_mixture_posterior_mode,
    gamma_nb_predictive_and_posterior,
)


def main():
    torch.manual_seed(260911)
    a = (torch.rand(1, 1, 3, 2, 2) + 1.0).requires_grad_()
    b = torch.full_like(a, 4.0)
    y = torch.rand(1, 1, 3, 2, 2) * 2.0
    kwargs = dict(alpha=1.0, beta=0.5, kmax=64, tail_tol=1e-7, chunk_t=2)

    single = gamma_nb_predictive_and_posterior(a, b, y, **kwargs)
    mixture_one = gamma_mixture_nb_predictive_and_posterior(a, b, y, **kwargs)
    assert torch.allclose(single['log_predictive'], mixture_one['log_predictive'], atol=1e-6)
    assert torch.allclose(single['x_post_phys'], mixture_one['x_post_phys'], atol=1e-6)

    a_three = a.repeat(1, 3, 1, 1, 1)
    b_three = b.repeat(1, 3, 1, 1, 1)
    mixture_three = gamma_mixture_nb_predictive_and_posterior(a_three, b_three, y, **kwargs)
    assert torch.allclose(single['log_predictive'], mixture_three['log_predictive'], atol=1e-6)
    assert torch.allclose(single['x_post_phys'], mixture_three['x_post_phys'], atol=1e-6)
    assert torch.allclose(
        mixture_three['component_responsibility'],
        torch.full_like(mixture_three['component_responsibility'], 1 / 3),
        atol=1e-6,
    )
    mask = torch.ones_like(y, dtype=torch.bool)
    bounded_nll = gamma_mixture_nb_nll_from_ab(
        a_three, b_three, y, mask, **kwargs
    )
    assert torch.allclose(bounded_nll, -mixture_three['log_predictive'].mean(), atol=1e-6)

    (-mixture_three['log_predictive'].mean()).backward()
    assert a.grad is not None and torch.isfinite(a.grad).all() and a.grad.abs().sum() > 0

    direct_a = torch.tensor([[[[[2.0]]]]])
    direct_b = torch.tensor([[[[[3.0]]]]])
    direct_log_q = torch.zeros(1, 1, 1, 1, 1, 1)
    direct_k = torch.zeros(1, 1, 1, 1, 1, 1)
    mode, boundary = gamma_mixture_posterior_mode(
        direct_a, direct_b, direct_log_q, direct_k
    )
    assert torch.allclose(mode, torch.tensor([[[[[0.25]]]]]), atol=1e-6)
    assert not boundary.any()
    zero_mode, boundary = gamma_mixture_posterior_mode(
        direct_a * 0.25, direct_b, direct_log_q, direct_k
    )
    assert torch.equal(zero_mode, torch.zeros_like(zero_mode)) and boundary.all()
    print('Gamma mixture checks passed')


if __name__ == '__main__':
    main()
