import torch

from likelihood.losses import (
    mpgn_nll_single_target,
    mpgn_pn2v_marginal_nll,
    mpgn_pn2v_posterior,
)


def test_pn2v_invariants():
    torch.manual_seed(7)
    candidates = torch.randn(1, 4, 3, 4, 4, requires_grad=True) * 1000 + 5000
    target = torch.randn(1, 1, 3, 4, 4) * 1000 + 5000
    mask = torch.zeros_like(target)
    mask[:, :, 1:, 1:, 1:] = 1
    kwargs = dict(pred_img_mean=0., target_img_mean=0., alpha=5000., beta=1600.,
                  kmax=512, quant_step=1.)
    loss = mpgn_pn2v_marginal_nll(candidates, target, mask, **kwargs)
    perm = mpgn_pn2v_marginal_nll(candidates[:, [2, 0, 3, 1]], target, mask, **kwargs)
    assert torch.allclose(loss, perm, atol=1e-5, rtol=1e-6)

    changed = target.clone()
    changed[mask == 0] = 1e9
    assert torch.allclose(
        loss, mpgn_pn2v_marginal_nll(candidates, changed, mask, **kwargs),
        atol=1e-5, rtol=1e-6)

    one = candidates[:, :1].detach().requires_grad_()
    pn = mpgn_pn2v_marginal_nll(one, target, mask, **kwargs)
    ref = mpgn_nll_single_target(one, target, mask, 0., 0., 5000., 1600.,
                                 kmax=512, chunk_t=3, quant_step=1.)
    assert torch.allclose(pn, ref, atol=1e-5, rtol=1e-6)
    pn.backward()

    mean, ess = mpgn_pn2v_posterior(
        candidates.detach(), target, 0., 0., 5000., 1600.,
        kmax=512, quant_step=1.)
    assert mean.shape == target.shape
    assert 1.0 <= float(ess) <= 4.0


if __name__ == '__main__':
    test_pn2v_invariants()
    print('pn2v invariants: OK')
