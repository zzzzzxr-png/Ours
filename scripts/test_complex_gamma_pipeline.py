"""End-to-end smoke check: masked intensity -> DTCWT/SRDTrans -> pixel NLL."""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from likelihood.backbone_factory import build_denoise_network_srdtrans  # noqa: E402
from posterior.gamma_posterior import gamma_ab_from_mu_kappa  # noqa: E402
from posterior.losses_gamma import gamma_nb_predictive_and_posterior  # noqa: E402


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = SimpleNamespace(
        backbone='srdtrans_v2', patch_x=8, patch_t=4,
        srdtrans_root=str(ROOT / 'prior' / 'srdtrans' / 'SRDTrans_v2'),
        srdtrans_f_maps=[4, 8], embedding_dim=8, num_heads=2,
        hidden_dim=16, window_size=4, num_transBlock=1,
        attn_dropout_rate=0.0, input_dropout_rate=0.0,
        trans_order='ts', space_post_norm=False, space_dropout_rate=0.0,
        use_msconv_before_trans=False, kappa_mode='fixed', mpgn_kappa=10.0,
        representation='dtcwt', dtcwt_dim=2, dtcwt_levels=3,
    )
    model = build_denoise_network_srdtrans(cfg).to(device)
    masked = torch.randn(1, 1, 4, 8, 8, device=device)
    target = torch.randn_like(masked)
    prediction = model(masked)
    assert prediction.shape == masked.shape and not prediction.is_complex()
    assert ((prediction - masked).norm() / masked.norm()).item() < 1e-6

    mean = 10.0
    mu_lambda = (prediction + mean).clamp_min(1e-6)
    a, b = gamma_ab_from_mu_kappa(mu_lambda, torch.full_like(mu_lambda, 10.0))
    result = gamma_nb_predictive_and_posterior(
        a, b, target + mean, alpha=1.0, beta=1.0,
        valid_mask=torch.ones_like(target, dtype=torch.bool),
        kmax=64, tail_tol=1e-7, chunk_t=2,
    )
    loss = result['nll_mean']
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert gradients and all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    decoder_gradients = [
        parameter.grad for projection in model.adapter.output_projections
        for parameter in projection.parameters()
    ]
    assert any(gradient.abs().sum() > 0 for gradient in decoder_gradients)
    print('DTCWT complex Gamma pipeline checks passed')


if __name__ == '__main__':
    main()
