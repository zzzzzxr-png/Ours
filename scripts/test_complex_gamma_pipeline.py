"""End-to-end smoke check: masked intensity -> DTCWT/SRDTrans -> pixel NLL."""

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from likelihood.backbone_factory import build_denoise_network_srdtrans  # noqa: E402
from likelihood.trainer import (  # noqa: E402
    _TrainingDiagnostics,
    _adaptive_clip_grad_,
    _diagnostic_parameter_group,
    _estimate_fixed_dtcwt_scales,
    training_class_srdtrans,
)
from posterior.gamma_posterior import gamma_ab_from_mu_kappa  # noqa: E402
from posterior.losses_gamma import gamma_nb_predictive_and_posterior  # noqa: E402
from SRDTrans_v2.complex_layers import ComplexRMSNorm3d  # noqa: E402


def main():
    resume_tmp = tempfile.TemporaryDirectory()
    resume_trainer = training_class_srdtrans({})
    resume_trainer.pth_path = resume_tmp.name
    resume_trainer.local_model = torch.nn.Linear(2, 1)
    resume_optimizer = torch.optim.Adam(resume_trainer.local_model.parameters(), lr=1e-3)
    resume_loss = resume_trainer.local_model(torch.ones(1, 2)).sum()
    resume_optimizer.zero_grad()
    resume_loss.backward()
    resume_optimizer.step()
    resume_trainer.save_model(2, 4, resume_optimizer, global_iter=123)

    resumed = training_class_srdtrans({})
    resumed.pth_path = resume_tmp.name
    resumed.local_model = torch.nn.Linear(2, 1)
    assert resumed._try_resume_checkpoint()
    assert resumed._resume_start_epoch == 3
    assert resumed._resume_global_iter == 123
    resumed_optimizer = torch.optim.Adam(resumed.local_model.parameters(), lr=1e-3)
    resumed_optimizer.load_state_dict(resumed._resume_optimizer_state)
    assert resumed_optimizer.state
    resume_tmp.cleanup()

    complex_input = torch.complex(torch.randn(2, 3, 4, 5, 6),
                                  torch.randn(2, 3, 4, 5, 6))
    normalized = ComplexRMSNorm3d()(complex_input)
    normalized_rms = normalized.abs().square().mean(dim=(2, 3, 4)).sqrt()
    assert torch.allclose(normalized_rms, torch.ones_like(normalized_rms), atol=1e-5)
    assert torch.allclose(torch.angle(normalized), torch.angle(complex_input))

    assert _diagnostic_parameter_group(
        'backbone.encoders.2.conv_net.weight') == 'encoders.2'
    parameter = torch.nn.Parameter(torch.tensor([0.0]))
    warmup_norms = []
    threshold = None
    for value in (1.0, 2.0, 3.0, 4.0):
        parameter.grad = torch.tensor([value])
        _, threshold = _adaptive_clip_grad_(
            [parameter], warmup_norms, threshold, 4, 95.0, 1e8, 2.0)
    assert abs(threshold - 3.85) < 1e-6
    parameter.grad = torch.tensor([10.0])
    norm, threshold = _adaptive_clip_grad_(
        [parameter], warmup_norms, threshold, 4, 95.0, 1e8, 2.0)
    assert norm == 10.0 and abs(parameter.grad.item()) <= threshold

    contaminated_norms = []
    contaminated_threshold = None
    for value in (1.0, 1.0, 1.0, 100.0):
        parameter.grad = torch.tensor([value])
        _, contaminated_threshold = _adaptive_clip_grad_(
            [parameter], contaminated_norms, contaminated_threshold,
            4, 95.0, 1e8, 2.0)
    assert contaminated_threshold == 2.0

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
        dtcwt_channel_normalize=True,
        dtcwt_channel_scales=[1.0, 1.0, 1.0, 1.0],
        gradient_checkpointing=False,
    )
    model = build_denoise_network_srdtrans(cfg).to(device)
    assert model.backbone.f_maps == [20, 20]
    assert model.backbone.gradient_checkpointing is False
    masked = torch.randn(1, 1, 4, 8, 8, device=device)
    target = torch.randn_like(masked)
    diagnostic_tmp = tempfile.TemporaryDirectory()
    diagnostic_path = str(Path(diagnostic_tmp.name) / 'diagnostics.csv')
    diagnostics = _TrainingDiagnostics(model, diagnostic_path, interval=1)
    before = diagnostics.begin(1)
    prediction = model(masked)
    assert prediction.shape == masked.shape and not prediction.is_complex()
    assert torch.isfinite(prediction).all()

    coefficients = model.representation(masked)
    direct = model.adapter.decode(model.adapter.encode(coefficients), coefficients)
    assert torch.allclose(direct.low, coefficients.low, atol=2e-3, rtol=2e-3)
    assert all(torch.allclose(output, value, atol=2e-3, rtol=2e-3)
               for output, value in zip(direct.highs, coefficients.highs))

    fixed_scales = _estimate_fixed_dtcwt_scales(
        [np.random.default_rng(1024).normal(size=(8, 8, 8)).astype(np.float32)],
        levels=3, chunk_frames=4,
    )
    assert len(fixed_scales) == 4 and all(value > 0 for value in fixed_scales)

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
    torch.optim.Adam(model.parameters(), lr=1e-5).step()
    diagnostics.finish(1, float(loss.detach().cpu()), before)
    assert 'activation_rms,dtcwt_input' in Path(diagnostic_path).read_text()
    assert ',parameter,adapter,' in Path(diagnostic_path).read_text()
    diagnostic_tmp.cleanup()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert gradients and all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    backbone_gradients = [
        parameter.grad for parameter in model.backbone.parameters()
    ]
    assert any(gradient.abs().sum() > 0 for gradient in backbone_gradients)
    print('DTCWT complex Gamma pipeline checks passed')


if __name__ == '__main__':
    main()
