"""End-to-end exact-MPGN proximal unfolding with a shared denoiser.

Training uses masked self-supervision:

    y_masked = masked / neighbor-replaced observation
    y_target = raw observation, used only in the masked exact MPGN NLL

The network follows a prior-warm-start + exact-MPGN-correction + prior-refinement form:

    x_0 = Pi_+(D_theta(y_masked))

    for k = 1, ..., K:
        z_k = C_exactMPGN(x_{k-1}, y_obs, rho_k, Kmax)
        x_k = Pi_+(D_theta(z_k))

where y_obs should be y_masked during masked training to avoid target leakage,
and raw y during validation / inference.

The final training loss is evaluated only on x_K:

    L = masked exact MPGN NLL(x_K, y_target)

No intermediate loss is used.
"""
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from prior.deepcadrt.unet3d import UNet3D


class ExactMPGNProxCorrection(nn.Module):
    """
    Few-step exact-MPGN proximal correction.

    MPGN model:
        Y = offset + alpha * K + eps
        K   ~ Poisson(signal / alpha)
        eps ~ N(0, beta)

    Exact marginal likelihood:
        p(y | x) = sum_k Pois(k; (x-offset)/alpha)
                         N(y; offset + alpha*k, beta)

    Correction solves approximately:

        z = argmin_x  ell_MPGN(y_obs | x)
                    + 0.5 * rho * (x - x_ref)^2 / v_ref

    where:
        v_ref = alpha * max(x_ref - offset, 0) + beta

    This is implemented as a fixed number of differentiable Newton-like steps.
    """

    def __init__(
        self,
        alpha: float,
        beta: float,
        offset: float = 0.0,
        rho_min: float = 1e-6,
        eps: float = 1e-8,
        kmax: int = 32,
        n_iter: int = 2,
        chunk_t: int = 8,
        step_cap_sigma: float = 1.0,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.offset = float(offset)
        self.rho_min = float(rho_min)
        self.eps = float(eps)
        self.kmax = int(kmax)
        self.n_iter = int(n_iter)
        self.chunk_t = int(chunk_t)
        self.step_cap_sigma = float(step_cap_sigma)

    @staticmethod
    def to_phys(x_centered, patch_mean):
        if patch_mean is None:
            raise ValueError(
                "patch_mean is required for exact-MPGN correction, "
                "because correction must be computed in physical units."
            )
        patch_mean = patch_mean.to(dtype=x_centered.dtype, device=x_centered.device)
        return x_centered + patch_mean

    @staticmethod
    def to_centered(x_phys, patch_mean):
        patch_mean = patch_mean.to(dtype=x_phys.dtype, device=x_phys.device)
        return x_phys - patch_mean

    def _mpgn_terms(self, x_phys, y_phys):
        """
        Return exact-MPGN posterior moments and NLL for a chunk.

        Shapes:
            x_phys, y_phys: [B, C, T, H, W]
        """
        dtype = x_phys.dtype
        device = x_phys.device

        alpha = torch.as_tensor(self.alpha, dtype=dtype, device=device)
        beta = torch.clamp(
            torch.as_tensor(self.beta, dtype=dtype, device=device),
            min=self.eps,
        )
        offset = torch.as_tensor(self.offset, dtype=dtype, device=device)

        signal = torch.clamp(x_phys - offset, min=self.eps)
        lam = torch.clamp(signal / alpha, min=self.eps)

        k = torch.arange(
            self.kmax + 1,
            dtype=dtype,
            device=device,
        ).view(1, 1, self.kmax + 1, 1, 1, 1)

        lam_e = lam.unsqueeze(2)
        y_e = y_phys.unsqueeze(2)

        mu_k = offset + alpha * k

        log_pois = (
            k * torch.log(lam_e)
            - lam_e
            - torch.lgamma(k + 1.0)
        )

        log_gauss = -0.5 * (
            ((y_e - mu_k) ** 2) / beta
            + torch.log(
                2.0
                * torch.as_tensor(math.pi, dtype=dtype, device=device)
                * beta
            )
        )

        log_terms = log_pois + log_gauss
        logp = torch.logsumexp(log_terms, dim=2)
        resp = torch.softmax(log_terms, dim=2)

        mean_k = (resp * k).sum(dim=2)
        second_k = (resp * k * k).sum(dim=2)
        var_k = torch.clamp(second_k - mean_k * mean_k, min=0.0)

        nll = -logp

        return lam, mean_k, var_k, nll

    def _correct_chunk(self, x_ref_phys, y_phys, rho):
        dtype = x_ref_phys.dtype
        device = x_ref_phys.device

        alpha = torch.as_tensor(self.alpha, dtype=dtype, device=device)
        beta = torch.clamp(
            torch.as_tensor(self.beta, dtype=dtype, device=device),
            min=self.eps,
        )
        offset = torch.as_tensor(self.offset, dtype=dtype, device=device)
        eps = torch.as_tensor(self.eps, dtype=dtype, device=device)

        rho = torch.clamp(
            rho.to(dtype=dtype, device=device),
            min=self.rho_min,
        )

        signal_ref = torch.clamp(x_ref_phys - offset, min=0.0)
        v_ref = torch.clamp(alpha * signal_ref + beta, min=eps)

        x = x_ref_phys

        lam0, mean_k0, var_k0, nll_before = self._mpgn_terms(
            x_ref_phys,
            y_phys,
        )

        last_step = torch.zeros_like(x_ref_phys)

        for _ in range(max(1, self.n_iter)):
            lam, mean_k, var_k, _ = self._mpgn_terms(x, y_phys)

            g_data = (1.0 - mean_k / torch.clamp(lam, min=eps)) / alpha

            h_data = (
                (mean_k - var_k)
                / torch.clamp(alpha * alpha * lam * lam, min=eps)
            )
            h_data = torch.clamp(h_data, min=0.0)

            g_prox = rho * (x - x_ref_phys) / v_ref
            h_prox = rho / v_ref

            step = (g_data + g_prox) / torch.clamp(h_data + h_prox, min=eps)

            max_step = self.step_cap_sigma * torch.sqrt(v_ref)
            step = torch.clamp(step, min=-max_step, max=max_step)

            x = torch.clamp(x - step, min=offset)
            last_step = step

        _, _, _, nll_after = self._mpgn_terms(x, y_phys)

        prox_after = 0.5 * rho * (x - x_ref_phys) ** 2 / v_ref
        obj_before = nll_before
        obj_after = nll_after + prox_after

        return x, {
            "nll_before": nll_before,
            "nll_after": nll_after,
            "prox_objective_before": obj_before,
            "prox_objective_after": obj_after,
            "step": last_step,
            "v_ref": v_ref,
        }

    def forward(
        self,
        x_ref_centered,
        y_centered,
        patch_mean,
        rho,
        return_diagnostics=False,
    ):
        x_ref_phys = self.to_phys(
            x_ref_centered,
            patch_mean,
        )
        y_phys = self.to_phys(
            y_centered,
            patch_mean,
        )

        T = x_ref_phys.shape[2]
        chunk_t = max(1, int(self.chunk_t))

        z_chunks = []
        diag_chunks = []

        for t0 in range(0, T, chunk_t):
            t1 = min(t0 + chunk_t, T)

            z_chunk, diag_chunk = self._correct_chunk(
                x_ref_phys[:, :, t0:t1],
                y_phys[:, :, t0:t1],
                rho,
            )

            z_chunks.append(z_chunk)

            if return_diagnostics:
                diag_chunks.append(diag_chunk)

        z_phys = torch.cat(z_chunks, dim=2)

        z_centered = self.to_centered(
            z_phys,
            patch_mean,
        )

        if not return_diagnostics:
            return z_centered

        with torch.no_grad():
            def batch_mean(v):
                return v.detach().float().flatten(1).mean(dim=1)

            def batch_std(v):
                return v.detach().float().flatten(1).std(dim=1, unbiased=False)

            def batch_fraction(m):
                return m.detach().float().flatten(1).mean(dim=1)

            nll_before = torch.cat(
                [d["nll_before"] for d in diag_chunks],
                dim=2,
            )
            nll_after = torch.cat(
                [d["nll_after"] for d in diag_chunks],
                dim=2,
            )
            obj_before = torch.cat(
                [d["prox_objective_before"] for d in diag_chunks],
                dim=2,
            )
            obj_after = torch.cat(
                [d["prox_objective_after"] for d in diag_chunks],
                dim=2,
            )
            step = torch.cat(
                [d["step"] for d in diag_chunks],
                dim=2,
            )
            v_ref = torch.cat(
                [d["v_ref"] for d in diag_chunks],
                dim=2,
            )

            finite_mask = (
                torch.isfinite(z_phys)
                & torch.isfinite(nll_after)
                & torch.isfinite(obj_after)
                & torch.isfinite(step)
            )

            rho_batch = rho.expand(
                z_phys.shape[0], 1, 1, 1, 1
            ).reshape(z_phys.shape[0], -1).mean(dim=1)

            diagnostics = {
                "rho": rho_batch,
                "negative_y_fraction": batch_fraction(
                    y_phys < self.offset
                ),
                "x_ref_floor_fraction": batch_fraction(
                    x_ref_phys <= self.offset
                ),
                "z_floor_fraction": batch_fraction(
                    z_phys <= self.offset
                ),
                "v_ref_mean": batch_mean(v_ref),
                "v_ref_std": batch_std(v_ref),

                "nll_before": batch_mean(nll_before),
                "nll_after": batch_mean(nll_after),
                "prox_objective_before": batch_mean(obj_before),
                "prox_objective_after": batch_mean(obj_after),

                "step_mean": batch_mean(step),
                "step_std": batch_std(step),
                "step_abs_mean": batch_mean(torch.abs(step)),
                "step_abs_p95": torch.quantile(
                    torch.abs(step).detach().float().flatten(1),
                    0.95,
                    dim=1,
                ),
                "delta_corr_mae": batch_mean(
                    torch.abs(z_phys - x_ref_phys)
                ),
                "delta_corr_std": batch_std(
                    z_phys - x_ref_phys
                ),
                "finite_fraction": batch_fraction(finite_mask),
            }

        return z_centered, diagnostics


class Network_SRDTrans_Unroll_Transformer(nn.Module):
    """
    End-to-end exact-MPGN proximal unfolding with a shared denoiser.

    x_0 = Pi_+(D_theta(y_masked))

    for k = 1..K:
        z_k = C_exactMPGN(x_{k-1}, y_obs, rho_k, Kmax)
        x_k = Pi_+(D_theta(z_k))

    Final output x_K is the prior/denoiser output used for the training loss.
    """

    def __init__(
        self,
        srdtrans_root,
        img_dim,
        img_time,
        prior_backbone='srdtrans_v2',
        unet_f_maps=16,
        embedding_dim=128,
        num_heads=8,
        hidden_dim=512,
        window_size=7,
        num_transBlock=1,
        attn_dropout_rate=0.1,
        f_maps=(8, 16, 32, 64),
        input_dropout_rate=0.0,
        unroll_steps=2,
        mpgn_alpha=5000.0,
        mpgn_beta=1600.0,
        mpgn_offset=0.0,
        mpgn_kmax=32,
        unroll_rho_sched_min=1.0,
        unroll_rho_sched_max=10.0,
        unroll_rho_min=1e-6,
        unroll_eps=1e-8,
        unroll_corr_n_iter=2,
        unroll_corr_chunk_t=8,
        unroll_corr_step_cap_sigma=1.0,
        unroll_gradient_checkpointing=False,
    ):
        super().__init__()

        self.prior_backbone = str(
            prior_backbone
        ).lower()

        if self.prior_backbone not in ('unet', 'srdtrans_v2'):
            raise ValueError(
                'prior_backbone must be "unet" or "srdtrans_v2", '
                'got {!r}.'.format(prior_backbone)
            )

        self.unroll_steps = int(unroll_steps)
        self.eps = float(unroll_eps)
        self.use_gradient_checkpointing = bool(
            unroll_gradient_checkpointing
        )

        if self.unroll_steps < 1:
            raise ValueError('unroll_steps must be >= 1.')
        if not 0 < float(unroll_rho_sched_min) <= float(unroll_rho_sched_max):
            raise ValueError(
                'Require 0 < unroll_rho_sched_min <= unroll_rho_sched_max. '
                'These arguments are interpreted as exact-MPGN prox rho bounds.'
            )

        self.register_buffer(
            'mpgn_alpha_buffer',
            torch.tensor(float(mpgn_alpha), dtype=torch.float32),
        )
        self.register_buffer(
            'mpgn_beta_buffer',
            torch.tensor(float(mpgn_beta), dtype=torch.float32),
        )
        self.register_buffer(
            'mpgn_offset_buffer',
            torch.tensor(float(mpgn_offset), dtype=torch.float32),
        )

        self.mpgn_kmax = int(mpgn_kmax)
        self.unroll_corr_n_iter = int(unroll_corr_n_iter)
        self.unroll_corr_chunk_t = int(unroll_corr_chunk_t)
        self.unroll_corr_step_cap_sigma = float(unroll_corr_step_cap_sigma)

        rho_schedule = torch.logspace(
            math.log10(float(unroll_rho_sched_min)),
            math.log10(float(unroll_rho_sched_max)),
            steps=int(unroll_steps),
            dtype=torch.float32,
        )
        self.register_buffer(
            'rho_schedule',
            rho_schedule.view(1, int(unroll_steps), 1, 1, 1),
        )

        if self.prior_backbone == 'unet':
            self.PriorNet = UNet3D(
                in_channels=1,
                out_channels=1,
                f_maps=int(unet_f_maps),
                final_sigmoid=True,
            )
            self.PriorNet.final_activation = nn.Identity()

        else:
            if srdtrans_root is not None:
                root = os.path.abspath(srdtrans_root)

                if (
                    os.path.basename(root) == 'SRDTrans_v2'
                    and os.path.isfile(
                        os.path.join(root, '__init__.py')
                    )
                ):
                    sys_path_root = os.path.dirname(root)
                else:
                    sys_path_root = root

                if sys_path_root not in sys.path:
                    sys.path.insert(0, sys_path_root)

            try:
                from SRDTrans_v2 import SRDTrans_v2
            except ImportError as exc:
                raise ImportError(
                    'Could not import SRDTrans_v2. '
                    'Set --srdtrans-root to the repository root '
                    'containing the SRDTrans_v2 package.'
                ) from exc

            self.PriorNet = SRDTrans_v2(
                img_dim=int(img_dim),
                img_time=int(img_time),
                in_channel=1,
                embedding_dim=int(embedding_dim),
                num_heads=int(num_heads),
                hidden_dim=int(hidden_dim),
                window_size=int(window_size),
                num_transBlock=int(num_transBlock),
                attn_dropout_rate=float(attn_dropout_rate),
                f_maps=list(f_maps),
                input_dropout_rate=float(input_dropout_rate),
            )

        print(
            '\033[1;31mEnd-to-end exact-MPGN proximal unfolding prior={} '
            'K={} rho=[{:.3g},{:.3g}] kmax={} corr_iter={} unet_f_maps={}\033[0m'.format(
                self.prior_backbone,
                self.unroll_steps,
                float(unroll_rho_sched_min),
                float(unroll_rho_sched_max),
                int(mpgn_kmax),
                int(unroll_corr_n_iter),
                int(unet_f_maps),
            )
        )

        self.correction = ExactMPGNProxCorrection(
            alpha=mpgn_alpha,
            beta=mpgn_beta,
            offset=mpgn_offset,
            rho_min=unroll_rho_min,
            eps=unroll_eps,
            kmax=int(mpgn_kmax),
            n_iter=int(unroll_corr_n_iter),
            chunk_t=int(unroll_corr_chunk_t),
            step_cap_sigma=float(unroll_corr_step_cap_sigma),
        )

    @property
    def mpgn_alpha(self):
        return float(self.mpgn_alpha_buffer.item())

    @property
    def mpgn_beta(self):
        return float(self.mpgn_beta_buffer.item())

    @property
    def mpgn_offset(self):
        return float(self.mpgn_offset_buffer.item())

    def _physical_positive_centered(self, x_centered, patch_mean):
        """Map centered output to a physically non-negative signal domain."""
        dtype = x_centered.dtype
        device = x_centered.device

        offset = self.mpgn_offset_buffer.to(dtype=dtype, device=device)
        beta = self.mpgn_beta_buffer.to(dtype=dtype, device=device)

        softness = torch.sqrt(beta.clamp_min(self.eps))

        x_phys = self.correction.to_phys(
            x_centered,
            patch_mean,
        )
        signal_raw = x_phys - offset

        signal_pos = softness * F.softplus(
            signal_raw / softness.clamp_min(self.eps)
        )

        x_phys_pos = offset + signal_pos
        return self.correction.to_centered(
            x_phys_pos,
            patch_mean,
        )

    def _run_prior(self, x):
        if (
            self.training
            and self.use_gradient_checkpointing
            and x.requires_grad
        ):
            try:
                return checkpoint(
                    self.PriorNet,
                    x,
                    use_reentrant=False,
                )
            except TypeError:
                return checkpoint(
                    self.PriorNet,
                    x,
                )
        return self.PriorNet(x)

    def _relative_iterate_residual(self, x_new, x_old):
        eps = torch.as_tensor(
            self.eps,
            dtype=x_new.dtype,
            device=x_new.device,
        )
        num = (x_new - x_old).flatten(1).norm(dim=1)
        denom = x_old.flatten(1).norm(dim=1).clamp_min(eps)
        return num / denom

    def forward(
        self,
        y_masked,
        patch_mean,
        y_obs=None,
        return_debug=False,
        return_last_prediction=False,
    ):
        """
        End-to-end exact-MPGN proximal unfolding.

        Training:
            y_masked = masked / neighbor-replaced observation
            y_obs    = y_masked

        Validation / inference:
            y_masked = raw observation
            y_obs    = raw observation

        Output:
            x_state = final prior output x_K
        """
        if y_obs is None:
            y_obs = y_masked

        rho_all = self.rho_schedule.to(
            dtype=y_masked.dtype,
            device=y_masked.device,
        )

        debug = {
            'y_obs': y_obs.detach(),
            'y_masked': y_masked.detach(),
            'prior_backbone': self.prior_backbone,
            'rho_all': rho_all.detach(),
            'stages': [],
            'loss_output_name': 'x_K_masked_exact_mpgn',
        }

        # Prior warm-start.
        x0_pred = self._run_prior(y_masked)
        x_state = self._physical_positive_centered(
            x0_pred,
            patch_mean=patch_mean,
        )

        if return_debug:
            debug['x0_pred'] = x0_pred.detach()
            debug['x0_prior'] = x_state.detach()

        last_z_corr = None
        residuals = []

        for k in range(self.unroll_steps):
            x_prev = x_state
            rho_k = rho_all[:, k:k + 1]

            if return_debug:
                z_corr, corr_diag = self.correction(
                    x_ref_centered=x_prev,
                    y_centered=y_obs,
                    patch_mean=patch_mean,
                    rho=rho_k,
                    return_diagnostics=True,
                )
            else:
                z_corr = self.correction(
                    x_ref_centered=x_prev,
                    y_centered=y_obs,
                    patch_mean=patch_mean,
                    rho=rho_k,
                    return_diagnostics=False,
                )
                corr_diag = None

            x_pred = self._run_prior(z_corr)

            x_prior = self._physical_positive_centered(
                x_pred,
                patch_mean=patch_mean,
            )

            residual_k = self._relative_iterate_residual(
                x_prior,
                x_prev,
            )
            residuals.append(residual_k.detach())

            if return_debug:
                debug['stages'].append({
                    'stage_idx': k,
                    'rho': rho_k.detach(),

                    'x_prev': x_prev.detach(),
                    'z_corr': z_corr.detach(),
                    'x_pred': x_pred.detach(),
                    'x_prior': x_prior.detach(),

                    'delta_correction': (z_corr - x_prev).detach(),
                    'delta_prior': (x_prior - z_corr).detach(),
                    'delta_to_y': (x_prior - y_obs).detach(),
                    'iterate_delta': (x_prior - x_prev).detach(),
                    'iterate_residual': residual_k.detach(),

                    'corr_diag': corr_diag,

                    'output_for_loss': (
                        x_prior.detach()
                        if k == self.unroll_steps - 1
                        else None
                    ),
                })

            x_state = x_prior
            last_z_corr = z_corr

        residual_tensor = torch.stack(residuals, dim=1)

        diagnostics = {
            'iterate_residuals': residual_tensor.detach(),
            'rho': rho_all.detach(),
        }

        if return_debug:
            debug['convergence'] = diagnostics
            return x_state, debug

        if return_last_prediction:
            # Return data-corrected state and final prior output.
            return last_z_corr, x_state

        return x_state

    @torch.no_grad()
    def forward_data_and_prior(self, y_masked, patch_mean=None, y_obs=None):
        """
        Returns:
            z_K: final exact-MPGN correction state
            x_K: final prior output, main restoration output
        """
        return self.forward(
            y_masked,
            patch_mean=patch_mean,
            y_obs=y_obs,
            return_last_prediction=True,
        )
