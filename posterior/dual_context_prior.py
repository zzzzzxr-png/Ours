"""Dual axial replacement + context-dependent Gamma prior + moment fusion.

Replacement axis is selected by sampling_mode (option B):
  height_mask   -> H ± 1
  width_mask    -> W ± 1
  temporal_mask -> T ± 1

No circular / reflection padding: invalid border branches get weight 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


_AXIS_FROM_MODE = {
    'height_mask': 'h',
    'width_mask': 'w',
    'temporal_mask': 't',
}


def dual_axis_from_sampling_mode(sampling_mode: str) -> str:
    """Map sampling_mode to dual-replacement axis {'h','w','t'}."""
    if sampling_mode == 'spatial_mask':
        raise ValueError(
            "spatial_mask is split into height_mask and width_mask; "
            "choose one explicitly."
        )
    axis = _AXIS_FROM_MODE.get(sampling_mode)
    if axis is None:
        raise ValueError(
            "dual-context prior requires sampling_mode in "
            "{height_mask, width_mask, temporal_mask}, got {!r}".format(
                sampling_mode
            )
        )
    return axis


@dataclass
class GammaPrior:
    mu_lambda: torch.Tensor
    var_lambda: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor
    kappa: torch.Tensor


def make_dual_axis_replacements(
    noisy: torch.Tensor,
    loss_mask: torch.Tensor,
    axis: str,
):
    """Create axis-1 and axis+1 replacements for the same masked positions.

    Args:
        noisy: [B, C, T, H, W]
        loss_mask: bool [B, 1, T, H, W]
        axis: 'h', 'w', or 't'

    Returns:
        input_minus, input_plus, valid_minus, valid_plus
    """
    if noisy.ndim != 5:
        raise ValueError(
            'noisy must be [B,C,T,H,W], got {}'.format(tuple(noisy.shape))
        )
    if loss_mask.ndim != 5 or loss_mask.shape[1] != 1:
        raise ValueError(
            'loss_mask must be [B,1,T,H,W], got {}'.format(
                tuple(loss_mask.shape)
            )
        )
    if noisy.shape[0] != loss_mask.shape[0] or noisy.shape[2:] != loss_mask.shape[2:]:
        raise ValueError('noisy and loss_mask shapes do not match')

    axis = axis.lower()
    if axis not in ('h', 'w', 't'):
        raise ValueError("axis must be 'h', 'w', or 't', got {!r}".format(axis))

    mask = loss_mask.bool()
    neighbor_minus = torch.zeros_like(noisy)
    neighbor_plus = torch.zeros_like(noisy)
    valid_minus = torch.zeros_like(mask)
    valid_plus = torch.zeros_like(mask)

    if axis == 'h':
        # dim=3
        neighbor_minus[:, :, :, 1:, :] = noisy[:, :, :, :-1, :]
        neighbor_plus[:, :, :, :-1, :] = noisy[:, :, :, 1:, :]
        valid_minus[:, :, :, 1:, :] = True
        valid_plus[:, :, :, :-1, :] = True
    elif axis == 'w':
        # dim=4
        neighbor_minus[:, :, :, :, 1:] = noisy[:, :, :, :, :-1]
        neighbor_plus[:, :, :, :, :-1] = noisy[:, :, :, :, 1:]
        valid_minus[:, :, :, :, 1:] = True
        valid_plus[:, :, :, :, :-1] = True
    else:
        # dim=2
        neighbor_minus[:, :, 1:, :, :] = noisy[:, :, :-1, :, :]
        neighbor_plus[:, :, :-1, :, :] = noisy[:, :, 1:, :, :]
        valid_minus[:, :, 1:, :, :] = True
        valid_plus[:, :, :-1, :, :] = True

    replace_minus = (mask & valid_minus).expand_as(noisy)
    replace_plus = (mask & valid_plus).expand_as(noisy)

    input_minus = torch.where(replace_minus, neighbor_minus, noisy)
    input_plus = torch.where(replace_plus, neighbor_plus, noisy)

    if torch.any(mask & ~(valid_minus | valid_plus)):
        raise RuntimeError(
            'A masked position has no valid {}-direction replacement.'.format(
                axis
            )
        )

    return input_minus, input_plus, valid_minus, valid_plus


def decode_gamma_variant(
    model_output: torch.Tensor,
    img_mean,
    alpha,
    offset,
    *,
    kappa_mode: str = 'learned_map',
    fixed_kappa: float = 50.0,
    kappa_min: float = 1e-4,
    min_lambda: float = 1e-12,
):
    """Decode one replacement branch into Gamma prior moments.

    learned_map: output [B,2,T,H,W] -> ch0=mu_x_centered, ch1=raw_kappa
    fixed:       output [B,1,T,H,W] -> mu_x_centered, kappa=fixed_kappa
    """
    if model_output.ndim != 5:
        raise ValueError(
            'model output must be [B,C,T,H,W], got {}'.format(
                tuple(model_output.shape)
            )
        )

    dtype = model_output.dtype
    device = model_output.device
    img_mean_t = torch.as_tensor(img_mean, dtype=dtype, device=device)
    alpha_t = torch.as_tensor(alpha, dtype=dtype, device=device)
    offset_t = torch.as_tensor(offset, dtype=dtype, device=device)

    if kappa_mode == 'learned_map':
        if model_output.shape[1] != 2:
            raise ValueError(
                'learned_map expects [B,2,T,H,W], got {}'.format(
                    tuple(model_output.shape)
                )
            )
        mu_x_centered = model_output[:, 0:1]
        raw_kappa = model_output[:, 1:2]
        kappa = torch.exp(raw_kappa) + float(kappa_min)
    elif kappa_mode == 'fixed':
        if model_output.shape[1] != 1:
            raise ValueError(
                'fixed kappa expects [B,1,T,H,W], got {}'.format(
                    tuple(model_output.shape)
                )
            )
        mu_x_centered = model_output[:, 0:1]
        raw_kappa = None
        kappa = torch.full_like(mu_x_centered, float(fixed_kappa))
    else:
        raise ValueError('Unknown kappa_mode: {!r}'.format(kappa_mode))

    mu_x_phys = mu_x_centered + img_mean_t
    mu_lambda = torch.clamp(
        (mu_x_phys - offset_t) / alpha_t,
        min=float(min_lambda),
    )
    var_lambda = mu_lambda / kappa

    return {
        'mu_x_centered': mu_x_centered,
        'mu_x_phys': mu_x_phys,
        'raw_kappa': raw_kappa,
        'mu_lambda': mu_lambda,
        'kappa': kappa,
        'var_lambda': var_lambda,
    }


def fuse_two_gamma_priors(
    minus_prior,
    plus_prior,
    valid_minus: torch.Tensor,
    valid_plus: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    min_variance: float = 1e-12,
    min_mean: float = 1e-12,
) -> GammaPrior:
    """Equal-weight moment matching of two Gamma priors (invalid weight=0)."""
    dtype = minus_prior['mu_lambda'].dtype

    vm = valid_minus.to(dtype=dtype)
    vp = valid_plus.to(dtype=dtype)
    target_mask = target_mask.bool()

    denom = vm + vp
    if torch.any(target_mask & (denom <= 0)):
        raise RuntimeError(
            'At least one target position has no valid replacement direction.'
        )

    denom_safe = denom.clamp_min(1.0)

    mu_minus = minus_prior['mu_lambda']
    mu_plus = plus_prior['mu_lambda']
    var_minus = minus_prior['var_lambda']
    var_plus = plus_prior['var_lambda']

    mu_fused = (vm * mu_minus + vp * mu_plus) / denom_safe
    second_fused = (
        vm * (var_minus + mu_minus.square())
        + vp * (var_plus + mu_plus.square())
    ) / denom_safe

    var_fused = torch.clamp(
        second_fused - mu_fused.square(),
        min=float(min_variance),
    )
    mu_fused = torch.clamp(mu_fused, min=float(min_mean))

    a_fused = mu_fused.square() / var_fused
    b_fused = mu_fused / var_fused

    if not torch.isfinite(mu_fused).all():
        raise FloatingPointError('Non-finite fused Gamma mean')
    if not torch.isfinite(var_fused).all():
        raise FloatingPointError('Non-finite fused Gamma variance')
    if not torch.isfinite(a_fused).all():
        raise FloatingPointError('Non-finite fused Gamma shape')
    if not torch.isfinite(b_fused).all():
        raise FloatingPointError('Non-finite fused Gamma rate')

    return GammaPrior(
        mu_lambda=mu_fused,
        var_lambda=var_fused,
        a=a_fused,
        b=b_fused,
        kappa=b_fused,
    )


def dual_axis_context_prior(
    model,
    noisy: torch.Tensor,
    loss_mask: torch.Tensor,
    img_mean,
    alpha,
    offset,
    *,
    axis: str,
    kappa_mode: str = 'learned_map',
    fixed_kappa: float = 50.0,
    kappa_min: float = 1e-4,
    min_lambda: float = 1e-12,
    min_variance: float = 1e-12,
):
    """Run both replacement branches in one parallel model call (batch 2B)."""
    input_minus, input_plus, valid_minus, valid_plus = make_dual_axis_replacements(
        noisy, loss_mask, axis=axis
    )

    dual_input = torch.cat([input_minus, input_plus], dim=0)
    dual_output = model(dual_input)

    if dual_output.shape[0] != 2 * noisy.shape[0]:
        raise RuntimeError(
            'Unexpected dual-output batch size: {} vs {}'.format(
                dual_output.shape[0], 2 * noisy.shape[0]
            )
        )

    output_minus, output_plus = dual_output.chunk(2, dim=0)

    decode_kw = dict(
        img_mean=img_mean,
        alpha=alpha,
        offset=offset,
        kappa_mode=kappa_mode,
        fixed_kappa=fixed_kappa,
        kappa_min=kappa_min,
        min_lambda=min_lambda,
    )
    minus_prior = decode_gamma_variant(output_minus, **decode_kw)
    plus_prior = decode_gamma_variant(output_plus, **decode_kw)

    fused_prior = fuse_two_gamma_priors(
        minus_prior,
        plus_prior,
        valid_minus,
        valid_plus,
        loss_mask,
        min_variance=min_variance,
        min_mean=min_lambda,
    )

    return {
        'input_minus': input_minus,
        'input_plus': input_plus,
        'valid_minus': valid_minus,
        'valid_plus': valid_plus,
        'minus_prior': minus_prior,
        'plus_prior': plus_prior,
        'fused_prior': fused_prior,
    }


def dual_h_context_prior(model, noisy, loss_mask, img_mean, alpha, offset, **kwargs):
    """H-axis convenience wrapper around dual_axis_context_prior."""
    kwargs.setdefault('axis', 'h')
    return dual_axis_context_prior(
        model, noisy, loss_mask, img_mean, alpha, offset, **kwargs
    )


def make_dual_h_replacements(noisy, loss_mask):
    return make_dual_axis_replacements(noisy, loss_mask, axis='h')
