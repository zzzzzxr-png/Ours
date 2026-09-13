"""Training loss helpers: L1+L2 and MPGN NLL (optional quantization / clipping)."""

import math

import torch


def l1_l2_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    l1_weight: float = 0.5,
    l2_weight: float = 0.5,
) -> torch.Tensor:
    """Unmasked 0.5*L1 + 0.5*L2 (DeepCAD / SRDTrans default reconstruction loss)."""
    if output.shape != target.shape:
        raise ValueError(
            'output and target shapes must match, got {} and {}'.format(
                tuple(output.shape), tuple(target.shape)
            )
        )
    diff = output - target
    l1 = diff.abs().mean()
    l2 = diff.pow(2).mean()
    return float(l1_weight) * l1 + float(l2_weight) * l2


def masked_l1_l2_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    l1_weight: float = 0.5,
    l2_weight: float = 0.5,
) -> torch.Tensor:
    """L1+L2 loss evaluated only at masked positions."""
    if output.shape != target.shape:
        raise ValueError(
            'output and target shapes must match, got {} and {}'.format(
                tuple(output.shape), tuple(target.shape)
            )
        )
    mask_f = mask.to(dtype=output.dtype).expand_as(output)
    denom = mask_f.sum().clamp_min(1.0)
    diff = output - target
    l1 = (diff.abs() * mask_f).sum() / denom
    l2 = (diff.pow(2) * mask_f).sum() / denom
    return float(l1_weight) * l1 + float(l2_weight) * l2


def _to_phys_units(x, img_mean):
    """Convert centered tensor back to acquisition/physical units."""
    if torch.is_tensor(img_mean):
        img_mean = img_mean.to(dtype=x.dtype, device=x.device)
    else:
        img_mean = torch.as_tensor(img_mean, dtype=x.dtype, device=x.device)

    return x + img_mean


def _log_ndtr(x):
    """Stable log Phi(x)."""
    if hasattr(torch.special, 'log_ndtr'):
        return torch.special.log_ndtr(x)

    # Fallback. Less stable in extreme tails, but works for moderate ranges.
    cdf = 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    return torch.log(cdf.clamp_min(torch.finfo(x.dtype).tiny))


def _normal_log_interval_prob(lower, upper, mean, std):
    """
    log P(lower <= N(mean,std^2) <= upper).

    lower/upper may contain +/-inf.
    Broadcast shape should match mean.
    """
    a = (lower - mean) / std
    b = (upper - mean) / std

    logPhi_a = _log_ndtr(a)
    logPhi_b = _log_ndtr(b)

    # finite interval: log(Phi(b)-Phi(a))
    # assume b >= a
    eps = torch.finfo(mean.dtype).eps
    ratio = torch.exp(logPhi_a - logPhi_b).clamp(max=1.0 - eps)
    log_interval = logPhi_b + torch.log1p(-ratio)

    # lower = -inf: log Phi(b)
    lower_inf = torch.isneginf(lower)
    log_interval = torch.where(lower_inf, logPhi_b, log_interval)

    # upper = +inf: log(1-Phi(a)) = log Phi(-a)
    upper_inf = torch.isposinf(upper)
    log_upper_tail = _log_ndtr(-a)
    log_interval = torch.where(upper_inf, log_upper_tail, log_interval)

    return log_interval


def _mpgn_log_gauss_observation(
    target_phys,
    mu_k,
    beta_t,
    *,
    quant_step=1.0,
    clip_low=None,
    clip_high=None,
    boundary_atol=1e-6,
):
    """
    Return log P(observed target_phys | K=k).

    If quant_step is not None:
        quantized likelihood:
            interior: P(y-q/2 <= Y < y+q/2)
            lower clip: P(Y <= L+q/2)
            upper clip: P(Y >= U-q/2)

    If quant_step is None:
        continuous likelihood:
            interior: Gaussian density at y
            lower clip: P(Y <= L)
            upper clip: P(Y >= U)
    """
    dtype = target_phys.dtype
    device = target_phys.device

    std = torch.sqrt(beta_t)

    y_e = target_phys.unsqueeze(2)

    if quant_step is None:
        # Continuous density for non-clipped interior pixels.
        log_density = -0.5 * (
            ((y_e - mu_k) ** 2) / beta_t
            + torch.log(
                2.0
                * torch.as_tensor(math.pi, dtype=dtype, device=device)
                * beta_t
            )
        )

        if clip_low is None and clip_high is None:
            return log_density

        logp = log_density

        if clip_low is not None:
            L = torch.as_tensor(float(clip_low), dtype=dtype, device=device)
            low_mask = target_phys <= L + float(boundary_atol)
            lower = torch.full_like(y_e, -float('inf'))
            upper = torch.full_like(y_e, float(clip_low))
            log_low = _normal_log_interval_prob(lower, upper, mu_k, std)
            logp = torch.where(low_mask.unsqueeze(2), log_low, logp)

        if clip_high is not None:
            U = torch.as_tensor(float(clip_high), dtype=dtype, device=device)
            high_mask = target_phys >= U - float(boundary_atol)
            lower = torch.full_like(y_e, float(clip_high))
            upper = torch.full_like(y_e, float('inf'))
            log_high = _normal_log_interval_prob(lower, upper, mu_k, std)
            logp = torch.where(high_mask.unsqueeze(2), log_high, logp)

        return logp

    # Quantized likelihood.
    q = float(quant_step)
    half_q = 0.5 * q

    lower = y_e - half_q
    upper = y_e + half_q

    if clip_low is not None:
        L = torch.as_tensor(float(clip_low), dtype=dtype, device=device)
        low_mask = target_phys <= L + float(boundary_atol)
        lower = torch.where(
            low_mask.unsqueeze(2),
            torch.full_like(lower, -float('inf')),
            lower,
        )
        upper = torch.where(
            low_mask.unsqueeze(2),
            torch.full_like(upper, float(clip_low) + half_q),
            upper,
        )

    if clip_high is not None:
        U = torch.as_tensor(float(clip_high), dtype=dtype, device=device)
        high_mask = target_phys >= U - float(boundary_atol)
        lower = torch.where(
            high_mask.unsqueeze(2),
            torch.full_like(lower, float(clip_high) - half_q),
            lower,
        )
        upper = torch.where(
            high_mask.unsqueeze(2),
            torch.full_like(upper, float('inf')),
            upper,
        )

    return _normal_log_interval_prob(lower, upper, mu_k, std)


def mpgn_nll_single_target(
    pred_tensor,
    target_tensor,
    valid_mask,
    pred_img_mean,
    target_img_mean,
    alpha,
    beta,
    offset=0.0,
    kmax=32,
    min_signal=1e-8,
    chunk_t=8,
    quant_step=1.0,
    clip_low=None,
    clip_high=None,
    boundary_atol=1e-6,
):
    """
    Exact MPGN marginal NLL with optional quantization and clipping.

    Model:
        K ~ Poisson((X-offset)/alpha)
        Y | K=k ~ N(offset + alpha*k, beta)

    If quant_step is not None:
        Uses quantized bin probability.

    If clip_low / clip_high are provided:
        Boundary observations are treated as censored:
            y == L -> Y <= L + q/2
            y == U -> Y >= U - q/2

    If clip_low and clip_high are both None:
        No clipping is used.

    If quant_step is None and clip bounds are None:
        This reduces to the original continuous exact MPGN density.
    """
    dtype = pred_tensor.dtype
    device = pred_tensor.device

    alpha_t = torch.as_tensor(alpha, dtype=dtype, device=device)
    beta_t = torch.clamp(
        torch.as_tensor(beta, dtype=dtype, device=device),
        min=1e-12,
    )
    offset_t = torch.as_tensor(offset, dtype=dtype, device=device)

    k = torch.arange(
        kmax + 1,
        dtype=dtype,
        device=device,
    ).view(1, 1, kmax + 1, 1, 1, 1)

    mu_k = offset_t + alpha_t * k

    total_nll = pred_tensor.new_zeros(())
    total_count = pred_tensor.new_zeros(())

    temporal_size = pred_tensor.shape[2]
    chunk_t = max(1, int(chunk_t))

    for t0 in range(0, temporal_size, chunk_t):
        t1 = min(t0 + chunk_t, temporal_size)

        pred_chunk = pred_tensor[:, :, t0:t1]
        target_chunk = target_tensor[:, :, t0:t1]
        mask_chunk = valid_mask[:, :, t0:t1]

        pred_phys = _to_phys_units(
            pred_chunk,
            pred_img_mean,
        )
        target_phys = _to_phys_units(
            target_chunk,
            target_img_mean,
        )

        signal = torch.clamp(
            pred_phys - offset_t,
            min=float(min_signal),
        )

        lam = torch.clamp(
            signal / alpha_t,
            min=1e-12,
        )

        lam_e = lam.unsqueeze(2)

        log_pois = (
            k * torch.log(lam_e)
            - lam_e
            - torch.lgamma(k + 1.0)
        )

        log_obs_given_k = _mpgn_log_gauss_observation(
            target_phys,
            mu_k,
            beta_t,
            quant_step=quant_step,
            clip_low=clip_low,
            clip_high=clip_high,
            boundary_atol=boundary_atol,
        )

        logp = torch.logsumexp(
            log_pois + log_obs_given_k,
            dim=2,
        )

        nll = -logp

        vm = mask_chunk.to(dtype)

        # Robust count under broadcasting.
        # If valid_mask is [B,1,T,H,W] and nll is [B,C,T,H,W],
        # this counts all valid channel elements correctly.
        weight = torch.ones_like(nll) * vm
        total_nll = total_nll + (nll * weight).sum()
        total_count = total_count + weight.sum()

    return total_nll / total_count.clamp_min(1.0)


def mpgn_predictive_mean_var(
    x,
    alpha,
    beta,
    offset=0.0,
    kmax=32,
    min_signal=1e-8,
):
    """E[Y|X], Var(Y|X) under the same truncated-K mixture as mpgn_nll_single_target.

    K ∈ {0,…,kmax} with Poisson(λ) weights, λ=(X−offset)/α, renormalized by
    logsumexp (identical to the NLL). Y|K=k ~ N(offset+αk, β).

    These are moments of that latent mixture. Quantization/clipping change
    p(observed y|X), not this predictive second moment.
    """
    x_t = torch.as_tensor(x, dtype=torch.float64)
    alpha_t = torch.as_tensor(float(alpha), dtype=torch.float64)
    beta_t = torch.clamp(torch.as_tensor(float(beta), dtype=torch.float64), min=1e-12)
    offset_t = torch.as_tensor(float(offset), dtype=torch.float64)

    signal = torch.clamp(x_t - offset_t, min=float(min_signal))
    lam = torch.clamp(signal / alpha_t, min=1e-12)
    k = torch.arange(int(kmax) + 1, dtype=torch.float64)
    log_pois = k * torch.log(lam.unsqueeze(-1)) - lam.unsqueeze(-1) - torch.lgamma(k + 1.0)
    log_pi = log_pois - torch.logsumexp(log_pois, dim=-1, keepdim=True)
    pi = torch.exp(log_pi)
    mu_k = offset_t + alpha_t * k
    mean = (pi * mu_k).sum(dim=-1)
    var = beta_t + (pi * (mu_k - mean.unsqueeze(-1)).square()).sum(dim=-1)
    return mean.cpu().numpy(), var.cpu().numpy()


# Backward-compatible alias
_mpgn_nll_single_target = mpgn_nll_single_target
