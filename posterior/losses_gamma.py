"""Continuous MPGN/Gamma-NB likelihoods with rigorously adaptive K support."""

import math

import torch

from .gamma_posterior import (
    gamma_ab_from_mu_kappa,
    log_negbinom_pmf,
    count_posterior_log_q,
    posterior_count_moments,
    posterior_mean_x,
    posterior_mean_photon_count,
    posterior_mean_shot_noise,
)


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


def _mpgn_log_gauss_observation(
    target_phys,
    mu_k,
    beta_t,
):
    """Continuous Gaussian readout density log N(Y; offset + alpha*k, beta)."""
    dtype = target_phys.dtype
    device = target_phys.device
    y_e = target_phys.unsqueeze(2)
    return -0.5 * (
        ((y_e - mu_k) ** 2) / beta_t
        + torch.log(
            2.0 * torch.as_tensor(math.pi, dtype=dtype, device=device) * beta_t
        )
    )


@torch.no_grad()
def adaptive_kmax(
    a,
    b,
    y_phys,
    alpha,
    beta,
    offset=0.0,
    *,
    relative_tail_tol=1e-8,
    hard_cap=512,
    step=16,
):
    """Choose a global Kmax with a ratio-test bound on the omitted positive tail.

    The exact support always starts at Kmin=0.  For t_k = NB(k|a,b)N(y|o+αk,β),
    the returned K satisfies tail/sum_[0:K] <= relative_tail_tol for every
    supplied voxel, provided hard_cap is not reached.  The proof uses the exact
    t_(k+1)/t_k ratio and an analytic bound on all later ratios.
    """
    if hard_cap < 0:
        raise ValueError('hard_cap must be non-negative')
    if step < 1:
        raise ValueError('step must be positive')

    # Selection is detached and evaluated in float64 for a reliable bound.
    a = a.detach().reshape(-1).double()
    b = b.detach().reshape(-1).double().clamp_min(torch.finfo(torch.float64).tiny)
    y = y_phys.detach().reshape(-1).double()
    if not (torch.isfinite(a).all() and torch.isfinite(b).all() and torch.isfinite(y).all()):
        raise RuntimeError('nonfinite a, b, or y before adaptive K selection')
    if (a <= 0).any() or (b <= 0).any():
        raise RuntimeError('a and b must be positive before adaptive K selection')

    alpha = float(alpha)
    beta = max(float(beta), 1e-300)
    offset = float(offset)
    log_norm = math.log(2.0 * math.pi * beta)
    if relative_tail_tol <= 0 or not math.isfinite(float(relative_tail_tol)):
        raise ValueError('relative_tail_tol must be positive and finite')
    log_tol = math.log(float(relative_tail_tol))

    def log_term(k):
        k_t = torch.as_tensor(float(k), dtype=torch.float64, device=a.device)
        log_nb = (
            torch.lgamma(a + k_t) - torch.lgamma(a) - torch.lgamma(k_t + 1.0)
            + a * (torch.log(b) - torch.log1p(b))
            + k_t * (-torch.log1p(b))
        )
        d = y - offset - alpha * k_t
        log_readout = -0.5 * (d.square() / beta + log_norm)
        return log_nb + log_readout

    def log_ratio(k):
        # t_(k+1)/t_k, evaluated directly to avoid subtracting large log terms.
        k_t = torch.as_tensor(float(k), dtype=torch.float64, device=a.device)
        d = y - offset - alpha * k_t
        return (
            torch.log(a + k_t) - torch.log(k_t + 1.0) - torch.log1p(b)
            + alpha * d / beta - alpha * alpha / (2.0 * beta)
        )

    # The ratio of consecutive ratios is bounded for all j >= k by this value.
    # This lets us prove that the ratio will remain <= M after the selected K.
    def future_ratio_factor_bound(k):
        k_t = torch.as_tensor(float(k), dtype=torch.float64, device=a.device)
        rational = (a + k_t + 1.0) * (k_t + 1.0) / ((a + k_t) * (k_t + 2.0))
        return math.exp(-alpha * alpha / beta) * torch.maximum(
            rational, torch.ones_like(rational)
        )

    k = 0
    while True:
        k_next = min(k + step, int(hard_cap))
        log_terms = torch.stack([log_term(j) for j in range(k_next + 1)], dim=1)
        log_partial = torch.logsumexp(log_terms, dim=1)
        log_r = log_ratio(k_next)
        factor_bound = future_ratio_factor_bound(k_next)
        ratio_ok = bool((factor_bound <= 1.0).all())
        if ratio_ok:
            max_log_r = float(log_r.max())
            ratio_ok = max_log_r < 0.0
        if ratio_ok:
            # t_(K+1)/(1-M) bounds the entire tail after K.
            M = log_r.exp().max()
            log_tail_bound = log_term(k_next + 1) - torch.log1p(-M)
            if bool((log_tail_bound - log_partial <= log_tol).all()):
                return k_next
        if k_next >= int(hard_cap):
            raise RuntimeError(
                'adaptive K support failed: hard cap {} reached without '
                'relative tail bound <= {}'.format(hard_cap, relative_tail_tol)
            )
        k = k_next


def mpgn_nll_single_target(
    pred_tensor,
    target_tensor,
    valid_mask,
    pred_img_mean,
    target_img_mean,
    alpha,
    beta,
    offset=0.0,
    kmax=512,
    min_signal=1e-8,
    chunk_t=8,
    tail_tol=1e-8,
):
    """
    Exact continuous MPGN marginal NLL with strict adaptive K support.

    Model:
        K ~ Poisson((X-offset)/alpha)
        Y | K=k ~ N(offset + alpha*k, beta)

    The omitted positive series tail is bounded by an exact ratio test.
    """
    dtype = pred_tensor.dtype
    device = pred_tensor.device

    alpha_t = torch.as_tensor(alpha, dtype=dtype, device=device)
    beta_t = torch.clamp(
        torch.as_tensor(beta, dtype=dtype, device=device),
        min=1e-12,
    )
    offset_t = torch.as_tensor(offset, dtype=dtype, device=device)

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

        mu_lambda = torch.clamp(signal / alpha_t, min=1e-12)
        a, b = gamma_ab_from_mu_kappa(
            mu_lambda, torch.as_tensor(1e12, dtype=dtype, device=device)
        )
        k_used = adaptive_kmax(
            a, b, target_phys, alpha, beta, offset,
            relative_tail_tol=tail_tol, hard_cap=int(kmax),
        )
        k = torch.arange(k_used + 1, dtype=dtype, device=device).view(
            1, 1, k_used + 1, 1, 1, 1
        )
        mu_k = offset_t + alpha_t * k
        log_nb = log_negbinom_pmf(k, a.unsqueeze(2), b)
        log_obs_given_k = _mpgn_log_gauss_observation(target_phys, mu_k, beta_t)

        logp = torch.logsumexp(
            log_nb + log_obs_given_k,
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


# Backward-compatible alias
_mpgn_nll_single_target = mpgn_nll_single_target


def gamma_nb_nll_single_target(
    pred_tensor,
    target_tensor,
    valid_mask,
    pred_img_mean,
    target_img_mean,
    alpha,
    beta,
    kappa,
    offset=0.0,
    kmax=32,
    min_signal=1e-8,
    chunk_t=8,
    tail_tol=1e-8,
):
    """
    Gamma-NB predictive NLL with continuous Gaussian readout.

    Architecture:
        λ | C ~ Gamma(a = κ*μ_λ, b = κ)  where μ_λ = (μ_x - offset) / α
        K | λ ~ Poisson(λ)  (marginalized into NB)
        Y | K=k ~ N(offset + alpha*k, beta)

    Training:
        L = -log p(Y | C) = -log Σ_k R(Y|k) NB(k; a, b)

    Test inference:
        q(K|Y,C) ∝ R(Y|k) NB(k; a, b)
        k̄ = E[K|Y,C]
        x̂ = offset + α*(a + k̄)/(b+1)

    Args:
        pred_tensor: μ_x in centered units, shape [B, 1, T, H, W]
        target_tensor: Y in centered units, shape [B, 1, T, H, W]
        valid_mask: [B, 1, T, H, W] bool, True at supervised locations
        pred_img_mean, target_img_mean: centering offset (scalars or [B,1,1,1,1])
        alpha, beta, kappa: detector parameters (scalars)
        offset: ADC offset (scalar, default 0)
        kmax: hard cap for adaptive count support
        min_signal: clamp λ_min = min_signal / alpha (scalar)
        chunk_t: temporal chunk size for memory efficiency
        tail_tol: maximum relative omitted positive tail

    Returns:
        mean NLL over valid pixels (scalar)
    """
    dtype = pred_tensor.dtype
    device = pred_tensor.device

    alpha_t = torch.as_tensor(alpha, dtype=dtype, device=device)
    beta_t = torch.clamp(
        torch.as_tensor(beta, dtype=dtype, device=device),
        min=1e-12,
    )
    offset_t = torch.as_tensor(offset, dtype=dtype, device=device)
    kappa_t = torch.as_tensor(kappa, dtype=dtype, device=device)

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

        mu_lambda = torch.clamp(
            signal / alpha_t,
            min=1e-12,
        )

        a, b = gamma_ab_from_mu_kappa(mu_lambda, kappa_t)

        k_used = adaptive_kmax(
            a, b, target_phys, alpha, beta, offset,
            relative_tail_tol=tail_tol, hard_cap=int(kmax),
        )
        k = torch.arange(k_used + 1, dtype=dtype, device=device).view(
            1, 1, k_used + 1, 1, 1, 1
        )
        mu_k = offset_t + alpha_t * k

        # log NB(k; a, b) — broadcast a, b to match k along dim 2
        log_nb = log_negbinom_pmf(k, a.unsqueeze(2), b)

        log_obs_given_k = _mpgn_log_gauss_observation(target_phys, mu_k, beta_t)

        # log p(Y|C) = logsumexp_k(log R(Y|k) + log NB(k))
        logp = torch.logsumexp(
            log_obs_given_k + log_nb,
            dim=2,
        )

        nll = -logp

        vm = mask_chunk.to(dtype)
        weight = torch.ones_like(nll) * vm
        total_nll = total_nll + (nll * weight).sum()
        total_count = total_count + weight.sum()

    return total_nll / total_count.clamp_min(1.0)


# Backward-compatible alias
_gamma_nb_nll_single_target = gamma_nb_nll_single_target


def gamma_nb_predictive_and_posterior(
    a,
    b,
    y_phys,
    alpha,
    beta,
    offset=0.0,
    *,
    valid_mask=None,
    kmax=32,
    tail_tol=1e-8,
    chunk_t=8,
):
    """Continuous MPGN predictive NLL + posterior moments with adaptive K.

    a,b: [B,1,T,H,W] fused Gamma shape/rate
    y_phys: [B,1,T,H,W] original-center observation in physical units

    Explicit unsqueeze on a,b:
        a_e,b_e: [B,1,1,T,H,W] after unsqueeze(2) wait - a is [B,1,T,H,W],
        unsqueeze(2) -> [B,1,1,T,H,W]; k is [1,1,K,1,1,1]
        Actually for broadcast with k [1,1,K,1,1,1] we need a as [B,1,1,T,H,W]
        so unsqueeze(2) on [B,1,T,H,W] gives [B,1,1,T,H,W] - yes.
    """
    dtype = a.dtype
    device = a.device

    alpha_t = torch.as_tensor(alpha, dtype=dtype, device=device)
    beta_t = torch.clamp(
        torch.as_tensor(beta, dtype=dtype, device=device),
        min=1e-12,
    )
    offset_t = torch.as_tensor(offset, dtype=dtype, device=device)

    temporal_size = a.shape[2]
    chunk_t = max(1, int(chunk_t))

    log_predictive_chunks = []
    log_q_chunks = []
    kbar_chunks = []
    var_k_chunks = []
    entropy_chunks = []
    kmax_used = []

    for t0 in range(0, temporal_size, chunk_t):
        t1 = min(t0 + chunk_t, temporal_size)
        a_c = a[:, :, t0:t1]
        b_c = b[:, :, t0:t1]
        y_c = y_phys[:, :, t0:t1]

        a_e = a_c.unsqueeze(2)
        b_e = b_c.unsqueeze(2)

        k_used = adaptive_kmax(
            a_c, b_c, y_c, alpha, beta, offset,
            relative_tail_tol=tail_tol, hard_cap=int(kmax),
        )
        kmax_used.append(k_used)
        k = torch.arange(k_used + 1, dtype=dtype, device=device).view(
            1, 1, k_used + 1, 1, 1, 1
        )
        mu_k = offset_t + alpha_t * k

        log_nb = log_negbinom_pmf(k, a_e, b_e)
        log_readout = _mpgn_log_gauss_observation(y_c, mu_k, beta_t)

        log_joint = log_nb + log_readout
        log_predictive = torch.logsumexp(log_joint, dim=2)
        log_q = log_joint - log_predictive.unsqueeze(2)

        kbar, var_k, entropy = posterior_count_moments(log_q, k)

        log_predictive_chunks.append(log_predictive)
        log_q_chunks.append(log_q)
        kbar_chunks.append(kbar)
        var_k_chunks.append(var_k)
        entropy_chunks.append(entropy)

    log_predictive = torch.cat(log_predictive_chunks, dim=2)
    max_k_used = max(kmax_used)
    padded_log_q = []
    for q in log_q_chunks:
        if q.shape[2] < max_k_used + 1:
            pad = torch.full(
                (*q.shape[:2], max_k_used + 1 - q.shape[2], *q.shape[3:]),
                -float('inf'), dtype=q.dtype, device=q.device,
            )
            q = torch.cat([q, pad], dim=2)
        padded_log_q.append(q)
    log_q = torch.cat(padded_log_q, dim=3)
    kbar = torch.cat(kbar_chunks, dim=2)
    var_k = torch.cat(var_k_chunks, dim=2)
    entropy = torch.cat(entropy_chunks, dim=2)

    mu_lambda = a / b
    x_prior_phys = offset_t + alpha_t * mu_lambda
    x_post_phys = offset_t + alpha_t * (a + kbar) / (b + 1.0)
    correction = x_post_phys - x_prior_phys

    posterior_var_lambda = (a + kbar + var_k) / (b + 1.0).square()
    posterior_var_x = alpha_t.square() * posterior_var_lambda

    nll_mean = None
    if valid_mask is not None:
        mask_f = valid_mask.to(dtype=dtype).expand_as(log_predictive)
        nll = -log_predictive
        nll_mean = (nll * mask_f).sum() / mask_f.sum().clamp_min(1.0)

    return {
        'log_predictive': log_predictive,
        'log_q': log_q,
        'kbar': kbar,
        'var_k': var_k,
        'entropy': entropy,
        'x_prior_phys': x_prior_phys,
        'x_post_phys': x_post_phys,
        'correction': correction,
        'posterior_var_x': posterior_var_x,
        'nll_mean': nll_mean,
        'kmax_used': kmax_used,
        'kmax_used_max': max_k_used,
    }


def gamma_mixture_nb_predictive_and_posterior(
    a,
    b,
    y_phys,
    alpha,
    beta,
    offset=0.0,
    *,
    valid_mask=None,
    mixture_weights=None,
    compute_map=False,
    map_grid_points=33,
    map_refinements=2,
    kmax=32,
    tail_tol=1e-8,
    chunk_t=8,
):
    """MPGN posterior for a Gamma mixture with component axis 1.

    Args:
        a, b: [B,M,T,H,W] Gamma shape/rate.
        y_phys: [B,1,T,H,W] observed intensity.
        mixture_weights: optional positive length-M vector; defaults to uniform.
    """
    if a.shape != b.shape or a.ndim != 5:
        raise ValueError('a and b must share shape [B,M,T,H,W]')
    if y_phys.ndim != 5 or y_phys.shape[1] != 1 or y_phys.shape[0] != a.shape[0] or y_phys.shape[2:] != a.shape[2:]:
        raise ValueError('y_phys must be [B,1,T,H,W] matching a and b')
    if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise FloatingPointError('non-finite Gamma mixture parameters')
    if (a <= 0).any() or (b <= 0).any():
        raise ValueError('Gamma mixture shape and rate must be positive')

    dtype, device = a.dtype, a.device
    components = a.shape[1]
    if mixture_weights is None:
        weights = torch.full((components,), 1.0 / components, dtype=dtype, device=device)
    else:
        weights = torch.as_tensor(mixture_weights, dtype=dtype, device=device)
        if weights.shape != (components,) or (weights <= 0).any() or not torch.isfinite(weights).all():
            raise ValueError('mixture_weights must be a positive finite length-M vector')
        weights = weights / weights.sum()
    log_weights = weights.log().view(1, components, 1, 1, 1, 1)

    alpha_t = torch.as_tensor(alpha, dtype=dtype, device=device)
    beta_t = torch.as_tensor(beta, dtype=dtype, device=device).clamp_min(1e-12)
    offset_t = torch.as_tensor(offset, dtype=dtype, device=device)
    chunk_t = max(1, int(chunk_t))

    log_predictive_chunks = []
    log_q_chunks = []
    kbar_chunks = []
    var_k_chunks = []
    entropy_chunks = []
    post_mean_chunks = []
    post_var_chunks = []
    responsibility_chunks = []
    map_chunks = []
    map_boundary_chunks = []
    kmax_used = []

    for t0 in range(0, a.shape[2], chunk_t):
        t1 = min(t0 + chunk_t, a.shape[2])
        a_c, b_c, y_c = a[:, :, t0:t1], b[:, :, t0:t1], y_phys[:, :, t0:t1]
        y_for_bound = y_c.expand(-1, components, -1, -1, -1)
        k_used = adaptive_kmax(
            a_c, b_c, y_for_bound, alpha, beta, offset,
            relative_tail_tol=tail_tol, hard_cap=int(kmax),
        )
        kmax_used.append(k_used)
        k = torch.arange(k_used + 1, dtype=dtype, device=device).view(1, 1, k_used + 1, 1, 1, 1)
        a_e, b_e = a_c.unsqueeze(2), b_c.unsqueeze(2)
        log_nb = log_negbinom_pmf(k, a_e, b_e)
        log_readout = _mpgn_log_gauss_observation(y_c, offset_t + alpha_t * k, beta_t)
        log_joint = log_weights + log_nb + log_readout
        log_predictive = torch.logsumexp(log_joint, dim=(1, 2), keepdim=True).squeeze(2)
        log_q = log_joint - log_predictive.unsqueeze(2)
        q = log_q.exp()

        kbar = (q * k).sum(dim=(1, 2), keepdim=True).squeeze(2)
        second_k = (q * k.square()).sum(dim=(1, 2), keepdim=True).squeeze(2)
        var_k = (second_k - kbar.square()).clamp_min(0)
        conditional_mean = (a_e + k) / (b_e + 1.0)
        conditional_second = (a_e + k) * (a_e + k + 1.0) / (b_e + 1.0).square()
        post_mean = (q * conditional_mean).sum(dim=(1, 2), keepdim=True).squeeze(2)
        post_second = (q * conditional_second).sum(dim=(1, 2), keepdim=True).squeeze(2)

        log_predictive_chunks.append(log_predictive)
        log_q_chunks.append(log_q)
        kbar_chunks.append(kbar)
        var_k_chunks.append(var_k)
        entropy_chunks.append(-(q * log_q).sum(dim=(1, 2), keepdim=True).squeeze(2))
        post_mean_chunks.append(post_mean)
        post_var_chunks.append((post_second - post_mean.square()).clamp_min(0))
        responsibility_chunks.append(q.sum(dim=2))
        if compute_map:
            map_chunk, boundary_chunk = gamma_mixture_posterior_mode(
                a_c,
                b_c,
                log_q,
                k,
                grid_points=map_grid_points,
                refinements=map_refinements,
            )
            map_chunks.append(map_chunk)
            map_boundary_chunks.append(boundary_chunk)

    max_k_used = max(kmax_used)
    padded_log_q = []
    for log_q in log_q_chunks:
        if log_q.shape[2] < max_k_used + 1:
            padding = torch.full(
                (*log_q.shape[:2], max_k_used + 1 - log_q.shape[2], *log_q.shape[3:]),
                -float('inf'), dtype=dtype, device=device,
            )
            log_q = torch.cat([log_q, padding], dim=2)
        padded_log_q.append(log_q)

    log_predictive = torch.cat(log_predictive_chunks, dim=2)
    posterior_mean_lambda = torch.cat(post_mean_chunks, dim=2)
    prior_mean_lambda = (weights.view(1, components, 1, 1, 1) * (a / b)).sum(dim=1, keepdim=True)
    x_prior_phys = offset_t + alpha_t * prior_mean_lambda
    x_post_phys = offset_t + alpha_t * posterior_mean_lambda
    nll_mean = None
    if valid_mask is not None:
        mask = valid_mask.to(dtype=dtype).expand_as(log_predictive)
        nll_mean = (-log_predictive * mask).sum() / mask.sum().clamp_min(1.0)

    result = {
        'log_predictive': log_predictive,
        'log_q': torch.cat(padded_log_q, dim=3),
        'component_responsibility': torch.cat(responsibility_chunks, dim=2),
        'kbar': torch.cat(kbar_chunks, dim=2),
        'var_k': torch.cat(var_k_chunks, dim=2),
        'entropy': torch.cat(entropy_chunks, dim=2),
        'x_prior_phys': x_prior_phys,
        'x_post_phys': x_post_phys,
        'correction': x_post_phys - x_prior_phys,
        'posterior_var_x': alpha_t.square() * torch.cat(post_var_chunks, dim=2),
        'nll_mean': nll_mean,
        'kmax_used': kmax_used,
        'kmax_used_max': max_k_used,
    }
    if compute_map:
        lambda_map = torch.cat(map_chunks, dim=2)
        map_boundary = torch.cat(map_boundary_chunks, dim=2)
        result['x_map_phys'] = offset_t + alpha_t * lambda_map
        result['map_boundary'] = map_boundary
        result['map_boundary_fraction'] = map_boundary.float().mean()
    return result


@torch.no_grad()
def gamma_mixture_posterior_mode(a, b, log_q, k, *, grid_points=33, refinements=2):
    """Global marginal mode of sum_(m,k) q_mk Gamma(a_m+k,b_m+1)."""
    if int(grid_points) < 3 or int(refinements) < 0:
        raise ValueError('grid_points must be >=3 and refinements must be non-negative')
    shape = a.unsqueeze(2) + k
    rate = b.unsqueeze(2) + 1.0
    component_mode = (shape - 1.0) / rate
    lower = component_mode.amin(dim=(1, 2), keepdim=True).squeeze(2).clamp_min(0)
    upper = component_mode.amax(dim=(1, 2), keepdim=True).squeeze(2).clamp_min(0)

    # A Gamma component with shape < 1 has infinite density at zero.  Its
    # posterior weight is mathematically positive under the Gaussian readout.
    boundary = (a < 1.0).any(dim=1, keepdim=True)
    tiny = torch.finfo(a.dtype).tiny

    def log_density(value):
        value_e = value.unsqueeze(2).clamp_min(tiny)
        log_pdf = (
            (shape - 1.0) * value_e.log()
            - rate * value_e
            + shape * rate.log()
            - torch.lgamma(shape)
        )
        return torch.logsumexp(log_q + log_pdf, dim=(1, 2), keepdim=True).squeeze(2)

    best = lower
    for _ in range(int(refinements) + 1):
        width = upper - lower
        best_score = torch.full_like(lower, -float('inf'))
        best_index = torch.zeros_like(lower, dtype=torch.long)
        for index in range(int(grid_points)):
            value = lower + width * (index / (int(grid_points) - 1))
            score = log_density(value)
            better = score > best_score
            best_score = torch.where(better, score, best_score)
            best = torch.where(better, value, best)
            best_index = torch.where(better, index, best_index)
        step = width / (int(grid_points) - 1)
        old_lower, old_upper = lower, upper
        lower = torch.maximum(old_lower, best - step)
        upper = torch.minimum(old_upper, best + step)

    return torch.where(boundary, torch.zeros_like(best), best), boundary


def gamma_mixture_nb_nll_from_ab(
    a,
    b,
    y_phys,
    valid_mask,
    alpha,
    beta,
    offset=0.0,
    *,
    mixture_weights=None,
    kmax=32,
    tail_tol=1e-8,
    chunk_t=8,
):
    """Memory-bounded Gamma-mixture predictive NLL used during training."""
    if a.shape != b.shape or a.ndim != 5:
        raise ValueError('a and b must share shape [B,M,T,H,W]')
    if y_phys.ndim != 5 or y_phys.shape[1] != 1 or y_phys.shape[0] != a.shape[0] or y_phys.shape[2:] != a.shape[2:]:
        raise ValueError('y_phys must be [B,1,T,H,W] matching a and b')
    if valid_mask.shape != y_phys.shape:
        raise ValueError('valid_mask must match y_phys')

    dtype, device = a.dtype, a.device
    components = a.shape[1]
    if mixture_weights is None:
        weights = torch.full((components,), 1.0 / components, dtype=dtype, device=device)
    else:
        weights = torch.as_tensor(mixture_weights, dtype=dtype, device=device)
        if weights.shape != (components,) or (weights <= 0).any() or not torch.isfinite(weights).all():
            raise ValueError('mixture_weights must be a positive finite length-M vector')
        weights = weights / weights.sum()
    log_weights = weights.log().view(1, components, 1, 1, 1, 1)
    alpha_t = torch.as_tensor(alpha, dtype=dtype, device=device)
    beta_t = torch.as_tensor(beta, dtype=dtype, device=device).clamp_min(1e-12)
    offset_t = torch.as_tensor(offset, dtype=dtype, device=device)

    total_nll = a.new_zeros(())
    total_count = a.new_zeros(())
    for t0 in range(0, a.shape[2], max(1, int(chunk_t))):
        t1 = min(t0 + max(1, int(chunk_t)), a.shape[2])
        a_c, b_c, y_c = a[:, :, t0:t1], b[:, :, t0:t1], y_phys[:, :, t0:t1]
        k_used = adaptive_kmax(
            a_c, b_c, y_c.expand(-1, components, -1, -1, -1),
            alpha, beta, offset,
            relative_tail_tol=tail_tol, hard_cap=int(kmax),
        )
        k = torch.arange(k_used + 1, dtype=dtype, device=device).view(1, 1, k_used + 1, 1, 1, 1)
        log_joint = (
            log_weights
            + log_negbinom_pmf(k, a_c.unsqueeze(2), b_c.unsqueeze(2))
            + _mpgn_log_gauss_observation(y_c, offset_t + alpha_t * k, beta_t)
        )
        log_predictive = torch.logsumexp(log_joint, dim=(1, 2), keepdim=True).squeeze(2)
        mask = valid_mask[:, :, t0:t1].to(dtype=dtype)
        total_nll = total_nll - (log_predictive * mask).sum()
        total_count = total_count + mask.sum()
    return total_nll / total_count.clamp_min(1.0)


def gamma_nb_predictive_nll_from_ab(
    a,
    b,
    target_tensor,
    valid_mask,
    target_img_mean,
    alpha,
    beta,
    offset=0.0,
    kmax=32,
    chunk_t=8,
    tail_tol=1e-8,
):
    """Marginal predictive NLL from fused (a,b) and original-center target."""
    y_phys = _to_phys_units(target_tensor, target_img_mean)
    result = gamma_nb_predictive_and_posterior(
        a,
        b,
        y_phys,
        alpha=alpha,
        beta=beta,
        offset=offset,
        valid_mask=valid_mask,
        kmax=kmax,
        tail_tol=tail_tol,
        chunk_t=chunk_t,
    )
    return result['nll_mean']
