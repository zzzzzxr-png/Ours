"""Single-Gamma context prior + fixed-kappa Gamma-Poisson posterior math.

All computation in log domain (lgamma / logsumexp), per project numerical
requirements. No mixture, no unrolling — Phase 2 (fixed kappa) only.

Mathematical reference (spec Sec 4-8):
  - Gamma(λ; a, b) shape-rate parametrization: λ > 0
  - a = κ * μ_λ (shape), b = κ (rate), E[λ] = a/b = μ_λ, Var[λ] = a/b² = μ_λ/κ
  - K | λ ~ Poisson(λ)  =>  K ~ NB(k; a, b) (marginalized over λ)
  - Y | K=k ~ N(offset + α*k, β) (fixed readout, reused from likelihood)
  - Posterior: q(K | Y, C) = softmax_k(log NB(k) + log R(Y|k))
  - Posterior mean: x̂ = offset + α*(a + k̄)/(b+1), where k̄ = E[K|Y,C]
"""

import torch


def gamma_ab_from_mu_kappa(mu_lambda, kappa):
    """Convert (mu_lambda, kappa) to Gamma (a, b) shape-rate parameters.

    Args:
        mu_lambda: expected electron count (λ = x / α), broadcastable tensor
        kappa: concentration parameter, broadcastable tensor

    Returns:
        (a, b) where a = kappa * mu_lambda, b = kappa
    """
    a = kappa * mu_lambda
    b = kappa
    return a, b


def log_negbinom_pmf(k, a, b):
    """log NB(k; a, b) under shape-rate parametrization.

    NB(k; a, b) = Γ(a+k) / (Γ(a) * Γ(k+1)) * (b/(b+1))^a * (1/(b+1))^k

    log NB(k; a, b) = log Γ(a+k) - log Γ(a) - log Γ(k+1)
                      + a * log(b/(b+1)) + k * log(1/(b+1))

    Args:
        k: count (integer, but computation treats as real for autodiff)
        a: shape parameter, broadcastable to k (optionally batched/spatial)
        b: rate parameter, same shape as a

    Returns:
        log-pmf, same shape as k
    """
    log_1_over_b1 = -torch.log1p(b)           # log(1/(b+1))
    log_b_over_b1 = torch.log(b) - torch.log1p(b)   # log(b/(b+1))
    return (
        torch.lgamma(a + k) - torch.lgamma(a) - torch.lgamma(k + 1.0)
        + a * log_b_over_b1 + k * log_1_over_b1
    )


def count_posterior_log_q(log_readout, log_nb):
    """Compute log q(K | Y, C) = softmax_k(log R(Y|k) + log NB(k; a, b)).

    Args:
        log_readout: log R(Y|k), shape [B, 1, K_max+1, T, H, W] or [B, C, K_max+1, T, H, W]
        log_nb: log NB(k; a, b), same shape

    Returns:
        log_q: log-posterior over k, same shape as inputs
    """
    log_joint = log_readout + log_nb
    log_norm = torch.logsumexp(log_joint, dim=2, keepdim=True)
    return log_joint - log_norm


def posterior_count_moments(log_q, k_grid):
    """Compute E[K|Y,C], Var[K|Y,C], entropy H(K) from posterior.

    Args:
        log_q: log q(K|Y,C), shape [B, C, K_max+1, T, H, W]
        k_grid: k ∈ {0, 1, ..., K_max}, shape [1, 1, K_max+1, 1, 1, 1]

    Returns:
        kbar: E[K|Y,C], shape [B, C, T, H, W]
        var_k: Var[K|Y,C], shape [B, C, T, H, W]
        entropy: -∑_k q_k * log q_k, shape [B, C, T, H, W]
    """
    q = log_q.exp()
    kbar = (q * k_grid).sum(dim=2)
    var_k = (q * (k_grid - kbar.unsqueeze(2)) ** 2).sum(dim=2)
    entropy = -(q * log_q).sum(dim=2)
    return kbar, var_k, entropy


def posterior_mean_x(a, b, kbar, alpha, offset):
    """Compute posterior mean clean signal.

    x̂ = offset + α * (a + k̄) / (b + 1)

    Args:
        a: Gamma shape = κ * μ_λ, shape [B, C, T, H, W]
        b: Gamma rate = κ, shape [B, C, T, H, W]
        kbar: E[K|Y,C], shape [B, C, T, H, W]
        alpha: photon gain (scalar or batched)
        offset: ADC offset (scalar or batched)

    Returns:
        x_hat: posterior mean, shape [B, C, T, H, W]
    """
    return offset + alpha * (a + kbar) / (b + 1.0)


def posterior_mean_photon_count(alpha, kbar):
    """Posterior mean photon-count component (not a denoised image).

    P̂ = α * k̄

    Args:
        alpha: photon gain
        kbar: E[K|Y,C]

    Returns:
        P_hat: posterior mean count, same shape as kbar
    """
    return alpha * kbar


def posterior_mean_shot_noise(p_hat, x_hat):
    """Posterior mean shot-noise component.

    N̂_P = P̂ - x̂

    Args:
        p_hat: posterior mean photon-count component
        x_hat: posterior mean clean signal

    Returns:
        noise_hat: shot-noise, same shape as inputs
    """
    return p_hat - x_hat


def posterior_mean_gaussian_readout_simple(y_phys, offset, alpha, kbar):
    """Approximate posterior mean Gaussian readout (ignoring quantization/clipping).

    Ĝ ≈ Y - o - α*k̄

    (Exact version would compute conditional mean A_i | Y_i, K_i = k for each k,
    then marginalize; deferred to Phase 2b. This approximation suffices for
    continuous-domain analysis and assumes ADC quantization is fine relative to
    read noise scale.)

    Args:
        y_phys: observed measurement Y in physical units
        offset: ADC offset
        alpha: photon gain
        kbar: E[K|Y,C]

    Returns:
        g_hat: approximate posterior mean readout, same shape as y_phys
    """
    return y_phys - offset - alpha * kbar
