"""Axis-wise Hilbert analytic operators for [B,C,T,H,W] videos."""

import torch


_DIRECTION_TO_DIM = {"x": -1, "y": -2, "t": -3}


def _check_video(x, *, complex_input):
    if x.ndim != 5:
        raise ValueError("expected [B,C,T,H,W], got {}".format(tuple(x.shape)))
    if complex_input != x.is_complex():
        kind = "complex" if complex_input else "real"
        raise TypeError("expected a {} tensor, got {}".format(kind, x.dtype))
    if not complex_input and not x.is_floating_point():
        raise TypeError("expected a floating-point tensor, got {}".format(x.dtype))


def _axis_dim(direction):
    try:
        return _DIRECTION_TO_DIM[direction]
    except KeyError as exc:
        raise ValueError(
            "direction must be one of {}, got {!r}".format(
                tuple(_DIRECTION_TO_DIM), direction
            )
        ) from exc


def make_hilbert_multiplier(x, dim):
    """Return broadcastable -i sign(k_dim), with DC/Nyquist set to zero."""
    if dim not in _DIRECTION_TO_DIM.values():
        raise ValueError("dim must be one of -1, -2, -3, got {}".format(dim))
    real_dtype = x.real.dtype if x.is_complex() else x.dtype
    complex_dtype = torch.complex64 if real_dtype == torch.float32 else torch.complex128
    length = x.shape[dim]
    frequency = torch.fft.fftfreq(length, device=x.device, dtype=real_dtype)
    multiplier = -1j * torch.sign(frequency).to(complex_dtype)
    multiplier[0] = 0
    if length % 2 == 0:
        multiplier[length // 2] = 0
    shape = [1] * x.ndim
    shape[dim] = length
    return multiplier.reshape(shape)


def hilbert_axis(x, dim):
    """Apply the real axis-wise Hilbert transform along T, H, or W."""
    _check_video(x, complex_input=False)
    spectrum = torch.fft.fft(x, dim=dim, norm="ortho")
    return torch.fft.ifft(
        spectrum * make_hilbert_multiplier(x, dim), dim=dim, norm="ortho"
    ).real


def quadrature(x, direction):
    """Compatibility name for H_x(x), H_y(x), or H_t(x)."""
    return hilbert_axis(x, _axis_dim(direction))


def analytic_channel(x, direction):
    """Return A_d(x) = x + i H_d(x)."""
    return torch.complex(x, quadrature(x, direction))


def analytic_representation(x):
    """Map one centered real channel to [I+iH_x(I), I+iH_y(I), I+iH_t(I)]."""
    _check_video(x, complex_input=False)
    if x.shape[1] != 1:
        raise ValueError("analytic input must have one channel, got {}".format(x.shape[1]))
    return torch.cat(
        [analytic_channel(x, direction) for direction in _DIRECTION_TO_DIM], dim=1
    )


def analytic_inverse_axis(u, v, dim):
    """Least-squares A_d^dagger(u+i v) using the forward multiplier."""
    _check_video(u, complex_input=False)
    _check_video(v, complex_input=False)
    if u.shape != v.shape:
        raise ValueError("u and v must have the same shape")
    multiplier = make_hilbert_multiplier(u, dim)
    spectrum_u = torch.fft.fft(u, dim=dim, norm="ortho")
    spectrum_v = torch.fft.fft(v, dim=dim, norm="ortho")
    estimate = (
        spectrum_u + multiplier.conj() * spectrum_v
    ) / (1.0 + multiplier.abs().square())
    return torch.fft.ifft(estimate, dim=dim, norm="ortho").real


def analytic_pseudoinverse(z, direction):
    """Least-squares A_d^dagger for an arbitrary complex video tensor."""
    _check_video(z, complex_input=True)
    return analytic_inverse_axis(z.real, z.imag, _axis_dim(direction))


def inverse_candidates(z):
    """Convert structured [Z_x,Z_y,Z_t] to three Hilbert-axis candidates."""
    _check_video(z, complex_input=True)
    if z.shape[1] != 3:
        raise ValueError("analytic output must have three channels, got {}".format(z.shape[1]))
    return torch.cat(
        [
            analytic_pseudoinverse(z[:, index:index + 1], direction)
            for index, direction in enumerate(_DIRECTION_TO_DIM)
        ],
        dim=1,
    )


def quadrature_contribution(z, direction, eps=None):
    """Per-sample relative reconstruction contribution of the imaginary field."""
    _check_video(z, complex_input=True)
    full = analytic_pseudoinverse(z, direction)
    real_only = analytic_inverse_axis(
        z.real, torch.zeros_like(z.real), _axis_dim(direction)
    )
    reduce_dims = tuple(range(1, full.ndim))
    numerator = (full - real_only).square().sum(dim=reduce_dims).sqrt()
    denominator = full.square().sum(dim=reduce_dims).sqrt()
    if eps is None:
        eps = torch.finfo(full.dtype).eps
    return numerator / denominator.clamp_min(float(eps))


def analytic_projection(z, direction):
    """Orthogonally project a complex output onto range(A_d)."""
    return analytic_channel(analytic_pseudoinverse(z, direction), direction)


def analytic_residual_ratio(z, direction, eps=None):
    """Per-sample relative distance from the axis-Hilbert analytic manifold."""
    _check_video(z, complex_input=True)
    residual = z - analytic_projection(z, direction)
    reduce_dims = tuple(range(1, z.ndim))
    numerator = residual.abs().square().sum(dim=reduce_dims).sqrt()
    denominator = z.abs().square().sum(dim=reduce_dims).sqrt()
    if eps is None:
        eps = torch.finfo(z.real.dtype).eps
    return numerator / denominator.clamp_min(float(eps))
