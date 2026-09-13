"""Masking / sub-sampling strategies for the posterior (unroll-transformer) pipeline.

SRDTrans spatial-neighbor sampling, checkerboard / stripe pair sampling,
sparse directional masking, and the H/W random_transform augmentation.
"""

import random

import numpy as np
import torch

# ── SRDTrans spatial-neighbor sampling ───────────────────────────────────────
# Ported from /data/zhouxirou/SRDTrans/sampling.py.
# einops dependency removed; replaced with equivalent torch ops.

_srd_seed_counter = 0


def _srd_get_generator(device):
    global _srd_seed_counter
    _srd_seed_counter += 1
    g = torch.Generator(device=device)
    g.manual_seed(_srd_seed_counter)
    return g


def _srd_space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded = torch.nn.functional.unfold(x, block_size, stride=block_size)
    return unfolded.view(n, c * block_size ** 2, h // block_size, w // block_size)


def _srd_generate_mask_pair(img):
    """Generate three boolean masks for spatial-neighbor self-supervised sampling.

    img: [N, C, T, H, W]
    Returns mask1, mask2, mask3 each of length N*T*(H//2)*(W//2)*4.
    """
    n, c, t, h, w = img.shape
    device = img.device
    size = n * t * (h // 2) * (w // 2) * 4
    mask1 = torch.zeros(size, dtype=torch.bool, device=device)
    mask2 = torch.zeros(size, dtype=torch.bool, device=device)
    mask3 = torch.zeros(size, dtype=torch.bool, device=device)

    idx_pair = torch.tensor(
        [[0, 1, 2], [0, 2, 1],
         [1, 0, 3], [1, 3, 0],
         [2, 0, 3], [2, 3, 0],
         [3, 2, 1], [3, 1, 2]],
        dtype=torch.int64, device=device)

    rd_idx = torch.zeros(n * t * (h // 2) * (w // 2), dtype=torch.int64, device=device)
    torch.randint(low=0, high=8, size=(n * t * (h // 2) * (w // 2),),
                  generator=_srd_get_generator(device), out=rd_idx)

    rd_pair_idx = idx_pair[rd_idx]
    rd_pair_idx += torch.arange(
        start=0, end=size, step=4,
        dtype=torch.int64, device=device).reshape(-1, 1)

    mask1[rd_pair_idx[:, 0]] = 1
    mask2[rd_pair_idx[:, 1]] = 1
    mask3[rd_pair_idx[:, 2]] = 1
    return mask1, mask2, mask3


def _srd_generate_subimages(img, mask):
    """Extract a spatially downsampled subimage using the given mask.

    img:  [N, C, T, H, W]
    mask: boolean tensor of length N*T*(H//2)*(W//2)*4
    Returns: [N, C, T, H//2, W//2]
    """
    n, c, t, h, w = img.shape
    # (N, C, T, H, W) -> (N*T, C, H, W)
    img_flat = img.permute(0, 2, 1, 3, 4).contiguous().view(n * t, c, h, w)
    subimage = torch.zeros(n * t, c, h // 2, w // 2,
                           dtype=img.dtype, layout=img.layout, device=img.device)
    for i in range(c):
        img_ch = _srd_space_to_depth(img_flat[:, i:i + 1, :, :], block_size=2)
        img_ch = img_ch.permute(0, 2, 3, 1).reshape(-1)
        subimage[:, i:i + 1, :, :] = img_ch[mask].reshape(
            n * t, h // 2, w // 2, c).permute(0, 3, 1, 2)
    # (N*T, C, H//2, W//2) -> (N, C, T, H//2, W//2)
    subimage = subimage.view(n, t, c, h // 2, w // 2).permute(0, 2, 1, 3, 4).contiguous()
    return subimage


def _checkerboard_diag_h_pair(img):
    """
    Fixed checkerboard diagonal sampling with H compression.

    Input:
        img: [N, C, T, 2H, W], W must be even.

    For each 2x2 block:
        a b
        c d

    H is compressed:
        input  = [a, d]
        target = [c, b]

    Returns:
        inp, tgt: [N, C, T, H, W]
    """
    if img.shape[-2] % 2 != 0:
        raise ValueError("Raw height must be even for spatial_checker H-compression.")
    if img.shape[-1] % 2 != 0:
        raise ValueError("Raw width must be even for spatial_checker H-compression.")

    top = img[:, :, :, 0::2, :]   # a b rows
    bot = img[:, :, :, 1::2, :]   # c d rows

    inp = torch.empty_like(top)
    tgt = torch.empty_like(top)

    # input: [a, d]
    inp[..., 0::2] = top[..., 0::2]   # a
    inp[..., 1::2] = bot[..., 1::2]   # d moved to b position

    # target: [c, b]
    tgt[..., 0::2] = bot[..., 0::2]   # c moved to a position
    tgt[..., 1::2] = top[..., 1::2]   # b

    return inp, tgt


def _checkerboard_diag_w_pair(img):
    """
    Fixed checkerboard diagonal sampling with W compression.

    Input:
        img: [N, C, T, H, 2W], H must be even.

    For each 2x2 block:
        a b
        c d

    W is compressed:
        input  = [a; d]
        target = [b; c]

    Returns:
        inp, tgt: [N, C, T, H, W]
    """
    if img.shape[-2] % 2 != 0:
        raise ValueError("Raw height must be even for spatial_checker W-compression.")
    if img.shape[-1] % 2 != 0:
        raise ValueError("Raw width must be even for spatial_checker W-compression.")

    left = img[:, :, :, :, 0::2]   # a c columns
    right = img[:, :, :, :, 1::2]  # b d columns

    inp = torch.empty_like(left)
    tgt = torch.empty_like(left)

    # input: [a; d]
    inp[..., 0::2, :] = left[..., 0::2, :]    # a
    inp[..., 1::2, :] = right[..., 1::2, :]   # d moved to c position

    # target: [b; c]
    tgt[..., 0::2, :] = right[..., 0::2, :]   # b moved to a position
    tgt[..., 1::2, :] = left[..., 1::2, :]    # c

    return inp, tgt


def _stripe_h_pair(img):
    """
    Fixed horizontal stripe sampling with H compression.

    Input:
        img: [N, C, T, 2H, W]

    Even rows -> input, odd rows -> target (random swap may follow).
    Returns:
        inp, tgt: [N, C, T, H, W]
    """
    inp = img[:, :, :, 0::2, :]
    tgt = img[:, :, :, 1::2, :]
    return inp, tgt


def _stripe_w_pair(img):
    """
    Fixed vertical stripe sampling with W compression.

    Input:
        img: [N, C, T, H, 2W]

    Even columns -> input, odd columns -> target (random swap may follow).
    Returns:
        inp, tgt: [N, C, T, H, W]
    """
    inp = img[:, :, :, :, 0::2]
    tgt = img[:, :, :, :, 1::2]
    return inp, tgt


def _stripe_t_pair(img):
    """
    Fixed odd-even frame sampling with T compression.

    Input:
        img: [N, C, 2T, H, W]

    Even frames -> input, odd frames -> target (random swap may follow).
    Returns:
        inp, tgt: [N, C, T, H, W]
    """
    if img.shape[2] % 2 != 0:
        raise ValueError("Raw time must be even for temporal_checker T-compression.")

    inp = img[:, :, 0::2, :, :]
    tgt = img[:, :, 1::2, :, :]
    return inp, tgt


_FIXED_HW_SPATIAL_MODES = frozenset({'spatial_checker', 'spatial_stripe'})
_FIXED_TW_TEMPORAL_MODES = frozenset({'temporal_checker'})
_FIXED_SINGLE_TARGET_MODES = _FIXED_HW_SPATIAL_MODES | _FIXED_TW_TEMPORAL_MODES

_SPARSE_MASK_MODES = frozenset({
    'spatial_mask',
    'temporal_mask',
})


def _make_sparse_lattice_mask(x, mask_ratio, mask_min_dist, mode):
    """
    Generate sparse mask [B,1,T,H,W].

    spatial_mask: sparse in H/W lattice.
    temporal_mask: sparse in T lattice.
    """
    b, c, t, h, w = x.shape
    device = x.device
    dtype = x.dtype

    mask_ratio = float(mask_ratio)
    d = max(1, int(mask_min_dist))

    if mask_ratio <= 0:
        return torch.zeros((b, 1, t, h, w), dtype=dtype, device=device)

    if d <= 1:
        return (
            torch.rand((b, 1, t, h, w), device=device)
            < mask_ratio
        ).to(dtype)

    candidates = torch.zeros((b, 1, t, h, w), dtype=dtype, device=device)

    if mode == 'spatial_mask':
        off_h = torch.randint(0, d, (1,), device=device).item()
        off_w = torch.randint(0, d, (1,), device=device).item()
        candidates[:, :, :, off_h::d, off_w::d] = 1.0
    elif mode == 'temporal_mask':
        off_t = torch.randint(0, d, (1,), device=device).item()
        candidates[:, :, off_t::d, :, :] = 1.0
    else:
        raise ValueError('Unknown mask mode: {}'.format(mode))

    candidate_fraction = candidates.mean().clamp_min(1e-12)
    keep_prob = min(1.0, mask_ratio / float(candidate_fraction.item()))

    selected = (
        torch.rand_like(candidates)
        < keep_prob
    ).to(dtype)

    return candidates * selected


def _spatial_neighbor_replacement(x):
    """Randomly choose one H/W neighbor for each voxel."""
    padded = F.pad(x, (1, 1, 1, 1, 0, 0), mode='reflect')

    up = padded[:, :, :, 0:-2, 1:-1]
    down = padded[:, :, :, 2:, 1:-1]
    left = padded[:, :, :, 1:-1, 0:-2]
    right = padded[:, :, :, 1:-1, 2:]

    neighbors = torch.stack([up, down, left, right], dim=1)
    b, n, c, t, h, w = neighbors.shape

    idx = torch.randint(
        0,
        n,
        (b, 1, c, t, h, w),
        device=x.device,
    )

    return torch.gather(neighbors, dim=1, index=idx).squeeze(1)


def _temporal_neighbor_replacement(x):
    """Randomly choose previous or next frame using reflection padding."""
    padded = F.pad(x, (0, 0, 0, 0, 1, 1), mode='reflect')

    prev_frame = padded[:, :, 0:-2]
    next_frame = padded[:, :, 2:]

    neighbors = torch.stack([prev_frame, next_frame], dim=1)
    b, n, c, t, h, w = neighbors.shape

    idx = torch.randint(
        0,
        n,
        (b, 1, c, t, h, w),
        device=x.device,
    )

    return torch.gather(neighbors, dim=1, index=idx).squeeze(1)


def _make_masked_replacement(x_raw, mode, mask_ratio, mask_min_dist):
    """
    Return y_prime_raw and loss_mask.

    x_raw: original noisy patch in raw scaled units, not mean-centered.
    """
    loss_mask = _make_sparse_lattice_mask(
        x_raw,
        mask_ratio=mask_ratio,
        mask_min_dist=mask_min_dist,
        mode=mode,
    )

    if mode == 'spatial_mask':
        replacement = _spatial_neighbor_replacement(x_raw)
    elif mode == 'temporal_mask':
        replacement = _temporal_neighbor_replacement(x_raw)
    else:
        raise ValueError('Unknown mask mode: {}'.format(mode))

    y_prime = torch.where(
        loss_mask.bool(),
        replacement,
        x_raw,
    )

    return y_prime, loss_mask


def random_transform_1(x):
    """Apply random H/W rotation or flip to a single patch [T, H, W]."""
    p_trans = random.randrange(8)

    if p_trans == 0:
        return x
    elif p_trans == 1:
        return np.rot90(x, k=1, axes=(1, 2))
    elif p_trans == 2:
        return np.rot90(x, k=2, axes=(1, 2))
    elif p_trans == 3:
        return np.rot90(x, k=3, axes=(1, 2))
    elif p_trans == 4:
        return x[:, :, ::-1]
    elif p_trans == 5:
        return np.rot90(x[:, :, ::-1], k=1, axes=(1, 2))
    elif p_trans == 6:
        return np.rot90(x[:, :, ::-1], k=2, axes=(1, 2))
    elif p_trans == 7:
        return np.rot90(x[:, :, ::-1], k=3, axes=(1, 2))


def random_transform_3(a, b, c):
    p_trans = random.randrange(8)

    def apply(x):
        if p_trans == 0:
            return x
        elif p_trans == 1:
            return np.rot90(x, k=1, axes=(1, 2))
        elif p_trans == 2:
            return np.rot90(x, k=2, axes=(1, 2))
        elif p_trans == 3:
            return np.rot90(x, k=3, axes=(1, 2))
        elif p_trans == 4:
            return x[:, :, ::-1]
        elif p_trans == 5:
            return np.rot90(x[:, :, ::-1], k=1, axes=(1, 2))
        elif p_trans == 6:
            return np.rot90(x[:, :, ::-1], k=2, axes=(1, 2))
        elif p_trans == 7:
            return np.rot90(x[:, :, ::-1], k=3, axes=(1, 2))

    return apply(a), apply(b), apply(c)
