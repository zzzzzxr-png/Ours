"""Masking strategies for SRDTrans-protocol self-supervised training.

Directional/lattice sparse masking used by ``likelihood.trainer``:
height_mask, width_mask, temporal_mask (spatial_mask removed: use height/width),
spatial_mask_mean, temporal_mask_mean, slice-based masking, and Noise2Void-style masking.

Complementary mask groups (train one / val exhaustive) share the same lattice
partition so train and validation use identical local operations.
"""

import math

import torch


def plan_mask_periods(mask_ratio: float, min_dist: int) -> tuple[int, int, int]:
    """Factor ~1/mask_ratio into (pt, ph, pw) with each period >= min_dist when possible.

    Defaults mask_ratio=0.05 -> target 20 groups -> (2, 2, 5) with min_dist=2.
    """
    d = max(1, int(min_dist))
    target = max(1, int(round(1.0 / float(mask_ratio))))

    # Prefer three factors each >= d whose product is close to target.
    best = None
    best_score = None
    # Search modest ranges; periods need not equal min_dist.
    max_p = max(d * 4, int(math.ceil(target ** (1.0 / 3.0)) * 3), target)
    for pt in range(1, max_p + 1):
        for ph in range(1, max_p + 1):
            if target % (pt * ph) != 0:
                continue
            pw = target // (pt * ph)
            if pw < 1 or pw > max_p:
                continue
            periods = (pt, ph, pw)
            # Score: prefer factors >= d, then balanced product match (exact here).
            penalty = sum(max(0, d - p) for p in periods)
            balance = max(periods) - min(periods)
            score = (penalty, balance, max(periods))
            if best_score is None or score < best_score:
                best_score = score
                best = periods

    if best is None:
        # Fallback: cubic lattice with stride d -> d^3 groups.
        best = (d, d, d)

    return best


def make_exhaustive_mask_groups(
    shape_or_tensor,
    mask_ratio: float,
    min_dist: int,
    *,
    device=None,
    dtype=torch.bool,
) -> list:
    """Return complementary lattice groups covering every voxel exactly once.

    Groups are residues of (t % pt, h % ph, w % pw). Number of groups is
    pt*ph*pw from ``plan_mask_periods`` (not assumed from 1/mask_ratio alone
    without factorization).

    Returns:
        list of bool tensors [B,1,T,H,W] (B=1 if shape tuple given without batch).
    """
    if torch.is_tensor(shape_or_tensor):
        x = shape_or_tensor
        if x.ndim != 5:
            raise ValueError('Expected [B,C,T,H,W], got {}'.format(tuple(x.shape)))
        B, _, T, H, W = x.shape
        device = x.device if device is None else device
    else:
        if len(shape_or_tensor) == 3:
            T, H, W = shape_or_tensor
            B = 1
        elif len(shape_or_tensor) == 5:
            B, _, T, H, W = shape_or_tensor
        else:
            raise ValueError('shape must be (T,H,W) or (B,C,T,H,W)')
        if device is None:
            device = torch.device('cpu')

    pt, ph, pw = plan_mask_periods(mask_ratio, min_dist)
    num_groups = pt * ph * pw
    print(
        '[mask groups] periods=(t={},h={},w={}) -> {} complementary groups '
        '(requested mask_ratio={:.4f}, effective≈{:.4f}, min_dist={})'.format(
            pt, ph, pw, num_groups, float(mask_ratio), 1.0 / num_groups, int(min_dist)
        )
    )

    tt = torch.arange(T, device=device).view(T, 1, 1)
    hh = torch.arange(H, device=device).view(1, H, 1)
    ww = torch.arange(W, device=device).view(1, 1, W)

    groups = []
    coverage = torch.zeros((B, 1, T, H, W), dtype=torch.int16, device=device)
    for gt in range(pt):
        for gh in range(ph):
            for gw in range(pw):
                mask = (
                    ((tt % pt) == gt)
                    & ((hh % ph) == gh)
                    & ((ww % pw) == gw)
                )
                mask_b = mask.view(1, 1, T, H, W).expand(B, 1, T, H, W).to(dtype)
                groups.append(mask_b.clone())
                coverage += mask_b.to(torch.int16)

    if int(coverage.min().item()) != 1 or int(coverage.max().item()) != 1:
        raise RuntimeError(
            'Exhaustive masks do not cover each voxel exactly once: '
            'coverage range=[{}, {}]'.format(
                int(coverage.min().item()), int(coverage.max().item())
            )
        )
    return groups


def make_training_mask_group(
    noisy: torch.Tensor,
    mask_ratio: float,
    min_dist: int,
    *,
    group_index: int | None = None,
) -> tuple:
    """Sample one complementary mask group (same partition as validation).

    Returns:
        loss_mask [B,1,T,H,W], group_index, num_groups
    """
    if noisy.ndim != 5:
        raise ValueError('Expected [B,C,T,H,W], got {}'.format(tuple(noisy.shape)))

    B, _, T, H, W = noisy.shape
    pt, ph, pw = plan_mask_periods(mask_ratio, min_dist)
    num_groups = pt * ph * pw

    if group_index is None:
        group_index = int(torch.randint(0, num_groups, (1,), device=noisy.device).item())
    group_index = int(group_index) % num_groups

    gt = group_index // (ph * pw)
    rem = group_index % (ph * pw)
    gh = rem // pw
    gw = rem % pw

    tt = torch.arange(T, device=noisy.device).view(T, 1, 1)
    hh = torch.arange(H, device=noisy.device).view(1, H, 1)
    ww = torch.arange(W, device=noisy.device).view(1, 1, W)
    mask = (
        ((tt % pt) == gt)
        & ((hh % ph) == gh)
        & ((ww % pw) == gw)
    )
    loss_mask = mask.view(1, 1, T, H, W).expand(B, 1, T, H, W).contiguous()
    return loss_mask, group_index, num_groups


def _make_sparse_lattice_mask(
    x: torch.Tensor,
    mask_ratio: float,
    min_dist: int,
    valid_t: tuple,
    valid_h: tuple,
    valid_w: tuple,
    *,
    lattice_axes: frozenset[str] | None = None,
    lattice_random_phase: bool = True,
) -> torch.Tensor:
    """Create sparse mask with a lattice-based minimum-distance constraint.

    Args:
        x: input tensor [B, C, T, H, W].
        mask_ratio: target ratio over all voxels.
        min_dist: lattice stride. min_dist=3 means selected centers are at least
            3 voxels apart on the active lattice axes before random subsampling.
        valid_t/h/w: inclusive-exclusive valid ranges for mask centers.
        lattice_axes: subset of {'t', 'h', 'w'} on which to apply the lattice.
            Each active axis is shifted by a random offset in [0, min_dist) when
            lattice_random_phase is True; otherwise phase is fixed at 0.
            Inactive axes allow mask centers at every index (no stride).

    Returns:
        mask: bool tensor [B, 1, T, H, W].
    """
    if x.ndim != 5:
        raise ValueError('Expected input shape [B, C, T, H, W], got {}'.format(tuple(x.shape)))
    if not (0.0 < float(mask_ratio) < 1.0):
        raise ValueError('mask_ratio must be in (0, 1), got {}'.format(mask_ratio))

    B, _, T, H, W = x.shape
    device = x.device
    d = max(1, int(min_dist))
    axes = lattice_axes if lattice_axes is not None else frozenset({'t', 'h', 'w'})

    t0, t1 = valid_t
    h0, h1 = valid_h
    w0, w1 = valid_w
    if t1 <= t0 or h1 <= h0 or w1 <= w0:
        raise ValueError(
            'Invalid mask valid ranges: valid_t={}, valid_h={}, valid_w={}'.format(
                valid_t, valid_h, valid_w
            )
        )

    tt = torch.arange(T, device=device).view(T, 1, 1)
    hh = torch.arange(H, device=device).view(1, H, 1)
    ww = torch.arange(W, device=device).view(1, 1, W)
    valid = (
        (tt >= t0) & (tt < t1) &
        (hh >= h0) & (hh < h1) &
        (ww >= w0) & (ww < w1)
    )

    mask = torch.zeros((B, 1, T, H, W), dtype=torch.bool, device=device)
    target_num = max(1, int(round(float(mask_ratio) * T * H * W)))

    for b in range(B):
        candidate = valid
        if 't' in axes:
            phase_t = (
                int(torch.randint(0, d, (1,), device=device).item())
                if lattice_random_phase else 0
            )
            candidate = candidate & ((tt % d) == phase_t)
        if 'h' in axes:
            phase_h = (
                int(torch.randint(0, d, (1,), device=device).item())
                if lattice_random_phase else 0
            )
            candidate = candidate & ((hh % d) == phase_h)
        if 'w' in axes:
            phase_w = (
                int(torch.randint(0, d, (1,), device=device).item())
                if lattice_random_phase else 0
            )
            candidate = candidate & ((ww % d) == phase_w)

        idx = candidate.flatten().nonzero(as_tuple=False).flatten()
        if idx.numel() == 0:
            raise RuntimeError('No valid mask candidates. Check patch size and min_dist.')

        keep_num = min(target_num, idx.numel())
        perm = torch.randperm(idx.numel(), device=device)[:keep_num]
        chosen = idx[perm]
        mask[b, 0].view(-1)[chosen] = True

    return mask


def _shift_neighbor(x: torch.Tensor, dt: int = 0, dh: int = 0, dw: int = 0) -> torch.Tensor:
    """Return tensor y where y[t,h,w] = x[t+dt,h+dh,w+dw] for valid centers.

    Invalid border values are left as original x, but they will not be used
    because mask centers are restricted to valid ranges.
    """
    y = x.clone()
    if dt == -1:
        y[:, :, 1:, :, :] = x[:, :, :-1, :, :]
    elif dt == 1:
        y[:, :, :-1, :, :] = x[:, :, 1:, :, :]
    elif dt != 0:
        raise ValueError('Only dt in {-1, 0, 1} is supported.')

    if dh == -1:
        y[:, :, :, 1:, :] = x[:, :, :, :-1, :]
    elif dh == 1:
        y[:, :, :, :-1, :] = x[:, :, :, 1:, :]
    elif dh != 0:
        raise ValueError('Only dh in {-1, 0, 1} is supported.')

    if dw == -1:
        y[:, :, :, :, 1:] = x[:, :, :, :, :-1]
    elif dw == 1:
        y[:, :, :, :, :-1] = x[:, :, :, :, 1:]
    elif dw != 0:
        raise ValueError('Only dw in {-1, 0, 1} is supported.')

    return y


def _make_sparse_lattice_mask_2d(
    batch_size: int,
    d1: int,
    d2: int,
    mask_ratio: float,
    min_dist: int,
    valid_d1: tuple,
    valid_d2: tuple,
    device: torch.device,
) -> torch.Tensor:
    """Sparse lattice mask on a 2D plane [B, 1, D1, D2]."""
    if not (0.0 < float(mask_ratio) < 1.0):
        raise ValueError('mask_ratio must be in (0, 1), got {}'.format(mask_ratio))

    d = max(1, int(min_dist))
    d1_0, d1_1 = valid_d1
    d2_0, d2_1 = valid_d2
    if d1_1 <= d1_0 or d2_1 <= d2_0:
        raise ValueError(
            'Invalid 2D valid ranges: valid_d1={}, valid_d2={}'.format(valid_d1, valid_d2)
        )

    dd1 = torch.arange(d1, device=device).view(d1, 1)
    dd2 = torch.arange(d2, device=device).view(1, d2)
    valid = (
        (dd1 >= d1_0) & (dd1 < d1_1) &
        (dd2 >= d2_0) & (dd2 < d2_1)
    )

    mask = torch.zeros((batch_size, 1, d1, d2), dtype=torch.bool, device=device)
    target_num = max(1, int(round(float(mask_ratio) * d1 * d2)))

    for b in range(batch_size):
        phase_d1 = int(torch.randint(0, d, (1,), device=device).item())
        phase_d2 = int(torch.randint(0, d, (1,), device=device).item())
        candidate = (
            ((dd1 % d) == phase_d1) &
            ((dd2 % d) == phase_d2) &
            valid
        )
        idx = candidate.flatten().nonzero(as_tuple=False).flatten()
        if idx.numel() == 0:
            raise RuntimeError('No valid 2D mask candidates. Check patch size and min_dist.')
        keep_num = min(target_num, idx.numel())
        perm = torch.randperm(idx.numel(), device=device)[:keep_num]
        chosen = idx[perm]
        mask[b, 0].view(-1)[chosen] = True

    return mask


_SLICE_MASK_MODES = frozenset({
    'spatial_mask_slice',
    'temporal_mask_slice',
    'slice_mask',
})

_N2V_MASK_MODES = frozenset({
    'n2v',
})

_DIRECTIONAL_MASK_MODES = frozenset({
    'spatial_mask',  # legacy alias error; use height_mask / width_mask
    'temporal_mask',
    'height_mask',
    'width_mask',
    'spatial_mask_mean',
    'temporal_mask_mean',
    'height_mask_mean',
}) | _SLICE_MASK_MODES | _N2V_MASK_MODES

_DUAL_CONTEXT_MASK_MODES = frozenset({
    'height_mask',
    'width_mask',
    'temporal_mask',
})

_DIRECTIONAL_MASK_MEAN_MODES = frozenset({
    'spatial_mask_mean',
    'temporal_mask_mean',
    'height_mask_mean',
})


def _resolve_slice_axes(mode: str, batch_size: int, slice_axis: str, device: torch.device) -> list:
    """Return per-batch slice axis in {'t', 'h', 'w'}."""
    if mode == 'temporal_mask_slice':
        return ['t'] * batch_size
    if mode == 'spatial_mask_slice':
        pick = torch.randint(0, 2, (batch_size,), device=device)
        return ['h' if int(v) == 0 else 'w' for v in pick.tolist()]
    if mode != 'slice_mask':
        raise ValueError('Not a slice mask mode: {}'.format(mode))

    axis = (slice_axis or 'random').lower()
    if axis in ('t', 'h', 'w'):
        return [axis] * batch_size
    if axis == 'random':
        pick = torch.randint(0, 3, (batch_size,), device=device)
        return [['t', 'h', 'w'][int(v)] for v in pick.tolist()]
    raise ValueError('slice_axis must be t, h, w, or random; got {}'.format(slice_axis))


def make_slice_mask_pair(
    noisy: torch.Tensor,
    mode: str,
    mask_ratio: float,
    min_dist: int,
    slice_axis: str = 'random',
) -> tuple:
    """Single-slice replacement self-supervision along T / H / W.

    1. Pick axis (T, H, or W) per batch item.
    2. Random interior slice along that axis.
    3. Replace the **entire** 2D slice with **one** adjacent slice (random 50/50:
       all voxels from slice-1 **or** slice+1, not a per-pixel mix).
    4. Loss is computed on **all voxels** of that replaced slice.

    ``mask_ratio`` / ``min_dist`` are ignored (kept for API compatibility with
    full-volume ``spatial_mask`` / ``temporal_mask``).

    noisy: [B, C, T, H, W]
    Returns:
        masked_input, target (original noisy), loss_mask [B, 1, T, H, W]
    """
    del mask_ratio, min_dist  # slice modes use full-plane replacement only

    if noisy.ndim != 5:
        raise ValueError('Expected noisy shape [B, C, T, H, W], got {}'.format(tuple(noisy.shape)))
    if mode not in _SLICE_MASK_MODES:
        raise ValueError('Unknown slice mask mode: {}'.format(mode))

    B, _, T, H, W = noisy.shape
    device = noisy.device
    axes = _resolve_slice_axes(mode, B, slice_axis, device)

    masked_input = noisy.clone()
    loss_mask = torch.zeros((B, 1, T, H, W), dtype=torch.bool, device=device)

    for b in range(B):
        axis = axes[b]
        if axis == 't':
            if T < 3:
                raise ValueError('Need T>=3 for temporal slice mask, got {}'.format(T))
            sl = int(torch.randint(1, T - 1, (1,), device=device).item())
            loss_mask[b, 0, sl] = True
            use_minus = bool(torch.randint(0, 2, (1,), device=device).item())
            masked_input[b, :, sl] = (
                noisy[b, :, sl - 1] if use_minus else noisy[b, :, sl + 1]
            )

        elif axis == 'h':
            if H < 3:
                raise ValueError('Need H>=3 for H-slice mask, got {}'.format(H))
            sl = int(torch.randint(1, H - 1, (1,), device=device).item())
            loss_mask[b, 0, :, sl, :] = True
            use_minus = bool(torch.randint(0, 2, (1,), device=device).item())
            masked_input[b, :, :, sl, :] = (
                noisy[b, :, :, sl - 1, :] if use_minus else noisy[b, :, :, sl + 1, :]
            )

        elif axis == 'w':
            if W < 3:
                raise ValueError('Need W>=3 for W-slice mask, got {}'.format(W))
            sl = int(torch.randint(1, W - 1, (1,), device=device).item())
            loss_mask[b, 0, :, :, sl] = True
            use_minus = bool(torch.randint(0, 2, (1,), device=device).item())
            masked_input[b, :, :, :, sl] = (
                noisy[b, :, :, :, sl - 1] if use_minus else noisy[b, :, :, :, sl + 1]
            )
        else:
            raise ValueError('Unknown slice axis: {}'.format(axis))

    return masked_input, noisy, loss_mask


def make_n2v_mask_pair(
    noisy: torch.Tensor,
) -> tuple:
    """Noise2Void-style 3D blind-spot masking.

    This follows the default N2V training behavior:

      - sample stratified blind-spot coordinates in the 3D patch;
      - keep the original noisy patch as target;
      - replace selected input voxels by a random value sampled uniformly
        from a local 3D neighborhood around the blind-spot voxel;
      - compute loss only at the selected blind-spot voxels.

    Fixed N2V defaults:

      n2v_perc_pix = 1.5
      n2v_neighborhood_radius = 5
      n2v_manipulator = 'uniform_withCP'

    Notes:
      The official N2V source uses:

          box_size = round(sqrt(100 / perc_pix))

      also for 3D data. This function intentionally keeps that behavior
      instead of replacing it by a cube-root rule.

    Args:
        noisy: [B, C, T, H, W]

    Returns:
        masked_input: [B, C, T, H, W]
        target:       original noisy patch
        loss_mask:    [B, 1, T, H, W]
    """
    if noisy.ndim != 5:
        raise ValueError(
            'Expected noisy shape [B, C, T, H, W], got {}'.format(
                tuple(noisy.shape)
            )
        )

    n2v_perc_pix = 1.5
    n2v_neighborhood_radius = 5

    B, C, T, H, W = noisy.shape
    device = noisy.device

    # Official N2V stratified coordinate rule.
    box_size = int(
        torch.round(
            torch.sqrt(
                torch.tensor(
                    100.0 / n2v_perc_pix,
                    device=device,
                )
            )
        ).item()
    )
    box_size = max(1, box_size)

    masked_input = noisy.clone()
    loss_mask = torch.zeros(
        (B, 1, T, H, W),
        dtype=torch.bool,
        device=device,
    )

    box_count_t = int(math.ceil(T / box_size))
    box_count_h = int(math.ceil(H / box_size))
    box_count_w = int(math.ceil(W / box_size))

    grid_t, grid_h, grid_w = torch.meshgrid(
        torch.arange(box_count_t, device=device),
        torch.arange(box_count_h, device=device),
        torch.arange(box_count_w, device=device),
        indexing='ij',
    )

    base_t = grid_t.flatten() * box_size
    base_h = grid_h.flatten() * box_size
    base_w = grid_w.flatten() * box_size
    num_boxes = base_t.numel()

    for b in range(B):
        # One random coordinate per stratified 3D box.
        offset_t = torch.randint(
            low=0,
            high=box_size,
            size=(num_boxes,),
            device=device,
        )
        offset_h = torch.randint(
            low=0,
            high=box_size,
            size=(num_boxes,),
            device=device,
        )
        offset_w = torch.randint(
            low=0,
            high=box_size,
            size=(num_boxes,),
            device=device,
        )

        coord_t = base_t + offset_t
        coord_h = base_h + offset_h
        coord_w = base_w + offset_w

        valid = (
            (coord_t < T)
            & (coord_h < H)
            & (coord_w < W)
        )

        coord_t = coord_t[valid]
        coord_h = coord_h[valid]
        coord_w = coord_w[valid]

        if coord_t.numel() == 0:
            raise RuntimeError(
                'No valid N2V blind-spot coordinates were sampled.'
            )

        loss_mask[
            b,
            0,
            coord_t,
            coord_h,
            coord_w,
        ] = True

        # N2V default 'uniform_withCP':
        # sample uniformly from the local sub-patch around each coordinate.
        # The center pixel is not excluded.
        low_t = torch.clamp(
            coord_t - n2v_neighborhood_radius,
            min=0,
        )
        high_t = torch.clamp(
            coord_t + n2v_neighborhood_radius,
            max=T - 1,
        )

        low_h = torch.clamp(
            coord_h - n2v_neighborhood_radius,
            min=0,
        )
        high_h = torch.clamp(
            coord_h + n2v_neighborhood_radius,
            max=H - 1,
        )

        low_w = torch.clamp(
            coord_w - n2v_neighborhood_radius,
            min=0,
        )
        high_w = torch.clamp(
            coord_w + n2v_neighborhood_radius,
            max=W - 1,
        )

        rand_t = (
            low_t
            + torch.floor(
                torch.rand(coord_t.numel(), device=device)
                * (high_t - low_t + 1).to(torch.float32)
            ).to(torch.long)
        )

        rand_h = (
            low_h
            + torch.floor(
                torch.rand(coord_h.numel(), device=device)
                * (high_h - low_h + 1).to(torch.float32)
            ).to(torch.long)
        )

        rand_w = (
            low_w
            + torch.floor(
                torch.rand(coord_w.numel(), device=device)
                * (high_w - low_w + 1).to(torch.float32)
            ).to(torch.long)
        )

        masked_input[
            b,
            :,
            coord_t,
            coord_h,
            coord_w,
        ] = noisy[
            b,
            :,
            rand_t,
            rand_h,
            rand_w,
        ]

    return masked_input, noisy, loss_mask


def make_directional_mask_pair(
    noisy: torch.Tensor,
    mode: str,
    mask_ratio: float,
    min_dist: int,
    *,
    lattice_random_phase: bool = True,
) -> tuple:
    """Full-resolution directional masking (legacy single-replacement path).

    noisy: [B, C, T, H, W]
    mode:
        height_mask   -> replace masked voxel by h neighbor.
        width_mask    -> replace masked voxel by w neighbor.
        temporal_mask -> replace masked voxel by t neighbor.
        spatial_mask  -> removed; raises (use height_mask or width_mask).

    For dual-context Gamma training, prefer make_training_mask_group +
    dual_axis replacements instead of this single-replacement helper.

    Returns:
        masked_input: full-resolution input with sparse replacement.
        target: original noisy patch.
        mask: bool tensor [B, 1, T, H, W], loss is computed only here.
    """
    if noisy.ndim != 5:
        raise ValueError('Expected noisy shape [B, C, T, H, W], got {}'.format(tuple(noisy.shape)))

    B, _, T, H, W = noisy.shape
    if mode == 'spatial_mask':
        # Legacy 4-neighbor H/W (likelihood baseline). Dual-context Gamma
        # training must use height_mask or width_mask instead.
        valid_t = (0, T)
        valid_h = (1, H - 1)
        valid_w = (1, W - 1)
        directions = [(0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
    elif mode == 'temporal_mask':
        # Need one-frame temporal border for t neighbor replacement.
        valid_t = (1, T - 1)
        valid_h = (0, H)
        valid_w = (0, W)
        directions = [(-1, 0, 0), (1, 0, 0)]
    elif mode == 'height_mask':
        # Need one-pixel h border for h-neighbor replacement.
        valid_t = (0, T)
        valid_h = (1, H - 1)
        valid_w = (0, W)
        directions = [(0, -1, 0), (0, 1, 0)]
    elif mode == 'width_mask':
        valid_t = (0, T)
        valid_h = (0, H)
        valid_w = (1, W - 1)
        directions = [(0, 0, -1), (0, 0, 1)]
    else:
        raise ValueError('Unknown directional mask mode: {}'.format(mode))

    # 3-axis lattice; spatial/temporal differ only in valid ranges and replacement.
    mask = _make_sparse_lattice_mask(
        noisy,
        mask_ratio=mask_ratio,
        min_dist=min_dist,
        valid_t=valid_t,
        valid_h=valid_h,
        valid_w=valid_w,
        lattice_random_phase=lattice_random_phase,
    )

    masked_input = noisy.clone()
    dir_id = torch.randint(
        low=0,
        high=len(directions),
        size=mask.shape,
        device=noisy.device,
    )
    for i, (dt, dh, dw) in enumerate(directions):
        submask = mask & (dir_id == i)
        if not bool(submask.any()):
            continue
        neighbor = _shift_neighbor(noisy, dt=dt, dh=dh, dw=dw)
        masked_input = torch.where(submask.expand_as(noisy), neighbor, masked_input)

    return masked_input, noisy, mask


def _spatial_random_axis_mean_replacement(x: torch.Tensor) -> torch.Tensor:
    """Per-voxel random H or W axis mean (center pixel excluded).

    H axis: mean(up, down); W axis: mean(left, right).
    """
    up = _shift_neighbor(x, dt=0, dh=-1, dw=0)
    down = _shift_neighbor(x, dt=0, dh=1, dw=0)
    left = _shift_neighbor(x, dt=0, dh=0, dw=-1)
    right = _shift_neighbor(x, dt=0, dh=0, dw=1)
    h_mean = (up + down) * 0.5
    w_mean = (left + right) * 0.5
    b, _, t, h, w = x.shape
    axis_id = torch.randint(0, 2, size=(b, 1, t, h, w), device=x.device)
    use_h = (axis_id == 0).expand_as(x)
    return torch.where(use_h, h_mean, w_mean)


def _temporal_mean_replacement(x: torch.Tensor) -> torch.Tensor:
    """Mean of 2 temporal neighbors (center frame excluded)."""
    prev_t = _shift_neighbor(x, dt=-1, dh=0, dw=0)
    next_t = _shift_neighbor(x, dt=1, dh=0, dw=0)
    return (prev_t + next_t) / 2.0


def _height_mean_replacement(x: torch.Tensor) -> torch.Tensor:
    """Mean of 2 H neighbors (center row excluded). Not random H/W."""
    up = _shift_neighbor(x, dt=0, dh=-1, dw=0)
    down = _shift_neighbor(x, dt=0, dh=1, dw=0)
    return (up + down) / 2.0


def make_directional_mask_mean_pair(
    noisy: torch.Tensor,
    mode: str,
    mask_ratio: float,
    min_dist: int,
    *,
    lattice_random_phase: bool = True,
) -> tuple:
    """Directional masking with local-mean replacement."""
    if noisy.ndim != 5:
        raise ValueError('Expected noisy shape [B, C, T, H, W], got {}'.format(tuple(noisy.shape)))

    _, _, T, H, W = noisy.shape
    if mode == 'spatial_mask_mean':
        valid_t = (0, T)
        valid_h = (1, H - 1)
        valid_w = (1, W - 1)
        replacement = _spatial_random_axis_mean_replacement(noisy)
    elif mode == 'height_mask_mean':
        valid_t = (0, T)
        valid_h = (1, H - 1)
        valid_w = (0, W)
        replacement = _height_mean_replacement(noisy)
    elif mode == 'temporal_mask_mean':
        valid_t = (1, T - 1)
        valid_h = (0, H)
        valid_w = (0, W)
        replacement = _temporal_mean_replacement(noisy)
    else:
        raise ValueError('Unknown directional mask-mean mode: {}'.format(mode))

    mask = _make_sparse_lattice_mask(
        noisy,
        mask_ratio=mask_ratio,
        min_dist=min_dist,
        valid_t=valid_t,
        valid_h=valid_h,
        valid_w=valid_w,
        lattice_random_phase=lattice_random_phase,
    )
    masked_input = torch.where(mask.expand_as(noisy), replacement, noisy)
    return masked_input, noisy, mask


