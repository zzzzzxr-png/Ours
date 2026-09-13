"""Diagnostics for the posterior (unroll-transformer) pipeline.

Checkpoint loading for the pretrained prior, SNR metrics, train/val
centered-to-physical coordinate consistency checks, and the shifted-Poisson
deviance auxiliary loss.
"""

import math

import numpy as np
import torch

def _strip_prior_checkpoint_key(key):
    """Normalize checkpoint keys for loading into PriorNet."""
    key = str(key)
    for prefix in ('module.', 'model.', 'net.', 'network.', 'PriorNet.'):
        if key.startswith(prefix):
            key = key[len(prefix):]
    return key


def load_pretrained_prior(prior_net, ckpt_path, verbose=True):
    """
    Load a standalone SRDTrans checkpoint into unroll_model.PriorNet with diagnostics.

    Returns a dict with load statistics.
    """
    raw = torch.load(ckpt_path, map_location='cpu')
    if isinstance(raw, dict) and 'model_state_dict' in raw:
        raw = raw['model_state_dict']

    cleaned = {}
    for key, value in raw.items():
        cleaned[_strip_prior_checkpoint_key(key)] = value

    target_sd = prior_net.state_dict()
    filtered = {}
    shape_mismatch = []
    unexpected = []

    for key, value in cleaned.items():
        if key not in target_sd:
            unexpected.append(key)
            continue
        if tuple(target_sd[key].shape) != tuple(value.shape):
            shape_mismatch.append(
                (key, tuple(value.shape), tuple(target_sd[key].shape))
            )
            continue
        filtered[key] = value

    msg = prior_net.load_state_dict(filtered, strict=False)
    missing = list(msg.missing_keys)
    loaded_count = len(filtered)

    stats = {
        'ckpt_path': ckpt_path,
        'checkpoint_tensors': len(cleaned),
        'prior_tensors': len(target_sd),
        'loaded_tensors': loaded_count,
        'missing_keys': missing,
        'unexpected_keys': unexpected,
        'shape_mismatch_keys': shape_mismatch,
    }

    if verbose:
        print('\033[1;36m[Prior init] load_pretrained_prior summary\033[0m')
        print('  checkpoint path:', ckpt_path)
        print('  checkpoint tensors:', stats['checkpoint_tensors'])
        print('  PriorNet tensors:', stats['prior_tensors'])
        print('  loaded tensors:', stats['loaded_tensors'])
        print('  missing keys ({:d}):'.format(len(missing)), missing[:20])
        if len(missing) > 20:
            print('    ... and {:d} more'.format(len(missing) - 20))
        print('  unexpected keys ({:d}):'.format(len(unexpected)), unexpected[:20])
        if len(unexpected) > 20:
            print('    ... and {:d} more'.format(len(unexpected) - 20))
        print('  shape-mismatch keys ({:d}):'.format(len(shape_mismatch)))
        for item in shape_mismatch[:20]:
            print('    {} ckpt{} vs model{}'.format(item[0], item[1], item[2]))
        if len(shape_mismatch) > 20:
            print('    ... and {:d} more'.format(len(shape_mismatch) - 20))

    return stats


def cal_snr(noisy_img: np.ndarray, clean_img: np.ndarray) -> float:
    noise_signal_2 = (noisy_img.astype(np.float32) - clean_img.astype(np.float32)) ** 2
    clean_signal_2 = clean_img.astype(np.float32) ** 2
    sum1 = float(clean_signal_2.sum())
    sum2 = float(noise_signal_2.sum())
    if sum2 <= 0:
        return float('inf')
    return 20 * math.log10(math.sqrt(sum1) / math.sqrt(sum2))


def cal_snr_scaled(pred_img: np.ndarray, clean_img: np.ndarray) -> float:
    """SNR after a global least-squares intensity scale on pred."""
    pred = pred_img.astype(np.float64)
    gt = clean_img.astype(np.float64)
    denom = float(np.sum(pred * pred))
    if denom <= 0:
        return cal_snr(pred_img, clean_img)
    scale = float(np.sum(pred * gt) / denom)
    return cal_snr(pred * scale, clean_img)


def _format_float_for_path(x) -> str:
    """Make a float safe and readable in experiment folder names."""
    s = "{:g}".format(float(x))
    s = s.replace("-", "m")
    s = s.replace("+", "")
    s = s.replace(".", "p")
    return s


def _accumulate_val_patch(output_centered, residual_mean, noise_patch_raw, single_coordinate,
                          denoise_before_match, denoise_img, img_mean, input_img=None):
    """Stitch one validation patch into full-volume buffers (centered -> physical patch units)."""
    fake_B = output_centered + residual_mean
    output_image = np.squeeze(fake_B.detach().cpu().numpy())
    raw_image = np.squeeze(noise_patch_raw.detach().cpu().numpy())

    if output_image.ndim == 3:
        postprocess_turn = 1
    else:
        postprocess_turn = output_image.shape[0]

    if postprocess_turn > 1:
        for patch_id in range(postprocess_turn):
            output_patch, raw_patch, \
                stack_start_w, stack_end_w, \
                stack_start_h, stack_end_h, \
                stack_start_s, stack_end_s = multibatch_test_save_srdtrans(
                    single_coordinate, patch_id, output_image, raw_image)
            output_patch = output_patch + img_mean
            raw_patch = raw_patch + img_mean
            denoise_before_match[stack_start_s:stack_end_s,
                                 stack_start_h:stack_end_h,
                                 stack_start_w:stack_end_w] = output_patch
            denoise_img[stack_start_s:stack_end_s,
                        stack_start_h:stack_end_h,
                        stack_start_w:stack_end_w] = \
                output_patch * (np.sum(raw_patch) / np.sum(output_patch)) ** 0.5
            if input_img is not None:
                input_img[stack_start_s:stack_end_s,
                          stack_start_h:stack_end_h,
                          stack_start_w:stack_end_w] = raw_patch
    else:
        output_patch, raw_patch, \
            stack_start_w, stack_end_w, \
            stack_start_h, stack_end_h, \
            stack_start_s, stack_end_s = singlebatch_test_save_srdtrans(
                single_coordinate, output_image, raw_image)
        output_patch = output_patch + img_mean
        raw_patch = raw_patch + img_mean
        denoise_before_match[stack_start_s:stack_end_s,
                             stack_start_h:stack_end_h,
                             stack_start_w:stack_end_w] = output_patch
        denoise_img[stack_start_s:stack_end_s,
                    stack_start_h:stack_end_h,
                    stack_start_w:stack_end_w] = \
            output_patch * (np.sum(raw_patch) / np.sum(output_patch)) ** 0.5
        if input_img is not None:
            input_img[stack_start_s:stack_end_s,
                      stack_start_h:stack_end_h,
                      stack_start_w:stack_end_w] = raw_patch

def _check_train_coordinate_consistency(sub_centered, sub_raw, patch_mean):
    reconstructed_phys = sub_centered + patch_mean
    expected_phys = sub_raw
    coord_error = (reconstructed_phys - expected_phys).abs().max()
    print(
        '[Train coordinate check] '
        f'max_abs_error={coord_error.item():.6g}'
    )
    if coord_error.item() > 1e-3:
        raise RuntimeError(
            'Training centered-to-physical conversion is inconsistent.'
        )


def _check_val_coordinate_consistency(
    real_A,
    model_mean,
    dataset_mean,
    noise_patch_raw,
):
    reconstructed_phys = real_A + model_mean
    expected_phys = noise_patch_raw + dataset_mean
    coord_error = (reconstructed_phys - expected_phys).abs().max()
    print(
        '[Validation coordinate check] '
        f'max_abs_error={coord_error.item():.6g}'
    )
    if coord_error.item() > 1e-3:
        raise RuntimeError(
            'Validation centered-to-physical conversion is inconsistent.'
        )


# Ported verbatim from
# /data/zhouxirou/SRDTrans_3d/260429_train_and_val_collection_likelihood.py.

def _to_phys_units(x, img_mean):
    """Convert centered tensor back to acquisition/physical units.

    Supports scalar mean or per-patch tensor mean.

    Expected centered coordinate:
        x_centered = x_scaled - patch_mean
        x_scaled   = x_phys

    Therefore:
        x_phys = x_centered + patch_mean
    """
    if torch.is_tensor(img_mean):
        img_mean = img_mean.to(dtype=x.dtype, device=x.device)
    else:
        img_mean = torch.as_tensor(img_mean, dtype=x.dtype, device=x.device)

    return x + img_mean


def _shifted_poisson_deviance_masked(
    pred_tensor,
    target_tensor,
    loss_mask,
    pred_img_mean,
    target_img_mean,
    alpha,
    beta,
    offset=0.0,
    eps=1e-8,
):
    """
    Shifted-Poisson deviance in count units, evaluated only on masked voxels.

    Sparse gather: only masked positions enter the log/deviance computation.

    q = (target_phys - offset + beta/alpha) / alpha
    u = (pred_phys   - offset + beta/alpha) / alpha

    D = 2 * [u - q + q * log(q/u)]
    """
    dtype = pred_tensor.dtype
    device = pred_tensor.device

    alpha_t = torch.as_tensor(alpha, dtype=dtype, device=device)
    beta_t = torch.as_tensor(beta, dtype=dtype, device=device)
    offset_t = torch.as_tensor(offset, dtype=dtype, device=device)
    eps_t = torch.as_tensor(eps, dtype=dtype, device=device)

    mask = loss_mask.to(device=device).bool()

    pred_phys = _to_phys_units(
        pred_tensor,
        pred_img_mean,
    )
    target_phys = _to_phys_units(
        target_tensor,
        target_img_mean,
    )

    pred_m = pred_phys[mask]
    target_m = target_phys[mask]

    if pred_m.numel() == 0:
        return pred_tensor.sum() * 0.0

    alpha_safe = alpha_t.clamp_min(eps_t)
    shift_phys = beta_t / alpha_safe

    u = (pred_m - offset_t + shift_phys) / alpha_safe
    q = (target_m - offset_t + shift_phys) / alpha_safe

    u = u.clamp_min(float(eps))
    q = q.clamp_min(float(eps))

    dev = 2.0 * (u - q + q * (torch.log(q) - torch.log(u)))
    return dev.mean()
