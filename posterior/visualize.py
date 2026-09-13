import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch


MPG_MAGENTA = LinearSegmentedColormap.from_list(
    'mpg_magenta',
    [
        '#08000f',
        '#33003d',
        '#85008f',
        '#ed00e8',
        '#ffd7ff',
    ],
)


def _tensor_to_phys(
    tensor,
    patch_mean,
    sample_index,
):
    if not torch.is_tensor(tensor):
        tensor = torch.as_tensor(tensor)

    tensor = tensor.detach().float().cpu()

    if tensor.ndim == 5:
        tensor = tensor[sample_index, 0]
    elif tensor.ndim == 4:
        tensor = tensor[sample_index]

    if torch.is_tensor(patch_mean):
        mean_value = float(
            patch_mean.detach().float().cpu().reshape(-1)[
                min(sample_index, patch_mean.numel() - 1)
            ].item()
        )
    else:
        mean_value = float(patch_mean)

    return tensor.numpy() + mean_value


def _snr(pred, gt):
    pred = pred.astype(np.float64)
    gt = gt.astype(np.float64)

    numerator = np.sum(gt ** 2)
    denominator = np.sum((pred - gt) ** 2)

    if denominator <= 0:
        return float('inf')

    return 10.0 * math.log10(
        max(numerator, 1e-20) / denominator
    )


def save_unroll_stage_figure(
    save_path,
    debug_dict,
    patch_mean,
    gt_patch_phys,
    t_index=-1,
    sample_index=0,
    use_physical_units=True,
):
    stages = debug_dict['stages']

    y_masked = _tensor_to_phys(
        debug_dict['y_masked'],
        patch_mean,
        sample_index,
    )
    y_raw_key = 'y_raw' if 'y_raw' in debug_dict else 'y_obs'
    y_raw = _tensor_to_phys(
        debug_dict[y_raw_key],
        patch_mean,
        sample_index,
    )

    x0_key = 'x0_prior' if 'x0_prior' in debug_dict else 'x0'
    x0 = _tensor_to_phys(
        debug_dict[x0_key],
        patch_mean,
        sample_index,
    )

    gt = np.asarray(gt_patch_phys, dtype=np.float32)

    common_t = min(x0.shape[0], gt.shape[0], y_masked.shape[0], y_raw.shape[0])

    if t_index < 0:
        t_index = common_t // 2
    t_index = min(max(int(t_index), 0), common_t - 1)

    vmin = float(np.percentile(gt, 1.0))
    vmax = float(np.percentile(gt, 99.5))
    if vmax <= vmin:
        vmax = vmin + 1e-6

    delta_volumes = []
    for stage in stages:
        delta_volumes.append(
            np.abs(
                _tensor_to_phys(
                    stage['delta_correction'],
                    patch_mean=0.0,
                    sample_index=sample_index,
                )
            )
        )
        delta_volumes.append(
            np.abs(
                _tensor_to_phys(
                    stage['delta_prior'],
                    patch_mean=0.0,
                    sample_index=sample_index,
                )
            )
        )

    delta_values = np.concatenate(
        [delta.reshape(-1) for delta in delta_volumes]
    )
    delta_max = max(
        float(np.percentile(delta_values, 99.0)),
        1e-8,
    )

    # y_masked, y_raw, x0 + per stage: x_prev, z_corr, x_prior, |HG|, |denoise|.
    ncols = 3 + 5 * len(stages)

    fig, axes = plt.subplots(
        2,
        ncols,
        figsize=(1.8 * ncols, 4.8),
        squeeze=False,
    )

    for ax in axes.reshape(-1):
        ax.axis('off')

    axes[0, 0].imshow(
        y_masked[t_index],
        cmap=MPG_MAGENTA,
        vmin=vmin,
        vmax=vmax,
    )
    axes[0, 0].set_title(
        'y_masked\n(prior in)\nSNR={:.2f} dB'.format(_snr(y_masked, gt)),
        fontsize=8,
    )

    axes[0, 1].imshow(
        y_raw[t_index],
        cmap=MPG_MAGENTA,
        vmin=vmin,
        vmax=vmax,
    )
    axes[0, 1].set_title(
        'y_raw\n(HG corr.)\nSNR={:.2f} dB'.format(_snr(y_raw, gt)),
        fontsize=8,
    )

    axes[0, 2].imshow(
        x0[t_index],
        cmap=MPG_MAGENTA,
        vmin=vmin,
        vmax=vmax,
    )
    axes[0, 2].set_title(
        'x0\nwarm-start\nSNR={:.2f} dB'.format(_snr(x0, gt)),
        fontsize=9,
    )

    axes[1, 0].axis('off')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(
        gt[t_index],
        cmap=MPG_MAGENTA,
        vmin=vmin,
        vmax=vmax,
    )
    axes[1, 2].set_title('GT', fontsize=9)

    for stage_index, stage in enumerate(stages):
        k = stage_index + 1
        base_col = 3 + 5 * stage_index

        x_prev_volume = _tensor_to_phys(
            stage['x_prev'],
            patch_mean,
            sample_index,
        )
        z_corr_volume = _tensor_to_phys(
            stage['z_corr'],
            patch_mean,
            sample_index,
        )
        x_prior_volume = _tensor_to_phys(
            stage['x_prior'],
            patch_mean,
            sample_index,
        )

        delta_hg = np.abs(
            _tensor_to_phys(
                stage['delta_correction'],
                patch_mean=0.0,
                sample_index=sample_index,
            )
        )
        delta_prior = np.abs(
            _tensor_to_phys(
                stage['delta_prior'],
                patch_mean=0.0,
                sample_index=sample_index,
            )
        )

        eta_key = 'eta' if 'eta' in stage else 'rho'
        eta = float(
            stage[eta_key]
            .detach()
            .float()
            .cpu()
            .mean()
            .item()
        )

        axes[0, base_col].imshow(
            x_prev_volume[t_index],
            cmap=MPG_MAGENTA,
            vmin=vmin,
            vmax=vmax,
        )
        axes[0, base_col].set_title(
            'x{}\nprior\nSNR={:.2f}'.format(
                k - 1,
                _snr(x_prev_volume, gt),
            ),
            fontsize=7,
        )

        axes[0, base_col + 1].imshow(
            z_corr_volume[t_index],
            cmap=MPG_MAGENTA,
            vmin=vmin,
            vmax=vmax,
        )
        axes[0, base_col + 1].set_title(
            'z{}\nHG corr.\nSNR={:.2f}'.format(
                k,
                _snr(z_corr_volume, gt),
            ),
            fontsize=7,
        )

        axes[0, base_col + 2].imshow(
            x_prior_volume[t_index],
            cmap=MPG_MAGENTA,
            vmin=vmin,
            vmax=vmax,
        )
        axes[0, base_col + 2].set_title(
            'x{}\nafter D\neta={:.1e}\nSNR={:.2f}'.format(
                k,
                eta,
                _snr(x_prior_volume, gt),
            ),
            fontsize=7,
        )

        axes[1, base_col].imshow(
            delta_hg[t_index],
            cmap=MPG_MAGENTA,
            vmin=0,
            vmax=delta_max,
        )
        axes[1, base_col].set_title(
            '|z{}-x{}|'.format(k, k - 1),
            fontsize=7,
        )

        axes[1, base_col + 1].imshow(
            delta_prior[t_index],
            cmap=MPG_MAGENTA,
            vmin=0,
            vmax=delta_max,
        )
        axes[1, base_col + 1].set_title(
            '|x{}-z{}|'.format(k, k),
            fontsize=7,
        )

    fig.suptitle(
        'End-to-end HG-MPGN unfolding: HG correction / denoiser refinement',
        fontsize=13,
    )

    plt.tight_layout(rect=(0, 0, 1, 0.94))
    fig.subplots_adjust(hspace=fig.subplotpars.hspace * 2.0)
    fig.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
