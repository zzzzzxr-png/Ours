import argparse
import csv
import math
from pathlib import Path

import numpy as np
import tifffile as tiff
from skimage.metrics import structural_similarity

# Metric definitions are taken from the "Evaluation metrics" section of
# "Spatial redundancy transformer for self-supervised fluorescence image
# denoising" and its supplementary material in the project root. The paper
# prints a full-image SSIM formula, so this script exposes both:
# 1) ssim_paper: direct implementation of the printed equation
# 2) ssim_local: standard sliding-window SSIM averaged over frames
#
# The supplementary material also repeatedly refers to "average SNR" and
# reports statistics over independent frames, so for stacks we report both
# whole-stack metrics and frame-mean metrics.


PAPER_K1 = 0.01
PAPER_K2 = 0.03
PAPER_DATA_RANGE = 65535.0
EPS = 1e-12


def load_tiff(path, max_frames=None):
    array = np.asarray(tiff.imread(str(path)))
    array = np.squeeze(array)
    if max_frames is not None and array.ndim == 3 and array.shape[0] > max_frames:
        array = array[:max_frames, :, :]
    return array


def validate_pair(pred, gt, pred_path, gt_path):
    if pred.shape != gt.shape:
        raise ValueError(
            "Shape mismatch between prediction and GT: "
            f"{pred_path} -> {pred.shape}, {gt_path} -> {gt.shape}"
        )


def resolve_data_range(data_range_arg, gt):
    if isinstance(data_range_arg, str) and data_range_arg.lower() == "auto":
        gt = np.asarray(gt, dtype=np.float64)
        value = float(gt.max() - gt.min())
        return value if value > 0 else 1.0
    return float(data_range_arg)


def compute_snr(pred, gt):
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    signal_power = np.sum(np.square(gt))
    noise_power = np.sum(np.square(gt - pred))
    if noise_power <= EPS:
        return float("inf")
    return 10.0 * math.log10((signal_power + EPS) / (noise_power + EPS))


def compute_frame_mean_metric(pred, gt, metric_fn, **kwargs):
    pred = np.asarray(pred)
    gt = np.asarray(gt)

    if pred.ndim == 2:
        return float(metric_fn(pred, gt, **kwargs))

    if pred.ndim == 3:
        scores = [
            float(metric_fn(pred[idx], gt[idx], **kwargs))
            for idx in range(pred.shape[0])
        ]
        return float(np.mean(scores))

    raise ValueError(
        "Frame-mean metrics currently support 2D images or 3D stacks with shape "
        f"(T, H, W), but got {pred.shape}."
    )


def compute_paper_ssim(pred, gt, data_range=PAPER_DATA_RANGE, k1=PAPER_K1, k2=PAPER_K2):
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)

    mu_x = pred.mean()
    mu_y = gt.mean()
    sigma_x2 = np.mean((pred - mu_x) ** 2)
    sigma_y2 = np.mean((gt - mu_y) ** 2)
    sigma_xy = np.mean((pred - mu_x) * (gt - mu_y))

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x2 + sigma_y2 + c2)

    if abs(denominator) <= EPS:
        return 1.0 if abs(numerator) <= EPS else 0.0
    return numerator / denominator


def _resolve_win_size(image_shape, requested=None):
    min_dim = min(image_shape)
    if min_dim < 3:
        raise ValueError(
            "Local SSIM needs spatial dimensions >= 3, "
            f"but got shape {image_shape}."
        )

    if requested is not None:
        if requested % 2 == 0:
            raise ValueError(f"win_size must be odd, but got {requested}.")
        if requested > min_dim:
            raise ValueError(
                f"win_size={requested} is larger than the smallest image "
                f"dimension {min_dim}."
            )
        return requested

    return min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)


def _compute_local_ssim_2d(pred, gt, data_range, win_size=None):
    resolved_win_size = _resolve_win_size(gt.shape[-2:], requested=win_size)
    return structural_similarity(
        gt,
        pred,
        data_range=data_range,
        win_size=resolved_win_size,
        gaussian_weights=False,
        use_sample_covariance=False,
        K1=PAPER_K1,
        K2=PAPER_K2,
    )


def compute_local_ssim(pred, gt, data_range=PAPER_DATA_RANGE, win_size=None):
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)

    if pred.ndim == 2:
        return _compute_local_ssim_2d(pred, gt, data_range, win_size=win_size)

    if pred.ndim == 3:
        scores = [
            _compute_local_ssim_2d(pred[idx], gt[idx], data_range, win_size=win_size)
            for idx in range(pred.shape[0])
        ]
        return float(np.mean(scores))

    raise ValueError(
        "Local SSIM currently supports 2D images or 3D stacks with shape "
        f"(T, H, W), but got {pred.shape}."
    )


def normalize_stem(path):
    stem = path.stem
    if stem.endswith("_output"):
        stem = stem[:-7]
    return stem


def find_matching_gt(pred_path, gt_files):
    gt_by_stem = {path.stem: path for path in gt_files}
    normalized_gt_by_stem = {normalize_stem(path): path for path in gt_files}

    pred_stem = pred_path.stem
    normalized_pred_stem = normalize_stem(pred_path)

    if pred_stem in gt_by_stem:
        return gt_by_stem[pred_stem]
    if normalized_pred_stem in gt_by_stem:
        return gt_by_stem[normalized_pred_stem]
    if pred_stem in normalized_gt_by_stem:
        return normalized_gt_by_stem[pred_stem]
    if normalized_pred_stem in normalized_gt_by_stem:
        return normalized_gt_by_stem[normalized_pred_stem]

    prefix_matches = []
    for gt_path in gt_files:
        gt_stem = gt_path.stem
        normalized_gt_stem = normalize_stem(gt_path)
        for candidate in (gt_stem, normalized_gt_stem):
            if (
                pred_stem.startswith(candidate + "_")
                or normalized_pred_stem.startswith(candidate + "_")
            ):
                prefix_matches.append((len(candidate), gt_path))

    if not prefix_matches:
        raise FileNotFoundError(f"Could not find GT file for {pred_path.name}.")

    prefix_matches.sort(key=lambda item: item[0], reverse=True)
    top_length = prefix_matches[0][0]
    top_matches = [path for length, path in prefix_matches if length == top_length]
    unique_paths = list(dict.fromkeys(top_matches))
    if len(unique_paths) > 1:
        names = ", ".join(path.name for path in unique_paths)
        raise FileNotFoundError(
            f"Ambiguous GT match for {pred_path.name}. Candidates: {names}"
        )
    return unique_paths[0]


def collect_tiff_files(path):
    suffixes = {".tif", ".tiff"}
    return sorted(
        file_path for file_path in path.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in suffixes
    )


def evaluate_pair(pred_path, gt_path, data_range_arg, win_size=None, max_frames=None):
    pred = load_tiff(pred_path, max_frames=max_frames)
    gt = load_tiff(gt_path, max_frames=max_frames)
    validate_pair(pred, gt, pred_path, gt_path)
    data_range = resolve_data_range(data_range_arg, gt)

    snr_global_db = compute_snr(pred, gt)
    snr_frame_mean_db = compute_frame_mean_metric(pred, gt, compute_snr)
    ssim_paper_global = compute_paper_ssim(pred, gt, data_range=data_range)
    ssim_paper_frame_mean = compute_frame_mean_metric(
        pred, gt, compute_paper_ssim, data_range=data_range
    )
    ssim_local = compute_local_ssim(pred, gt, data_range=data_range, win_size=win_size)

    return {
        "pred_path": str(pred_path),
        "gt_path": str(gt_path),
        "shape": tuple(int(dim) for dim in pred.shape),
        "snr_global_db": snr_global_db,
        "snr_frame_mean_db": snr_frame_mean_db,
        "ssim_paper_global": ssim_paper_global,
        "ssim_paper_frame_mean": ssim_paper_frame_mean,
        "ssim_local": ssim_local,
        "data_range": data_range,
    }


def summarize_rows(rows):
    return {
        "pair_count": len(rows),
        "mean_snr_global_db": float(np.mean([row["snr_global_db"] for row in rows])),
        "mean_snr_frame_mean_db": float(np.mean([row["snr_frame_mean_db"] for row in rows])),
        "mean_ssim_paper_global": float(np.mean([row["ssim_paper_global"] for row in rows])),
        "mean_ssim_paper_frame_mean": float(np.mean([row["ssim_paper_frame_mean"] for row in rows])),
        "mean_ssim_local": float(np.mean([row["ssim_local"] for row in rows])),
    }


def print_row(row):
    print(f"Prediction : {row['pred_path']}")
    print(f"GT         : {row['gt_path']}")
    print(f"Shape      : {row['shape']}")
    print(f"SNR global : {row['snr_global_db']:.6f}")
    print(f"SNR frame  : {row['snr_frame_mean_db']:.6f}")
    print(f"SSIM pgbl  : {row['ssim_paper_global']:.6f}")
    print(f"SSIM pfrm  : {row['ssim_paper_frame_mean']:.6f}")
    print(f"SSIM local : {row['ssim_local']:.6f}")
    print(f"Data range : {row['data_range']}")


def save_csv(rows, summary, output_path):
    fieldnames = [
        "pred_path",
        "gt_path",
        "shape",
        "snr_global_db",
        "snr_frame_mean_db",
        "ssim_paper_global",
        "ssim_paper_frame_mean",
        "ssim_local",
        "data_range",
    ]
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        writer.writerow({})
        writer.writerow({
            "pred_path": "__mean__",
            "snr_global_db": summary["mean_snr_global_db"],
            "snr_frame_mean_db": summary["mean_snr_frame_mean_db"],
            "ssim_paper_global": summary["mean_ssim_paper_global"],
            "ssim_paper_frame_mean": summary["mean_ssim_paper_frame_mean"],
            "ssim_local": summary["mean_ssim_local"],
            "data_range": rows[0]["data_range"] if rows else "",
        })


def build_pairs(pred_path, gt_path):
    if pred_path.is_file() and gt_path.is_file():
        return [(pred_path, gt_path)]

    if pred_path.is_dir() and gt_path.is_dir():
        pred_files = collect_tiff_files(pred_path)
        gt_files = collect_tiff_files(gt_path)
        if not pred_files:
            raise FileNotFoundError(f"No TIFF files found in {pred_path}.")
        if not gt_files:
            raise FileNotFoundError(f"No TIFF files found in {gt_path}.")
        return [(pred_file, find_matching_gt(pred_file, gt_files)) for pred_file in pred_files]

    raise ValueError(
        "--pred and --gt must both be files or both be directories."
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute SNR/SSIM for SRDTrans results. "
            "The SNR formula and SSIM constants follow the paper: "
            "k1=0.01, k2=0.03, L=65535."
        )
    )
    parser.add_argument("--pred", required=True, help="Prediction TIFF file or folder.")
    parser.add_argument("--gt", required=True, help="Ground-truth TIFF file or folder.")
    parser.add_argument(
        "--data_range",
        default=str(int(PAPER_DATA_RANGE)),
        help="SSIM data range. Use 65535 to match the paper, or 'auto' to infer from GT.",
    )
    parser.add_argument(
        "--win_size",
        type=int,
        default=None,
        help="Odd local SSIM window size. Defaults to min(7, min(H, W)).",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional path to save per-pair metrics as CSV.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Only evaluate the first N frames for 3D stacks shaped as (T, H, W).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pred_path = Path(args.pred)
    gt_path = Path(args.gt)

    pairs = build_pairs(pred_path, gt_path)
    rows = [
        evaluate_pair(
            pred,
            gt,
            args.data_range,
            win_size=args.win_size,
            max_frames=args.max_frames,
        )
        for pred, gt in pairs
    ]

    if len(rows) == 1:
        print_row(rows[0])
    else:
        for row in rows:
            print_row(row)
            print("-" * 80)
        summary = summarize_rows(rows)
        print(f"Pairs      : {summary['pair_count']}")
        print(f"Mean Sgbl  : {summary['mean_snr_global_db']:.6f}")
        print(f"Mean Sfrm  : {summary['mean_snr_frame_mean_db']:.6f}")
        print(f"Mean Pgbl  : {summary['mean_ssim_paper_global']:.6f}")
        print(f"Mean Pfrm  : {summary['mean_ssim_paper_frame_mean']:.6f}")
        print(f"Mean local : {summary['mean_ssim_local']:.6f}")

    if args.csv is not None:
        summary = summarize_rows(rows)
        save_csv(rows, summary, args.csv)
        print(f"CSV saved  : {args.csv}")


if __name__ == "__main__":
    main()
