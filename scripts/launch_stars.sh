#!/usr/bin/env bash
# Three physics-star jobs. GPU 0-3 only.
#   bash launch_stars.sh eval    # val frozen July ckpts into experiments/eval_stars
#   bash launch_stars.sh train   # from-scratch, --no_resume, experiments/stars
set -euo pipefail

ROOT=/data/zhouxirou/Ours_260911
PY=/home/zxr/.conda/envs/physics/bin/python
DATA=/data/zhouxirou/All_Datasets/dataset_CI_syn_Q004_a5000_b1600/dataset_zxr_all_frequency_Q004_beta1600-1-T1000H245W245
SCRIPT="$ROOT/scripts/train_and_val_posterior_gamma.py"
CORE=/data/zhouxirou/Ours_core/experiments
MODE="${1:-eval}"
TRANS_ORDER=st
[[ "$MODE" == eval ]] && TRANS_ORDER=ts

COMMON=(
  --backbone srdtrans_v2
  --trans_order "$TRANS_ORDER"
  --patch_xy 128
  --patch_t 128
  --overlap_factor 0.75
  --val_overlap_factor 0.5
  --lr 5e-5
  --b1 0.5
  --train_datasets_size 6000
  --n_epochs 50
  --seed 1024
  --mpgn_alpha 5000
  --mpgn_beta 1600
  --mpgn_kmax 512
  --mpgn_k_tail_tol 1e-8
  --mask_loss nll
  --val_process_frames 400
  --snr_margin 50
)

run_star() {
  local name="$1" gpu="$2" hz="$3" sampling="$4" extra=("${@:5}")
  echo "===== ${MODE} ${name} GPU ${gpu} ====="
  "$PY" "$SCRIPT" \
    --datasets_path "${DATA}/${hz}/${hz}_Noisy" \
    --gt "${DATA}/${hz}/${hz}_GT/clean_${hz}_1000frames.tif" \
    --gpu "$gpu" \
    --sampling_mode "$sampling" \
    "${COMMON[@]}" \
    "${extra[@]}"
}

eval_stars() {
  mkdir -p "$ROOT/experiments/eval_stars"
  run_star star_0.1Hz_dual 0 0.1Hz height_mask \
    --pth_dir "$ROOT/experiments/eval_stars/star_0.1Hz_dual" \
    --mask_ratio 0.05 --mask_min_dist 2 \
    --kappa_mode learned_map --val_patch_batch 2 \
    --eval_ckpt "$CORE/260722_dual_context_sweep/dual_0.1Hz_height_mask/0.1Hz_Noisy_srdtrans_gamma_height_mask_srdtrans_v2/E_47_Iter_6992.pth" \
    &
  run_star star_1Hz_k100 1 1Hz temporal_mask \
    --pth_dir "$ROOT/experiments/eval_stars/star_1Hz_k100" \
    --mask_ratio 0.01 --mask_min_dist 3 \
    --kappa_mode fixed --mpgn_kappa 100 \
    --eval_ckpt "$CORE/prev/260719_gamma_sweep/gamma_1Hz_kappa100_temporal_mask/1Hz_Noisy_srdtrans_gamma_temporal_mask_srdtrans_v2/E_50_Iter_6992.pth" \
    &
  run_star star_10Hz_k100 2 10Hz temporal_mask \
    --pth_dir "$ROOT/experiments/eval_stars/star_10Hz_k100" \
    --mask_ratio 0.01 --mask_min_dist 3 \
    --kappa_mode fixed --mpgn_kappa 100 \
    --eval_ckpt "$CORE/prev/260719_gamma_sweep/gamma_10Hz_kappa100_temporal_mask/10Hz_Noisy_srdtrans_gamma_temporal_mask_srdtrans_v2/E_45_Iter_6992.pth" \
    &
  wait
  echo "===== eval done ====="
  for d in star_0.1Hz_dual star_1Hz_k100 star_10Hz_k100; do
    echo "----- $d -----"
    tail -n 3 "$ROOT/experiments/eval_stars/$d"/*_srdtrans_gamma_*/val_metrics.md 2>/dev/null || true
  done
}

train_stars() {
  mkdir -p "$ROOT/experiments/stars"
  run_star star_0.1Hz_dual 0 0.1Hz height_mask \
    --pth_dir "$ROOT/experiments/stars/star_0.1Hz_dual" \
    --no_resume \
    --mask_ratio 0.05 --mask_min_dist 2 \
    --kappa_mode learned_map --val_patch_batch 2 \
    &
  run_star star_1Hz_k100 1 1Hz temporal_mask \
    --pth_dir "$ROOT/experiments/stars/star_1Hz_k100" \
    --no_resume \
    --mask_ratio 0.01 --mask_min_dist 3 \
    --kappa_mode fixed --mpgn_kappa 100 \
    &
  run_star star_10Hz_k100 2 10Hz temporal_mask \
    --pth_dir "$ROOT/experiments/stars/star_10Hz_k100" \
    --no_resume \
    --mask_ratio 0.01 --mask_min_dist 3 \
    --kappa_mode fixed --mpgn_kappa 100 \
    &
  wait
}

case "$MODE" in
  eval) eval_stars ;;
  train) train_stars ;;
  *)
    echo "usage: $0 eval|train" >&2
    exit 1
    ;;
esac
