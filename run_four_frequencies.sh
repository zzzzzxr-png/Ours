#!/usr/bin/env bash
set -uo pipefail

ROOT=/data/zhouxirou/Ours_260911
DATA=/data/zhouxirou/All_Datasets/dataset_CI_syn_Q004_a5000_b1600/dataset_zxr_all_frequency_Q004_beta1600-1-T1000H245W245
PY=/home/zxr/.conda/envs/physics/bin/python
SCRIPT="$ROOT/scripts/train_and_val_posterior_gamma.py"
RUNS="$ROOT/experiments/complex_gamma_hilbert_shared"
LOGS="$ROOT/logs/complex_gamma_hilbert_shared"

FREQUENCIES=(0.1Hz 1Hz 10Hz 30Hz)
SAMPLING=(height_mask temporal_mask temporal_mask temporal_mask)
MASK_RATIO=(0.05 0.01 0.01 0.01)
MASK_MIN_DIST=(2 3 3 3)
mkdir -p "$RUNS" "$LOGS"

pids=()
for gpu in "${!FREQUENCIES[@]}"; do
  hz="${FREQUENCIES[$gpu]}"
  "$PY" "$SCRIPT" \
    --datasets_path "$DATA/$hz/${hz}_Noisy" \
    --gt "$DATA/$hz/${hz}_GT/clean_${hz}_1000frames.tif" \
    --pth_dir "$RUNS/$hz" \
    --gpu "$gpu" \
    --backbone srdtrans_v2 \
    --srdtrans-root "$ROOT/prior/srdtrans/SRDTrans_v2" \
    --sampling_mode "${SAMPLING[$gpu]}" \
    --mask_ratio "${MASK_RATIO[$gpu]}" \
    --mask_min_dist "${MASK_MIN_DIST[$gpu]}" \
    --kappa_mode fixed \
    --mpgn_kappa 100 \
    --mask_loss nll \
    --patch_xy 128 \
    --patch_t 128 \
    --overlap_factor 0.75 \
    --val_overlap_factor 0.5 \
    --n_epochs 50 \
    --train_datasets_size 6000 \
    --lr 5e-5 \
    --b1 0.5 \
    --b2 0.999 \
    --trans_order st \
    --mpgn_alpha 5000 \
    --mpgn_beta 1600 \
    --mpgn_kmax 512 \
    --mpgn_k_tail_tol 1e-8 \
    --checkpoint-every-epochs 5 \
    --validation-every-epochs 1 \
    --val_process_frames 400 \
    --snr_margin 50 \
    --seed 1024 \
    >"$LOGS/$hz.log" 2>&1 &
  pids+=("$!")
  echo "started $hz on CUDA $gpu (pid ${pids[-1]})"
done

status=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "${FREQUENCIES[$index]} failed; see $LOGS/${FREQUENCIES[$index]}.log" >&2
    status=1
  fi
done
exit "$status"
