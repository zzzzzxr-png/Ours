#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/zhouxirou/Ours_260911
DATA_ROOT=/data/zhouxirou/All_Datasets/dataset_CI_syn_Q004_a5000_b1600/dataset_zxr_all_frequency_Q004_beta1600-1-T1000H245W245

common_args=(
  --patch_xy 128 --patch_t 128
  --overlap_factor 0.75 --val_overlap_factor 0.5
  --n_epochs 100 --lr 5e-5 --train_datasets_size 6000
  --sampling_mode temporal_mask --backbone srdtrans_v2 --fmap 16
  --embedding-dim 128 --num-heads 8 --hidden-dim 512 --window-size 7
  --num-trans-block 1 --attn-dropout-rate 0.1 --input-dropout-rate 0
  --srdtrans-f-maps 8,16,32,64 --select_img_num 100000
  --mask_ratio 0.01 --mask_min_dist 3 --trans_order st --mask_loss nll
  --kappa_mode fixed --mpgn_alpha 5000 --mpgn_beta 1600 --mpgn_kappa 100
  --mpgn_offset 0 --mpgn_kmax 512 --mpgn_nll_chunk_t 8
  --mpgn_k_tail_tol 1e-8 --num_workers 4 --val_process_frames 400
  --checkpoint-every-epochs 5 --validation-every-epochs 1 --no_resume
)

mkdir -p "$ROOT/logs/parallel_10hz_01hz"

bash "$ROOT/scripts/run_multigpu_posterior_gamma.sh" 0,1,2,3 \
  --gpu 0,1,2,3 \
  --datasets_path "$DATA_ROOT/10Hz/10Hz_Noisy" \
  --gt "$DATA_ROOT/10Hz/10Hz_GT/clean_10Hz_1000frames.tif" \
  --pth_dir "$ROOT/experiments/complex_gamma_hilbert_shared_10Hz_lr5e-5_fixed" \
  "${common_args[@]}" \
  > "$ROOT/logs/parallel_10hz_01hz/10Hz.log" 2>&1 &
pid_10hz=$!

bash "$ROOT/scripts/run_multigpu_posterior_gamma.sh" 4,5,6,7 \
  --gpu 4,5,6,7 \
  --datasets_path "$DATA_ROOT/0.1Hz/0.1Hz_Noisy" \
  --gt "$DATA_ROOT/0.1Hz/0.1Hz_GT/clean_0.1Hz_1000frames.tif" \
  --pth_dir "$ROOT/experiments/complex_gamma_hilbert_shared_0.1Hz_lr5e-5_fixed" \
  "${common_args[@]}" \
  > "$ROOT/logs/parallel_10hz_01hz/0.1Hz.log" 2>&1 &
pid_01hz=$!

echo "Started 10Hz PID=$pid_10hz on CUDA 0,1,2,3"
echo "Started 0.1Hz PID=$pid_01hz on CUDA 4,5,6,7"
wait "$pid_10hz" "$pid_01hz"
