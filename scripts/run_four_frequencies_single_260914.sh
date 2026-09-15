#!/usr/bin/env bash
set -euo pipefail

cd /data/zhouxirou/Ours_260911

DATA_ROOT=/data/zhouxirou/All_Datasets
DATASET=dataset_CI_syn_Q004_a5000_b1600
STACK=dataset_zxr_all_frequency_Q004_beta1600-1-T1000H245W245
DATA="$DATA_ROOT/$DATASET/$STACK"
PYTHON=/home/zxr/.conda/envs/physics/bin/python

run_one() {
    local gpu="$1" freq="$2" tag="$3"
    local run_dir="experiments/260914_dtcwt_${tag}_lr5e-5_single_cuda${gpu}"
    mkdir -p "$run_dir"

    CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$PYTHON" -u \
        scripts/train_and_val_posterior_gamma.py \
        --gpu "$gpu" \
        --datasets_path "$DATA/${freq}Hz/${freq}Hz_Noisy" \
        --gt "$DATA/${freq}Hz/${freq}Hz_GT/clean_${freq}Hz_1000frames.tif" \
        --pth_dir "$run_dir" \
        --patch_xy 128 \
        --patch_t 128 \
        --overlap_factor 0.75 \
        --val_overlap_factor 0.5 \
        --n_epochs 100 \
        --lr 5e-5 \
        --train_datasets_size 6000 \
        --sampling_mode temporal_mask \
        --mask_ratio 0.05 \
        --mask_min_dist 2 \
        --backbone srdtrans_v2 \
        --representation dtcwt \
        --dtcwt_dim 2 \
        --dtcwt_levels 3 \
        --embedding-dim 128 \
        --num-heads 8 \
        --hidden-dim 512 \
        --window-size 7 \
        --num-trans-block 1 \
        --attn-dropout-rate 0.1 \
        --input-dropout-rate 0 \
        --srdtrans-f-maps 8,16,32,64 \
        --trans_order st \
        --mask_loss nll \
        --kappa_mode fixed \
        --mpgn_alpha 5000 \
        --mpgn_beta 1600 \
        --mpgn_kappa 100 \
        --mpgn_offset 0 \
        --mpgn_kmax 512 \
        --mpgn_nll_chunk_t 8 \
        --mpgn_k_tail_tol 1e-8 \
        --num_workers 4 \
        --val_process_frames 400 \
        --checkpoint-every-epochs 5 \
        --validation-every-epochs 1 \
        --no_resume \
        2>&1 | tee "$run_dir/train.log" &
}

run_one 0 0.1 0p1
run_one 1 1 1
run_one 2 10 10
run_one 3 30 30

wait
