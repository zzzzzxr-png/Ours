#!/usr/bin/env bash
set -euo pipefail

cd /data/zhouxirou/Ours_260911

data=/data/zhouxirou/All_Datasets/dataset_CI_syn_Q004_a5000_b1600/dataset_zxr_all_frequency_Q004_beta1600-1-T1000H245W245
root=experiments/260915_dtcwt_complex_instability_diagnostic_lr5e-5
python=/home/zxr/.conda/envs/physics/bin/python

run_one() {
    local gpu="$1" tag="$2" normalize="$3"
    local run_dir="$root/$tag"
    local normalize_arg=()
    if [[ "$normalize" == true ]]; then
        normalize_arg+=(--dtcwt-channel-normalize)
    fi
    mkdir -p "$run_dir"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$python" -u \
        scripts/train_and_val_posterior_gamma.py \
        --pure-network --gpu 0 \
        --datasets_path "$data/0.1Hz/0.1Hz_Noisy" \
        --gt "$data/0.1Hz/0.1Hz_GT/clean_0.1Hz_1000frames.tif" \
        --pth_dir "$run_dir" \
        --patch_xy 128 --patch_t 128 \
        --overlap_factor 0.75 --val_overlap_factor 0.5 \
        --n_epochs 1 --lr 5e-5 --b1 0.5 --b2 0.999 \
        --train_datasets_size 4000 \
        --sampling_mode spatial_mask --mask_ratio 0.05 --mask_min_dist 2 \
        --backbone srdtrans_v2 --representation dtcwt \
        --dtcwt_dim 2 --dtcwt_levels 3 \
        --embedding-dim 128 --num-heads 8 --hidden-dim 512 \
        --window-size 7 --num-trans-block 1 \
        --attn-dropout-rate 0.1 --input-dropout-rate 0 \
        --srdtrans-f-maps 32,64,128,256 --trans_order st \
        --mask_loss l1l2 --num_workers 4 --val_process_frames 400 \
        --diagnostic-interval 100 --seed 1024 --no_resume \
        "${normalize_arg[@]}" \
        2>&1 | tee "$run_dir/train.log" &
    echo "launched 0.1Hz normalize=${normalize} GPU=${gpu} pid=$!"
}

run_one 0 no_channel_norm false
run_one 3 channel_norm true
wait
