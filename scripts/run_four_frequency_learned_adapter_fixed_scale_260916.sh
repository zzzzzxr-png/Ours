#!/usr/bin/env bash
set -euo pipefail

cd /data/zhouxirou/Ours_260911

data=/data/zhouxirou/All_Datasets/dataset_CI_syn_Q004_a5000_b1600/dataset_zxr_all_frequency_Q004_beta1600-1-T1000H245W245
root=experiments/260916_dtcwt_learned_adapter_fixed_scale_pure_l1l2_width32_lr5e-5_beta09_no_checkpoint
python=/home/zxr/.conda/envs/physics/bin/python

run_one() {
    local gpu="$1" freq="$2" mask="$3" tag="$4"
    local run_dir="$root/$tag"
    mkdir -p "$run_dir"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$python" -u \
        scripts/train_and_val_posterior_gamma.py \
        --pure-network --gpu 0 \
        --datasets_path "$data/${freq}Hz/${freq}Hz_Noisy" \
        --gt "$data/${freq}Hz/${freq}Hz_GT/clean_${freq}Hz_1000frames.tif" \
        --pth_dir "$run_dir" \
        --patch_xy 128 --patch_t 128 \
        --overlap_factor 0.75 --val_overlap_factor 0.5 \
        --n_epochs 50 --lr 5e-5 --b1 0.9 --b2 0.999 \
        --train_datasets_size 6000 \
        --sampling_mode "$mask" --mask_ratio 0.05 --mask_min_dist 2 \
        --backbone srdtrans_v2 --representation dtcwt \
        --dtcwt_dim 2 --dtcwt_levels 3 --dtcwt-channel-normalize \
        --embedding-dim 128 --num-heads 8 --hidden-dim 512 \
        --window-size 7 --num-trans-block 1 \
        --attn-dropout-rate 0.1 --input-dropout-rate 0 \
        --srdtrans-f-maps 32,64,128,256 --trans_order st \
        --no-gradient-checkpointing \
        --mask_loss l1l2 --num_workers 4 --val_process_frames 400 \
        --seed 1024 --no_resume \
        2>&1 | tee "$run_dir/train.log" &
    echo "launched ${freq}Hz mask=${mask} GPU=${gpu} pid=$!"
}

run_one 0 0.1 spatial_mask 0p1Hz
run_one 1 1 temporal_mask 1Hz
run_one 2 10 temporal_mask 10Hz
run_one 3 30 temporal_mask 30Hz
wait
