#!/usr/bin/env bash
set -euo pipefail

cd /data/zhouxirou/Ours_260911
data=/data/zhouxirou/All_Datasets/dataset_CI_syn_Q004_a5000_b1600/dataset_zxr_all_frequency_Q004_beta1600-1-T1000H245W245
python=/home/zxr/.conda/envs/physics/bin/python
root="${EXPERIMENT_ROOT:-experiments/260925_steerable_fourier_add_interleaved_width24_36_48_64_head8_hidden384_patch64_pure_l1l2_lr1e-4_v2}"

if [[ -e "$root" ]]; then
    echo "refusing to overwrite existing experiment: $root" >&2
    exit 1
fi

run_one() {
    local gpu="$1" freq="$2"
    local run_dir="$root/${freq}Hz"
    mkdir -p "$run_dir"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$python" -u \
        scripts/train_and_val_posterior_gamma.py \
        --pure-network --gpu 0 \
        --datasets_path "$data/${freq}Hz/${freq}Hz_Noisy" \
        --gt "$data/${freq}Hz/${freq}Hz_GT/clean_${freq}Hz_1000frames.tif" \
        --pth_dir "$run_dir" \
        --patch_xy 64 --patch_t 128 \
        --overlap_factor 0.75 --val_overlap_factor 0.5 \
        --n_epochs 100 --lr 1e-4 --b1 0.9 --b2 0.999 \
        --batch-size 1 --train_datasets_size 6000 \
        --sampling_mode "$([[ "$freq" == "0.1" ]] && echo spatial_mask || echo temporal_mask)" \
        --mask_ratio 0.05 --mask_min_dist 2 --random-patch-coordinates \
        --backbone srdtrans_v2 --representation steerable_fourier \
        --dtcwt_dim 2 --dtcwt_levels 3 --legacy-fourier-adapter \
        --embedding-dim 128 --num-heads 8 --hidden-dim 384 \
        --srdtrans-f-maps 24,36,48,64 --window-size 7 --num-trans-block 1 \
        --attn-dropout-rate 0.1 --input-dropout-rate 0 --trans_order st \
        --mask_loss l1l2 --num_workers 4 --val_process_frames 400 \
        --skip-fusion add --interleaved-transformer \
        --space-attention "${SPACE_ATTENTION:-swin}" \
        --no-gradient-checkpointing --seed 1024 \
        2>&1 | tee -a "$run_dir/train.log" &
    echo "launched freq=${freq}Hz cuda=${gpu} pid=$!"
}

run_one "${GPU_A:-0}" 0.1
run_one "${GPU_B:-1}" 30
wait
