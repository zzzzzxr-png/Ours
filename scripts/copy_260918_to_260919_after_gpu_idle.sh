#!/usr/bin/env bash
set -euo pipefail

cd /data/zhouxirou/Ours_260911

old_root=experiments/260918_dtcwt_learned_adapter_rmsnorm_skipscale_random_coords_width128_pure_l1l2_lr5e-5
new_root=experiments/260919_dtcwt_learned_adapter_rmsnorm_skipscale_random_coords_width128_pure_l1l2_lr5e-5
data=/data/zhouxirou/All_Datasets/dataset_CI_syn_Q004_a5000_b1600/dataset_zxr_all_frequency_Q004_beta1600-1-T1000H245W245
python=/home/zxr/.conda/envs/physics/bin/python

gpu_free() {
    local rows row mem util gpu
    rows="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)"
    for gpu in 0 1 2 3; do
        row="$(printf '%s\n' "$rows" | awk -F, -v wanted="$gpu" '$1+0 == wanted {gsub(/[[:space:]]/, "", $2); gsub(/[[:space:]]/, "", $3); print $2, $3}')"
        if [[ -z "$row" ]]; then
            return 1
        fi
        read -r mem util <<< "$row" || return 1
        if [[ ! "$mem" =~ ^[0-9]+$ || ! "$util" =~ ^[0-9]+$ || "$mem" -gt 100 || "$util" -gt 5 ]]; then
            return 1
        fi
    done
}

checkpoints_ready() {
    [[ -f "$old_root/0p1Hz/0.1Hz_Noisy_srdtrans_spatial_mask_srdtrans_v2/E_50_Iter_6992.pth" ]] &&
    [[ -f "$old_root/1Hz/1Hz_Noisy_srdtrans_temporal_mask_srdtrans_v2/E_50_Iter_6992.pth" ]] &&
    [[ -f "$old_root/10Hz/10Hz_Noisy_srdtrans_temporal_mask_srdtrans_v2/E_50_Iter_6992.pth" ]] &&
    [[ -f "$old_root/30Hz/30Hz_Noisy_srdtrans_temporal_mask_srdtrans_v2/E_50_Iter_6992.pth" ]]
}

[[ -d "$old_root" ]] || { echo "missing old experiment: $old_root" >&2; exit 1; }
[[ ! -e "$new_root" ]] || { echo "refusing to overwrite existing directory: $new_root" >&2; exit 1; }

echo "Waiting for GPUs 0-3 to become idle..."
while ! gpu_free; do
    sleep 30
done
checkpoints_ready || {
    echo "GPUs are idle, but one or more E50 checkpoints are missing; refusing to copy." >&2
    exit 1
}

echo "Copying $old_root -> $new_root"
cp -a "$old_root" "$new_root"

run_one() {
    local gpu="$1" freq="$2" mask="$3" tag="$4"
    local run_dir="$new_root/$tag"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$python" -u \
        scripts/train_and_val_posterior_gamma.py \
        --pure-network --gpu 0 \
        --datasets_path "$data/${freq}Hz/${freq}Hz_Noisy" \
        --gt "$data/${freq}Hz/${freq}Hz_GT/clean_${freq}Hz_1000frames.tif" \
        --pth_dir "$run_dir" \
        --patch_xy 128 --patch_t 128 \
        --overlap_factor 0.75 --val_overlap_factor 0.5 \
        --n_epochs 100 --lr 5e-5 --b1 0.9 --b2 0.999 \
        --train_datasets_size 6000 \
        --sampling_mode "$mask" --mask_ratio 0.05 --mask_min_dist 2 \
        --random-patch-coordinates \
        --backbone srdtrans_v2 --representation dtcwt \
        --dtcwt_dim 2 --dtcwt_levels 3 --dtcwt-channel-normalize \
        --embedding-dim 128 --num-heads 8 --hidden-dim 512 \
        --window-size 7 --num-trans-block 1 \
        --attn-dropout-rate 0.1 --input-dropout-rate 0 \
        --srdtrans-f-maps 32,64,96,128 --trans_order st \
        --no-gradient-checkpointing \
        --mask_loss l1l2 --num_workers 4 --val_process_frames 400 \
        --seed 1024 \
        2>&1 | tee -a "$run_dir/train.log" &
    echo "launched ${freq}Hz in copied directory, pid=$!"
}

run_one 0 0.1 spatial_mask 0p1Hz
run_one 1 1 temporal_mask 1Hz
run_one 2 10 temporal_mask 10Hz
run_one 3 30 temporal_mask 30Hz
wait
