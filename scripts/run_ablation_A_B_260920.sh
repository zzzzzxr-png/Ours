#!/usr/bin/env bash
set -euo pipefail

cd /data/zhouxirou/Ours_260911

data=/data/zhouxirou/All_Datasets/dataset_CI_syn_Q004_a5000_b1600/dataset_zxr_all_frequency_Q004_beta1600-1-T1000H245W245
python=/home/zxr/.conda/envs/physics/bin/python

root_a=experiments/260920_dtcwt_convadd_interleaved_width128_pure_l1l2_lr5e-5
root_b=experiments/260920_dtcwt_width32_48_64_96_pure_l1l2_lr5e-5

[[ ! -e "$root_a" ]] || { echo "refusing to overwrite $root_a" >&2; exit 1; }
[[ ! -e "$root_b" ]] || { echo "refusing to overwrite $root_b" >&2; exit 1; }

run_one() {
    local gpu="$1" root="$2" freq="$3" extra="$4"
    local run_dir="$root/${freq}Hz"
    mkdir -p "$run_dir"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$python" -u \
        scripts/train_and_val_posterior_gamma.py \
        --pure-network --gpu 0 \
        --datasets_path "$data/${freq}Hz/${freq}Hz_Noisy" \
        --gt "$data/${freq}Hz/${freq}Hz_GT/clean_${freq}Hz_1000frames.tif" \
        --pth_dir "$run_dir" \
        --patch_xy 128 --patch_t 128 \
        --overlap_factor 0.75 --val_overlap_factor 0.5 \
        --n_epochs 100 --lr 5e-5 --b1 0.9 --b2 0.999 \
        --batch-size 1 \
        --train_datasets_size 6000 \
        --sampling_mode "$([[ "$freq" == "0.1" ]] && echo spatial_mask || echo temporal_mask)" \
        --mask_ratio 0.05 --mask_min_dist 2 \
        --random-patch-coordinates \
        --backbone srdtrans_v2 --representation dtcwt \
        --dtcwt_dim 2 --dtcwt_levels 3 --dtcwt-channel-normalize \
        --embedding-dim 128 --num-heads 8 --hidden-dim 512 \
        --window-size 7 --num-trans-block 1 \
        --attn-dropout-rate 0.1 --input-dropout-rate 0 \
        --trans_order st --no-gradient-checkpointing \
        --mask_loss l1l2 --num_workers 4 --val_process_frames 400 \
        --seed 1024 $extra \
        2>&1 | tee -a "$run_dir/train.log" &
    echo "launched root=$root freq=${freq}Hz gpu=$gpu pid=$!"
}

# Experiment A: conv_add + interleaved S-T-S-T transformer.
run_one 0 "$root_a" 0.1 '--srdtrans-f-maps 32,64,96,128 --skip-fusion conv_add --interleaved-transformer'
run_one 1 "$root_a" 30 '--srdtrans-f-maps 32,64,96,128 --skip-fusion conv_add --interleaved-transformer'

# Experiment B: narrower backbone, original add fusion and grouped transformer.
run_one 2 "$root_b" 0.1 '--srdtrans-f-maps 32,48,64,96 --skip-fusion add'
run_one 3 "$root_b" 30 '--srdtrans-f-maps 32,48,64,96 --skip-fusion add'

wait
