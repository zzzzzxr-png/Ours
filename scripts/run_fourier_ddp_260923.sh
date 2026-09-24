#!/usr/bin/env bash
set -euo pipefail

cd /data/zhouxirou/Ours_260911

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "usage: CUDA_VISIBLE_DEVICES=4,5 bash $0 <frequency> <output_root> [master_port]" >&2
    exit 2
fi

freq="$1"
root="$2"
port="${3:-29541}"
data=/data/zhouxirou/All_Datasets/dataset_CI_syn_Q004_a5000_b1600/dataset_zxr_all_frequency_Q004_beta1600-1-T1000H245W245
python=/home/zxr/.conda/envs/physics/bin/python

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo 'set CUDA_VISIBLE_DEVICES to exactly two GPUs, e.g. 4,5' >&2
    exit 2
fi

mkdir -p "$root/${freq}Hz"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$python" -m torch.distributed.run \
    --standalone --master_port "$port" --nproc_per_node=2 \
    scripts/train_and_val_posterior_gamma.py \
    --pure-network --gpu 0 \
    --datasets_path "$data/${freq}Hz/${freq}Hz_Noisy" \
    --gt "$data/${freq}Hz/${freq}Hz_GT/clean_${freq}Hz_1000frames.tif" \
    --pth_dir "$root/${freq}Hz" \
    --patch_xy 128 --patch_t 128 \
    --overlap_factor 0.75 --val_overlap_factor 0.5 \
    --n_epochs 100 --lr 1e-4 --b1 0.9 --b2 0.999 \
    --batch-size 1 --train_datasets_size 6000 \
    --sampling_mode "$([[ "$freq" == "0.1" ]] && echo spatial_mask || echo temporal_mask)" \
    --mask_ratio 0.05 --mask_min_dist 2 --random-patch-coordinates \
    --backbone srdtrans_v2 --representation steerable_fourier \
    --dtcwt_dim 2 --dtcwt_levels 3 \
    --embedding-dim 128 --num-heads 8 --hidden-dim 512 \
    --srdtrans-f-maps 48,64,96,128 --window-size 7 --num-trans-block 1 \
    --attn-dropout-rate 0.1 --input-dropout-rate 0 \
    --trans_order st --no-gradient-checkpointing \
    --mask_loss l1l2 --num_workers 4 --val_process_frames 400 \
    --skip-fusion conv_add --interleaved-transformer --seed 1024 \
    2>&1 | tee -a "$root/${freq}Hz/train_ddp.log"
