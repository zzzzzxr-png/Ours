#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 GPU_LIST TRAIN_ARGS..." >&2
  echo "example: $0 4,5 --datasets_path ... --gt ..." >&2
  exit 2
fi

GPUS=$1
shift
IFS=',' read -r -a GPU_IDS <<< "$GPUS"
if (( ${#GPU_IDS[@]} < 2 )); then
  echo "GPU_LIST must contain at least two comma-separated IDs" >&2
  exit 2
fi

ROOT=/data/zhouxirou/Ours_260911
CUDA_VISIBLE_DEVICES="$GPUS" /home/zxr/.conda/envs/physics/bin/torchrun \
  --standalone \
  --nproc-per-node="${#GPU_IDS[@]}" \
  "$ROOT/scripts/train_and_val_posterior_gamma.py" \
  --gpu "$GPUS" \
  "$@"
