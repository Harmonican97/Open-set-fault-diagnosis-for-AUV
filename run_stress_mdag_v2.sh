#!/usr/bin/env bash
set -euo pipefail

# Stress test runner for ProtoNetMDAG (v2 bugfix)
# Usage:
#   bash run_stress_mdag_v2.sh CKPT_PATH [DATASET_DIR]

CKPT="${1:?Please provide checkpoint path (best_model.pth)}"
DATA_DIR="${2:-./dataset}"

python stress_mdag_openset_v2.py \
  --data_dir "${DATA_DIR}" \
  --ckpt "${CKPT}" \
  --out_dir "./stress_mdag_results" \
  --seq_len 50 \
  --stride 20 \
  --batch_size 64 \
  --win_size 11 \
  --md_quantile 0.95 \
  --unknown_files "1_ad.csv,4_pds.csv" \
  --stress_channels "5,12" \
  --repeats 10 \
  --plots
