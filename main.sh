#!/bin/bash

# Make sure wandb is available
if ! command -v wandb >/dev/null 2>&1; then
  echo "[INFO] 'wandb' not found in container, installing now..."
  pip3 install --no-cache-dir wandb >/dev/null
fi

# Make sure wandb is available
if ! command -v fvcore >/dev/null 2>&1; then
  echo "[INFO] 'fvcore' not found in container, installing now..."
  pip3 install --no-cache-dir fvcore >/dev/null
fi

export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"

echo "[INFO] Inside container"
echo "[INFO] MEDSEG_DATA_ROOT=$MEDSEG_DATA_ROOT"
echo "[INFO] MEDSEG_EXPERIMENTS_ROOT=$MEDSEG_EXPERIMENTS_ROOT"
echo "[INFO] PYTHONPATH=$PYTHONPATH"

python3 -m medsegformers.cli.train configs/config.yml

# python3 -m medsegformers.cli.evaluate \
#     --dataset endoscopy \
#     --experiment_id 5pct_linear_dinov3_base_448x448_lr0.0001_bs4_5folds \
#     --batch_size 4 \
#     --num_workers 4

