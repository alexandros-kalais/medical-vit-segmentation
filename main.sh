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

python3 -m medsegformers.cli.train_eomt configs/eomt.yml
# python3 -m medsegformers.cli.training_enc_dec configs/mla.yml
# python3 -m medsegformers.cli.eval \
# --dataset endoscopy \
# --experiments_file configs/evaluation.txt
# python3 -m medsegformers.cli.eval_eomt \
# --dataset endoscopy \
# --experiments_file configs/evaluation_eomt.txt
