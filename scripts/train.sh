#!/usr/bin/env bash

# --- edit these per run ---
DATASET="hyperkvasir"         # or endoscopy
MODEL="unet"
IMAGE_H=256
IMAGE_W=256
BATCH=4
EPOCHS=3
EXP_ID="unet-baseline"
# --------------------------

export PYTHONPATH=src
python -m medsegformers.train \
  --dataset "$DATASET" \
  --model "$MODEL" \
  --image-size "$IMAGE_H" "$IMAGE_W" \
  --batch-size "$BATCH" \
  --epochs "$EPOCHS" \
  --train-tf-kind basic \
  --val-tf-kind basic \
  --experiment-id "$EXP_ID"
