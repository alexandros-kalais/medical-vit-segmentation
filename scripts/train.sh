#!/usr/bin/env bash

# --- edit these per run ---
DATASET="hyperkvasir"         # or endoscopy
MODEL="unet"
IMAGE_H=224
IMAGE_W=224
BATCH=4
EPOCHS=3
EXP_ID="vit-linear-8-images"
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
  --experiment-id "$EXP_ID" \
  --subset 8
