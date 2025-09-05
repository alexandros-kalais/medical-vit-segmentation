#!/usr/bin/env bash
DATASET="hyperkvasir"   # or endoscopy
MODEL="unet"
IMAGE_H=256
IMAGE_W=256
CKPT="experiments/$DATASET/unet-baseline/checkpoints/best_model-epoch=0000-val_loss=0.6708.pth"

export PYTHONPATH=src
python -m medsegformers.evaluate \
  --dataset "$DATASET" \
  --model "$MODEL" \
  --image-size "$IMAGE_H" "$IMAGE_W" \
  --batch-size 4 \
  --checkpoint "$CKPT"
