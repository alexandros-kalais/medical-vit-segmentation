$env:PYTHONPATH = "src"

$dataset = "endoscopy"
$model   = "unet"
$imageH  = 256
$imageW  = 256
$ckpt    = ".\experiments\endoscopy\unet-baseline\checkpoints\best_model-epoch=0002-val_loss=0.8095954690856495.pth"

python -m medsegformers.evaluate `
  --dataset $dataset `
  --model $model `
  --image-size $imageH $imageW `
  --batch-size 4 `
  --checkpoint "$ckpt"
