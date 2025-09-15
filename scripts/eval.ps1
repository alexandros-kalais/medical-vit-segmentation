$env:PYTHONPATH = "src"

$dataset = "endoscopy"
$model   = "vit_linear"
$imageH  = 224
$imageW  = 224
$ckpt    = ".\experiments\endoscopy\vit_linear-\checkpoints\final_model-epoch=0049-val_loss=1.0741.pth"

python -m medsegformers.cli.eval `
  --dataset $dataset `
  --model $model `
  --image-size $imageH $imageW `
  --batch-size 4 `
  --checkpoint "$ckpt"
