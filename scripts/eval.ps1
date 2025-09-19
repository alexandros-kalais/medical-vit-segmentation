$env:PYTHONPATH = "src"

$dataset = "endoscopy"
$imageH  = 224
$imageW  = 224
$decoder = "naive"
$encoder = "vit_base_patch14_dinov2"
$ckpt    = ".\experiments\endoscopy\vit_dinov2_linear\checkpoints\best_model-epoch=0021-val_loss=0.9841.pth"

python -m medsegformers.cli.eval `
  --dataset $dataset `
  --image-size $imageH $imageW `
  --batch-size 8 `
  --checkpoint "$ckpt" `
  --decoder $decoder `
  --vit-name $encoder `


#"vit_base_patch14_dinov2"
