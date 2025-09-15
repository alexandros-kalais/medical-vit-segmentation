$env:PYTHONPATH = "src"

# --- config ---
$dataset = "endoscopy"
$imageH  = 224
$imageW  = 224
$batch   = 4
$epochs  = 50
$expId   = "vit_dinov2_linear"
$subset = 8
$lr = 1e-4
# -------------

python -m medsegformers.cli.train `
  --dataset $dataset `
  --image-size $imageH $imageW `
  --batch-size $batch `
  --epochs $epochs `
  --train-tf-kind aug `
  --val-tf-kind basic `
  --experiment-id $expId `
  --lr $lr


