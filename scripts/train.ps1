$env:PYTHONPATH = "src"

# --- config ---
$dataset = "endoscopy"
$model   = "unet"
$imageH  = 256
$imageW  = 256
$batch   = 4
$epochs  = 3
$expId   = "unet-baseline"
# -------------

python -m medsegformers.train `
  --dataset $dataset `
  --model $model `
  --image-size $imageH $imageW `
  --batch-size $batch `
  --epochs $epochs `
  --train-tf-kind basic `
  --val-tf-kind basic `
  --experiment-id $expId
