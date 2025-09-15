$env:PYTHONPATH = "src"

# --- config ---
$dataset = "endoscopy"
$model   = "vit_linear"
$imageH  = 224
$imageW  = 224
$batch   = 4
$epochs  = 50
$expId   = "vit_linear-"
$subset = 8
$lr = 1e-4
# -------------

python -m medsegformers.train `
  --dataset $dataset `
  --model $model `
  --image-size $imageH $imageW `
  --batch-size $batch `
  --epochs $epochs `
  --train-tf-kind aug `
  --val-tf-kind basic `
  --experiment-id $expId `
  --lr $lr `


