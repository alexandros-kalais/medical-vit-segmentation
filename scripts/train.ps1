$env:PYTHONPATH = "src"

# --- config ---
$dataset = "endoscopy"
$imageH  = 224
$imageW  = 224
$batch   = 8
$epochs  = 35
$expId   = "dinov2_base_mla_224"
$subset = 8
$lr = 1e-4
$decoder = "mla"
$encoder = "vit_base_patch14_dinov2.lvd142m"
# -------------

python -m medsegformers.cli.train `
  --dataset $dataset `
  --image-size $imageH $imageW `
  --batch-size $batch `
  --epochs $epochs `
  --train-tf-kind aug `
  --val-tf-kind basic `
  --experiment-id $expId `
  --lr $lr `
  --decoder $decoder `
  --vit-name $encoder `


