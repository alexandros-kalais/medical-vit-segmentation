$env:PYTHONPATH = "src"

# --- config ---
$dataset = "endoscopy"
$imageH  = 224
$imageW  = 224
$batch   = 4
$epochs  = 35
$expId   = "dinov3_small_masktfm_224"
$subset = 4
$lr = 1e-4
$decoder = "masktfm"
$encoder = "facebook/dinov3-vits16-pretrain-lvd1689m"#"vit_small_patch16_dinov3.lvd1689m" #"vit_small_patch14_dinov2.lvd142m"
# -------------

python -m medsegformers.cli.training_enc_dec `
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



