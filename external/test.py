#!/usr/bin/env python3
import os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ---- repo path ----
REPO_ROOT = "/home/akalais/medseg/repo/medical-vit-segmentation"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ---- imports from your repo ----
from external.mask2former.dinov3_adapter import DINOv3_Adapter
from external.mask2former.pixel_decoder import MSDeformAttnPixelDecoder as PixelDecoder
from external.mask2former.mask2former_transformer_decoder import MultiScaleMaskedTransformerDecoder
from medsegformers.models.vit import ViT
from medsegformers.data import get_dataset_class
from medsegformers.transforms import get_transforms
from medsegformers.utils.paths import get_data_root
from medsegformers.losses.mask_classification_loss import MaskClassificationLoss

# ------------------------- config -------------------------
BATCH       = 2
IMG_SIZE    = (224, 224)
NQ          = 100
INTER       = [1, 4, 7, 11]
VIT_NAME    = "facebook/dinov3-vits16-pretrain-lvd1689m"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS      = 1         # tiny train
MAX_STEPS   = 30        # total steps (across epochs)
LR          = 1e-4
WEIGHT_DECAY= 0.05
CLIP_NORM   = 1.0
PRINT_EVERY = 5
SEED        = 42
NUM_WORKERS = 2         # safe now (collate stays on CPU)
PIN_MEMORY  = True

# ------------------------- helpers -------------------------
def _to_tensor(x):
    """MONAI MetaTensor -> torch.Tensor; tensor/ndarray passthrough."""
    return x.as_tensor() if hasattr(x, "as_tensor") else torch.as_tensor(x)

def endoscopy_collate_cpu(batch):
    """
    CPU-only collate: DO NOT touch CUDA here (avoid CUDA init in worker).
    Returns: images (CPU float tensor), targets (list of CPU tensors).
    """
    images, targets = zip(*batch)
    images = torch.stack([_to_tensor(img) for img in images], dim=0).float()  # stay on CPU
    clean_targets = []
    for t in targets:
        labels   = _to_tensor(t["labels"]).long()   # CPU
        masks    = _to_tensor(t["masks"]).float()   # CPU
        is_crowd = _to_tensor(t["is_crowd"]).bool() # CPU
        clean_targets.append({"labels": labels, "masks": masks, "is_crowd": is_crowd})
    return images, clean_targets

def move_targets_to_device(targets, device, non_blocking=True):
    out = []
    for t in targets:
        out.append({
            "labels":   t["labels"].to(device, non_blocking=non_blocking),
            "masks":    t["masks"].to(device, non_blocking=non_blocking),
            "is_crowd": t["is_crowd"].to(device, non_blocking=non_blocking),
        })
    return out

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

# ------------------------- main -------------------------
def main():
    set_seed(SEED)
    torch.set_printoptions(sci_mode=False)

    # ----- backbone + adapter (frozen) -----
    print("[INFO] building ViT backbone from vit.py …")
    vit = ViT(img_size=IMG_SIZE, backbone_name=VIT_NAME, patch_size=16, ckpt_path=None).to(DEVICE).eval()

    print("[INFO] building DINOv3_Adapter …")
    adapter = DINOv3_Adapter(
        backbone=vit.backbone,
        interaction_indexes=INTER,
        deform_num_heads=8,
        n_points=4,
        with_cp=False,
        add_vit_feature=True,
        deform_ratio=0.5,
        drop_path_rate=0.0,
    ).to(DEVICE).eval()

    # freeze encoder + adapter
    for p in vit.backbone.parameters():
        p.requires_grad = False
    for p in adapter.parameters():
        p.requires_grad = False

    # ----- dataset / loader -----
    DatasetCls = get_dataset_class("endoscopy")
    root = DatasetCls.default_root(get_data_root())
    tf = get_transforms(dataset="endoscopy", kind="basic", image_size=IMG_SIZE)

    ds_train = DatasetCls(split="train", transform=tf, root=root, seed=SEED, return_masks=True)
    loader = DataLoader(
        ds_train,
        batch_size=BATCH,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=endoscopy_collate_cpu,
        drop_last=True,
    )

    # ----- probe one batch (move to device here) -----
    images_cpu, targets_cpu = next(iter(loader))
    images = images_cpu.to(DEVICE, non_blocking=True)
    targets = move_targets_to_device(targets_cpu, DEVICE)

    with torch.no_grad():
        feats = adapter(images)

    def C(x): return x.shape[1]
    input_shape = {
        "1": (C(feats["1"]), feats["1"].shape[-2], feats["1"].shape[-1], 4),
        "2": (C(feats["2"]), feats["2"].shape[-2], feats["2"].shape[-1], 8),
        "3": (C(feats["3"]), feats["3"].shape[-2], feats["3"].shape[-1], 16),
        "4": (C(feats["4"]), feats["4"].shape[-2], feats["4"].shape[-1], 32),
    }
    EMB = C(feats["1"])

    # ----- pixel decoder + transformer decoder (trainable) -----
    print("[INFO] building pixel decoder …")
    pixel_decoder = PixelDecoder(
        input_shape=input_shape,
        transformer_in_features=["1","2","3"],
        transformer_dropout=0.0,
        transformer_nheads=8,
        transformer_dim_feedforward=1024,
        transformer_enc_layers=3,
        conv_dim=EMB,
        mask_dim=EMB,
        common_stride=4,
    ).to(DEVICE)

    print("[INFO] building transformer decoder …")
    num_classes = DatasetCls.NUM_CLASSES
    decoder = MultiScaleMaskedTransformerDecoder(
        in_channels=EMB,
        mask_classification=True,
        num_classes=num_classes,  # +1 for no-object handled inside
        hidden_dim=EMB,
        num_queries=NQ,
        nheads=8,
        dim_feedforward=1024,
        dec_layers=3,
        pre_norm=False,
        mask_dim=EMB,
        enforce_input_project=False,
    ).to(DEVICE)

    # group trainable parts
    trainable = nn.ModuleList([pixel_decoder, decoder]).train()

    # ----- loss & optimizer -----
    criterion = MaskClassificationLoss(
        num_points=12544,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        mask_coefficient=5.0,
        dice_coefficient=5.0,
        class_coefficient=2.0,
        num_labels=num_classes,
        no_object_coefficient=0.1,
    ).to(DEVICE)

    opt = torch.optim.AdamW(
        (p for p in trainable.parameters() if p.requires_grad),
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    # ----- tiny training loop -----
    print("[INFO] starting tiny training loop …")
    step = 0
    for epoch in range(EPOCHS):
        for images_cpu, targets_cpu in loader:
            t0 = time.time()

            # move batch to device here (safe with workers>0)
            images = images_cpu.to(DEVICE, non_blocking=True)
            targets = move_targets_to_device(targets_cpu, DEVICE)

            # 1) backbone+adapter forward (frozen)
            with torch.no_grad():
                feats = adapter(images)

            # 2) pixel decoder + transformer decoder + loss
            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                mask_features, _mem_unused, multi_scale_features = pixel_decoder.forward_features(feats)
                out = decoder(multi_scale_features, mask_features, mask=None)
                pred_logits = out["pred_logits"] if isinstance(out, dict) else out[0]  # [B,Q,C+1]
                pred_masks  = out["pred_masks"]  if isinstance(out, dict) else out[1]  # [B,Q,Hm,Wm]

                loss_dict = criterion(
                    masks_queries_logits=pred_masks,
                    class_queries_logits=pred_logits,
                    targets=targets,
                )

                w_mask = getattr(criterion, "mask_coefficient", 1.0)
                w_dice = getattr(criterion, "dice_coefficient", 1.0)
                w_ce   = getattr(criterion, "class_coefficient", 1.0)
                total_loss = w_mask * loss_dict["loss_mask"] \
                           + w_dice * loss_dict["loss_dice"] \
                           + w_ce   * loss_dict["loss_cross_entropy"]

            # 3) optimize
            opt.zero_grad(set_to_none=True)
            scaler.scale(total_loss).backward()
            torch.nn.utils.clip_grad_norm_(trainable.parameters(), CLIP_NORM)
            scaler.step(opt)
            scaler.update()

            step += 1
            if (step % PRINT_EVERY) == 0:
                lm = float(loss_dict["loss_mask"])
                ld = float(loss_dict["loss_dice"])
                lce= float(loss_dict["loss_cross_entropy"])
                print(f"[E{epoch:02d} S{step:04d}] "
                      f"mask={lm:.4f} dice={ld:.4f} ce={lce:.4f} | total={float(total_loss):.4f} "
                      f"| dt={(time.time()-t0)*1000:.1f}ms")

            if step >= MAX_STEPS:
                break
        if step >= MAX_STEPS:
            break

    print("✅ tiny training loop finished OK.")

if __name__ == "__main__":
    main()
