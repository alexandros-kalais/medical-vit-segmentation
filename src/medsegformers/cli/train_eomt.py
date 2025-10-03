import math
import os
from typing import Iterable, List, Tuple
import torch.nn as nn 
import lightning as L
import torch
from monai.data import DataLoader
from torch.utils.data import Subset
from collections import Counter
from medsegformers.config.args import get_train_args_parser
from medsegformers.data import get_dataset_class
from medsegformers.transforms import get_transforms
from medsegformers.utils.paths import get_data_root

from medsegformers.models.eomt import EoMT
from medsegformers.models import ViT
from medsegformers.engines.eomt.mask_classification_semantic import (
    MaskClassificationSemantic,
)

from lightning.pytorch.loggers import WandbLogger

def scan_class_coverage(dataset, num_classes: int, max_items: int | None = None):
    """
    Iterates the dataset (which yields (image, target) where target['labels'] are class ids)
    and returns:
      - present_class_ids: sorted list of class ids that appear at least once
      - per_class_image_count: dict[class_id] -> number of images where the class appears
      - per_class_pixel_count: dict[class_id] -> total pixel count across the split
    """
    per_class_image_count = Counter()
    per_class_pixel_count = Counter()
    present = set()

    length = len(dataset) if max_items is None else min(max_items, len(dataset))
    for i in range(length):
        _, tgt = dataset[i]  # tgt: {"masks": [N,H,W] bool or 0-sized, "labels": [N], "is_crowd": [N]}
        labels = tgt["labels"].cpu().numpy().tolist()
        if len(labels) == 0:
            continue

        # Image-level presence
        seen_in_image = set(labels)
        for c in seen_in_image:
            per_class_image_count[c] += 1
            present.add(c)

        # Pixel-level presence (optional but useful)
        masks = tgt["masks"]   # shape [N,H,W] (bool) or empty
        if masks.numel() > 0:
            for k, c in enumerate(labels):
                per_class_pixel_count[c] += int(masks[k].sum().item())

    present_class_ids = sorted(list(present))
    # Ensure all classes are represented in the dicts (with 0 if missing)
    for c in range(num_classes):
        per_class_image_count.setdefault(c, 0)
        per_class_pixel_count.setdefault(c, 0)

    return present_class_ids, dict(per_class_image_count), dict(per_class_pixel_count)

def eomt_train_collate(batch):
    images, targets = [], []
    for image, target in batch:
        images.append(image)
        targets.append(target)
    images = torch.stack(images)
    return images, targets


def eomt_eval_collate(batch):
    return tuple(zip(*batch))


def lightning_optimizer_steps_per_epoch(trainer: L.Trainer, train_loader_len: int) -> int:
    ltb = trainer.limit_train_batches
    if isinstance(ltb, int):
        effective_batches = min(train_loader_len, ltb)
    elif isinstance(ltb, float):
        effective_batches = int(math.floor(train_loader_len * ltb))
    else:
        effective_batches = train_loader_len
    accum = getattr(trainer, "accumulate_grad_batches", 1)
    return max(1, math.ceil(effective_batches / accum))

def set_default(ns, name, value):
    if not hasattr(ns, name):
        setattr(ns, name, value)


def scale_milestones(total_steps: int, fractions: List[float]) -> List[int]:
    return [max(0, int(round(total_steps * f))) for f in fractions]


def default_mask_anneal_fracs(n_blocks: int) -> Tuple[List[float], List[float]]:

    if n_blocks <= 0:
        return [], []
    starts = [i * (0.80 / max(1, n_blocks - 1)) for i in range(n_blocks)]  # [0 .. 0.8]
    ends = [min(s + 0.18, 0.60) for s in starts]
    ends[-1] = 0.60
    return starts, ends


# ======================= main =======================

def main():
    args = get_train_args_parser().parse_args()

    # # ---------- sensible defaults (only if not already in your args) ----------
    # set_default(args, "vit_ckpt", None)
    set_default(args, "eomt_num_q", 7)          # capacity for semantic queries
    set_default(args, "eomt_num_blocks", 4)
    set_default(args, "eomt_disable_masked_attn", False)
    set_default(args, "ignore_index", 255)
    set_default(args, "freeze_encoder", True)
    set_default(args, "unfreeze_last_k", 0)        # try 2/4 on small data
    # set_default(args, "num_workers", 0)
    # set_default(args, "seed", 41)

    # ---------- runtime ----------
    use_gpu = torch.cuda.is_available()
    accelerator = "gpu" if use_gpu else "cpu"
    devices = 1
    precision = "16-mixed" if use_gpu else "32-true"
    num_workers = int(getattr(args, "num_workers", 0))
    pin_memory = use_gpu
    persistent_workers = bool(num_workers > 0)

    L.seed_everything(args.seed, workers=True)

    # ------------------------- datasets & transforms -------------------------
    DatasetCls = get_dataset_class(args.dataset)
    root = DatasetCls.default_root(get_data_root())

    train_tf = get_transforms(dataset=args.dataset, kind=args.train_tf_kind, image_size=args.image_size)
    val_tf   = get_transforms(dataset=args.dataset, kind=args.val_tf_kind,  image_size=args.image_size)

    train_ds = DatasetCls(split = "train",      transform=train_tf, root=root, seed=args.seed, return_masks = True)
    val_ds   = DatasetCls(split = "validation", transform=val_tf,   root=root, seed=args.seed, return_masks = True)

    num_classes = getattr(DatasetCls, "NUM_CLASSES", None)
    if num_classes is None:
        raise ValueError("Dataset class must define NUM_CLASSES")
    
    if args.subset > 0:
        train_ds = torch.utils.data.Subset(train_ds, list(range(args.subset)))
        val_ds   = torch.utils.data.Subset(val_ds,   list(range(args.subset)))


    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=eomt_train_collate,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=eomt_eval_collate,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=False,
    )

    # ------------------------- logging + trainer -------------------------
    wandb_logger = WandbLogger(
        project=getattr(args, "wandb_project", "Internship-medical-vit-segmentation"),
        name=args.experiment_id,
        log_model=False,
    )

    trainer = L.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=args.epochs,
        precision=precision,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        detect_anomaly=False,
        logger=wandb_logger,
    )

    # ------------------------- steps & annealing (ratio-scaled) -------------------------
    steps_per_epoch = lightning_optimizer_steps_per_epoch(trainer, len(train_loader))
    total_steps = steps_per_epoch * trainer.max_epochs

    # Default fractions (fast debug): fully off by ~60% of training
    starts_frac, ends_frac = default_mask_anneal_fracs(getattr(args, "eomt_num_blocks", 4))
    # If you want the official-ish schedule instead, uncomment:
    # starts_frac = [0.00, 0.40, 0.60, 0.80]
    # ends_frac   = [0.10, 0.35, 0.50, 0.60]  # or [0.18, 0.60, 0.80, 1.00]

    anneal_starts = scale_milestones(total_steps, starts_frac)
    anneal_ends = scale_milestones(total_steps, ends_frac)
    warm_steps = scale_milestones(total_steps, [0.05, 0.10])

    print(f"steps_per_epoch={steps_per_epoch}, total_steps={total_steps}")
    print("anneal_starts:", anneal_starts)
    print("anneal_ends  :", anneal_ends)
    print("warmup_steps :", warm_steps)

    # ------------------------- ViT -> EoMT -------------------------
    H, W = args.image_size
    encoder = ViT(
        img_size=(H, W),
        patch_size=16,
        backbone_name=args.vit_name,
        ckpt_path=getattr(args, "vit_ckpt", None),
    )

    if args.freeze_encoder:
        n = len(encoder.backbone.blocks) - args.unfreeze_last_k
        for blk in encoder.backbone.blocks[:n]:
            for p in blk.parameters():
                p.requires_grad_(False)
            for mod in blk.modules():
                if isinstance(mod, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
                    for p in mod.parameters():
                        p.requires_grad_(False)

    network = EoMT(
        encoder=encoder,
        num_classes=num_classes,
        num_q=getattr(args, "eomt_num_q", 100),
        num_blocks=getattr(args, "eomt_num_blocks", 4),
        masked_attn_enabled=not bool(getattr(args, "eomt_disable_masked_attn", False)),
    )


    # ------------------------- Lightning module -------------------------
    module = MaskClassificationSemantic(
        network=network,
        img_size=(H, W),
        num_classes=num_classes,
        attn_mask_annealing_enabled=not bool(getattr(args, "eomt_disable_masked_attn", False)),
        lr=args.lr,
        llrd=getattr(args, "llrd", 1.0),
        lr_mult=getattr(args, "lr_mult", 0.1),
        warmup_steps=warm_steps,
        ignore_idx=getattr(args, "ignore_index", 255),
        attn_mask_annealing_start_steps=anneal_starts,
        attn_mask_annealing_end_steps=anneal_ends,
    )

    # ------------------------- train -------------------------
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    try:
        import wandb
        wandb.finish()
    except Exception:
        pass


if __name__ == "__main__":
    main()
