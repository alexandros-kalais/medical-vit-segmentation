import os
import sys
import json
from datetime import datetime
from typing import Tuple, List

import lightning as L
import torch
import torch.nn as nn
from monai.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from medsegformers.utils import load_config
from medsegformers.dataset import get_dataset_class, get_transforms
from medsegformers.models import ViT
from medsegformers.utils.paths import get_data_root, ckpt_dir
from . import *

def main():
    # 1. Load config
    cfg_path = None
    for arg in sys.argv:
        if arg.endswith(".yml") or arg.endswith(".yaml"):
            cfg_path = arg
            break
    if cfg_path is None:
        raise ValueError("Please provide a config YAML path")
    
    args = load_config(cfg_path)
    
    if not hasattr(args, "model_type"):
        raise ValueError("Config must specify 'model_type' (one of: enc_dec, mask2former, eomt)")
    
    if args.model_type not in ["enc_dec", "mask2former", "eomt"]:
        raise ValueError(f"Invalid model_type: {args.model_type}. Must be one of: enc_dec, mask2former, eomt")
    
    print(f"\n{'='*60}")
    print(f"Training {args.model_type.upper()} model")
    print(f"{'='*60}\n")
    
    # 2. Runtime setup
    L.seed_everything(args.seed, workers=True)
    use_gpu = torch.cuda.is_available()
    accelerator = "gpu" if use_gpu else "cpu"
    devices = 1
    
    if args.model_type == "mask2former" and use_gpu:
        precision = "bf16-mixed"
    elif use_gpu:
        precision = "16-mixed"
    else:
        precision = "32-true"
    
    # 3. Data
    DatasetCls = get_dataset_class(args.dataset)
    root = DatasetCls.default_root(get_data_root())
    
    tf_train = get_transforms(dataset=args.dataset, kind=args.train_tf_kind, image_size=args.image_size)
    tf_val = get_transforms(dataset=args.dataset, kind=args.val_tf_kind, image_size=args.image_size)
    
    return_masks = args.model_type in ["mask2former", "eomt"]
    
    train_ds = DatasetCls(split="train", transform=tf_train, root=root, seed=args.seed, return_masks=return_masks)
    val_ds = DatasetCls(split="validation", transform=tf_val, root=root, seed=args.seed, return_masks=return_masks)
    
    num_classes = getattr(DatasetCls, "NUM_CLASSES", None)
    if num_classes is None:
        raise ValueError("Dataset class must define NUM_CLASSES")
    
    if args.subset > 0:
        train_ds = torch.utils.data.Subset(train_ds, list(range(args.subset)))
        val_ds = torch.utils.data.Subset(val_ds, list(range(args.subset)))
    
    train_collate_fn, val_collate_fn = select_collate(args.model_type)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=use_gpu,
        collate_fn=train_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_gpu,
        collate_fn=val_collate_fn,
    )
    
    # 4. Build encoder (ViT)
    patch_size = infer_patch_size(args.vit_name)
    
    encoder = ViT(
        img_size=args.image_size,
        patch_size=patch_size,
        backbone_name=args.vit_name,
        ckpt_path=getattr(args, "vit_ckpt", None),
    )
    
    freeze_encoder_layers(encoder, args.freeze_encoder, args.unfreeze_last_k)
    
    # 5. Build model based on type

    model = build_model(model_type = args.model_type, vit=encoder, num_classes=num_classes, config=args, mode="train")
    
    # 6. Compute training schedule
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    
    non_vit_warmup = steps_per_epoch * args.non_vit_warmup
    vit_warmup = steps_per_epoch * args.vit_warmup

    
    warmup_steps = (non_vit_warmup, vit_warmup)
    
    print(f"[INFO] steps_per_epoch={steps_per_epoch}, total_steps={total_steps}")
    print(f"[INFO] warmup_steps (non_vit, vit) = {warmup_steps}")
    
    # 7. Build Lightning module
    module = build_module(
        args=args,
        model=model,
        num_classes=num_classes,
        warmup_steps=warmup_steps,
        steps_per_epoch=steps_per_epoch,
        eomt_num_blocks=getattr(args, "eomt_num_blocks", None)
        )
    
    # 8. Logging and checkpoints
    if not hasattr(args, "experiment_id") or args.experiment_id in [None, "", "auto"]:
        args.experiment_id = generate_experiment_id(args)
        print(f"[INFO] Auto experiment_id set to: {args.experiment_id}")
    
    wandb_project = getattr(args, "wandb_project", "Internship-medical-vit-segmentation")
    wandb_logger = WandbLogger(
        project=wandb_project,
        name=args.experiment_id,
        log_model=False,
    )
    wandb_logger.experiment.config.update(vars(args))
    
    run_dir = ckpt_dir(args.dataset, args.experiment_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    
    ckpt_callback = ModelCheckpoint(
        dirpath=str(run_dir),
        filename="best-{val_miou:.4f}",
        monitor="metrics/val_iou_all",
        mode="max",
        save_top_k=1,
        save_last=False,
    )
    

    run_config = vars(args).copy()
    run_config.update({
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "warmup_steps": {
            "non_vit_warmup": non_vit_warmup,
            "vit_warmup": vit_warmup,
        },
    })
    
    if args.model_type == "eomt":
        run_config.update({
            "anneal_starts": module.attn_mask_annealing_start_steps,
            "anneal_ends": module.attn_mask_annealing_end_steps,
        })
    
    with open(run_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=4)
    
    # 9. Train
    trainer = L.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=args.epochs,
        precision=precision,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        detect_anomaly=False,
        logger=wandb_logger,
        callbacks=[ckpt_callback],
    )
    
    print(f"\n[INFO] Starting training for {args.epochs} epochs...")
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    print(f"\n[INFO] Training complete! Checkpoints saved to: {run_dir}")
    
    try:
        import wandb
        wandb.finish()
    except Exception:
        pass


if __name__ == "__main__":
    main()