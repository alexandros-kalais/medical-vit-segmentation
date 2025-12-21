import os
import sys
import json
from datetime import datetime
from typing import Tuple, List

import numpy as np
import lightning as L
import torch
import torch.nn as nn
from monai.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger

from medsegformers.dataset import get_dataset_class, get_transforms
from medsegformers.models import ViT
from medsegformers.utils.paths import get_data_root, ckpt_dir
from . import *

def main():

    # 1.Load config file
    cfg_path = None
    for arg in sys.argv:
        if arg.endswith(".yml") or arg.endswith(".yaml"):
            cfg_path = arg
            break
    if cfg_path is None:
        raise ValueError("Please provide a config YAML path")
    
    args = load_config(cfg_path)
    
    if args.model_type not in ["enc_dec", "mask2former", "eomt"]:
        raise ValueError(f"Invalid model_type: {args.model_type}. Must be one of: enc_dec, mask2former, eomt")
    
    
    # 2. Runtime setup
    L.seed_everything(args.seed, workers=True)
    use_gpu = torch.cuda.is_available()
    accelerator = "gpu" if use_gpu else "cpu"
    device = torch.device("cuda" if use_gpu else "cpu")
    devices = 1
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    precision = "bf16-mixed"
 
    
    # 3. Cross-validation setup
    n_folds = args.n_folds
    few_shot_pct = float(getattr(args, "few_shot_pct", 0.0))
    
    # 4. Data
    DatasetCls = get_dataset_class(args.dataset)
    root = DatasetCls.default_root(get_data_root())
    
    tf_train = get_transforms(dataset=args.dataset, kind=args.train_tf_kind, image_size=args.image_size)
    tf_val = get_transforms(dataset=args.dataset, kind=args.val_tf_kind, image_size=args.image_size)
    
    return_masks = args.model_type in ["mask2former", "eomt"]
    
    num_classes = getattr(DatasetCls, "NUM_CLASSES", None)
        
    # 5. Generate experiment ID
    if not hasattr(args, "experiment_id") or args.experiment_id in [None, "", "auto"]:
        args.experiment_id = generate_experiment_id(args) 
    
    # Storage for fold results
    fold_results = []

    
    # 6. Main cross-validation loop
    for fold_idx in range(n_folds):

        print(f"Training Fold {fold_idx + 1}/{n_folds}")

        
        # Re-seed for each fold
        L.seed_everything(args.seed, workers=True)
        
        # Create datasets for this fold
        train_ds = DatasetCls(
            split="cross-val",
            transform=tf_train,
            root=root,
            seed=args.seed,
            return_masks=return_masks,
            n_folds=n_folds,
            fold_idx=fold_idx,
            train=True,
        )
        val_ds = DatasetCls(
            split="cross-val",
            transform=tf_val,
            root=root,
            seed=args.seed,
            return_masks=return_masks,
            n_folds=n_folds,
            fold_idx=fold_idx,
            train=False,
        )
        
        if few_shot_pct > 0:
            if not (0.0 < few_shot_pct <= 1.0):
                raise ValueError("few_shot_pct must be in (0, 1].")
            n_total = len(train_ds)
            n_keep = max(1, int(np.ceil(n_total * few_shot_pct)))
            rng = np.random.default_rng(args.seed)
            indices = rng.choice(n_total, size=n_keep, replace=False)
            indices = np.sort(indices)
            train_ds = torch.utils.data.Subset(train_ds, indices.tolist())


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
        
        # Build encoder (ViT)
        patch_size = infer_patch_size(args.vit_name)
        
        encoder = ViT(
            img_size=args.image_size,
            patch_size=patch_size,
            backbone_name=args.vit_name,
            ckpt_path=getattr(args, "vit_ckpt", None),
        ).to(device)
        
        freeze_encoder_layers(encoder, args.freeze_encoder, args.unfreeze_last_k)
        
        # Build model based on type
        model = build_model(
            model_type=args.model_type,
            vit=encoder,
            num_classes=num_classes,
            config=args,
            mode="train"
        )
        
        # Compute training schedule
        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * args.epochs
        non_vit_warmup = steps_per_epoch * args.non_vit_warmup
        vit_warmup = steps_per_epoch * args.vit_warmup
        warmup_steps = (non_vit_warmup, vit_warmup)
        
        
        # Build Lightning module
        module = build_module(
            args=args,
            model=model,
            num_classes=num_classes,
            warmup_steps=warmup_steps,
            steps_per_epoch=steps_per_epoch,
            eomt_num_blocks=getattr(args, "eomt_num_blocks", None)
        )
        
        # Logging and checkpoints for this fold
        wandb_project = getattr(args, "wandb_project", "medical-vit-segmentation")
        experiment_name = f"{args.experiment_id}_fold_{fold_idx}"
        
        wandb_logger = WandbLogger(
            project=wandb_project,
            name=experiment_name,
            group=args.experiment_id,
            log_model=False,
        )
        
        wandb_logger.experiment.config.update(vars(args))
        wandb_logger.experiment.config.update({"fold_idx": fold_idx, "n_folds": n_folds})
        
        # Set up checkpoint directory for this fold
        run_dir = ckpt_dir(args.dataset, args.experiment_id) / f"fold_{fold_idx}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        ckpt_callback = ModelCheckpoint(
            dirpath=str(run_dir),
            filename="best-{val_miou:.4f}",
            monitor="metrics/val_iou_all",
            mode="max",
            save_top_k=1,
            save_last=False,
            save_weights_only=True,
        )
        
        # Early stopping: stop if no improvement for patience epochs
        early_stop_patience = getattr(args, "early_stop_patience", 15)
        early_stop_callback = EarlyStopping(
            monitor="metrics/val_iou_all",
            patience=early_stop_patience,
            mode="max",
            verbose=True,
            min_delta=0.0009,
        )
        
        run_config = vars(args).copy()
        run_config.update({
            "fold_idx": fold_idx,
            "n_folds": n_folds,
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
            "early_stop_patience": early_stop_patience,
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
        
        # Train this fold
        trainer = L.Trainer(
            accelerator=accelerator,
            devices=devices,
            max_epochs=args.epochs,
            precision=precision,
            log_every_n_steps=10,
            num_sanity_val_steps=0,
            detect_anomaly=False,
            logger=wandb_logger,
            callbacks=[ckpt_callback, early_stop_callback],
        )
        
        trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
        
        # Store fold results
        best_val_miou = ckpt_callback.best_model_score.item() if ckpt_callback.best_model_score is not None else 0.0
        fold_results.append({
            "fold": fold_idx,
            "val_miou": best_val_miou,
            "checkpoint_path": str(ckpt_callback.best_model_path) if ckpt_callback.best_model_path else None,
        })
        
        # Clean up WandB for this fold
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass
    
    # 7. Save cross-validation summary
    summary_dir = ckpt_dir(args.dataset, args.experiment_id)
    summary_path = summary_dir / "cv_results.json"
    
    # Calculate statistics
    val_mious = [r["val_miou"] for r in fold_results]
    cv_summary = {
        "folds": fold_results,
        "summary_statistics": {
            "mean": float(np.mean(val_mious)),
            "std": float(np.std(val_mious)),
            "median": float(np.median(val_mious)),
            "min": float(np.min(val_mious)),
            "max": float(np.max(val_mious)),
        },
        "config": vars(args),
    }
    
    with open(summary_path, "w") as f:
        json.dump(cv_summary, f, indent=4)
        
if __name__ == "__main__":
    main()