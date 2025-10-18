import math
import os, sys, json
from typing import Iterable, List, Tuple
import torch.nn as nn 
import lightning as L
import torch
from monai.data import DataLoader
from torch.utils.data import Subset
from lightning.pytorch.loggers import WandbLogger
from medsegformers.data import get_dataset_class
from medsegformers.transforms import get_transforms
from medsegformers.utils.paths import get_data_root, ckpt_dir
from medsegformers.models.eomt import EoMT
from medsegformers.models import ViT
from medsegformers.cli.config import load_config
from medsegformers.engines.eomt.mask_classification_semantic import MaskClassificationSemantic
from datetime import datetime
from lightning.pytorch.callbacks import ModelCheckpoint

def eomt_train_collate(batch):
    images, targets = [], []
    for image, target in batch:
        images.append(image)
        targets.append(target)
    images = torch.stack(images)
    return images, targets


def eomt_eval_collate(batch):
    return tuple(zip(*batch))


def compute_mask_anneal_windows(
    total_steps: int,
    num_blocks: int,
    a_start_frac: float = 0.10,   
    a_end_frac: float   = 0.60,   
    block_span_ratio: float = 0.50  
) -> Tuple[List[int], List[int]]:

    if num_blocks <= 0 or total_steps <= 1:
        return [], []

   
    A_start = max(0, min(total_steps - 2, int(total_steps * a_start_frac)))
    A_end   = max(A_start + 1, min(total_steps - 1, int(total_steps * a_end_frac)))
    A_span  = max(2, A_end - A_start)

    
    block_dur = max(1, int(A_span * block_span_ratio))
    block_dur = min(block_dur, A_span - 1)  

    if num_blocks == 1:
        return [A_start], [A_end]

    stride = max(1, (A_span - block_dur) // (num_blocks - 1))

    starts_steps: List[int] = []
    ends_steps:   List[int] = []

    for b in range(num_blocks):
        s = A_start + b * stride
        e = s + block_dur

        e = min(e, A_end)

        if b == num_blocks - 1:
            e = A_end
            s = min(s, e - 1)

        s = max(0, min(s, total_steps - 2))
        e = max(s + 1, min(e, total_steps - 1))

        starts_steps.append(s)
        ends_steps.append(e)

    return starts_steps, ends_steps

def main():

    # 1. Load experiment config
    cfg_path = None
    for arg in sys.argv:
        if arg.endswith("yml") or arg.endswith(".yaml"):
            cfg_path = arg
            break
    if cfg_path is None:
        raise ValueError("Plese provide a config YAML path")

    args = load_config(cfg_path)

    # 2. Runtime setup
    L.seed_everything(args.seed, workers=True)
    use_gpu = torch.cuda.is_available()
    accelerator = "gpu" if use_gpu else "cpu"
    devices = 1
    precision = "16-mixed" if use_gpu else "32-true"

    # 3. Transforms and Data
    DatasetCls = get_dataset_class(args.dataset)
    root = DatasetCls.default_root(get_data_root())

    args.image_size = DatasetCls.get_image_size(
    vit_name=args.vit_name,
    user_size=getattr(args, "image_size", None)
    )

    tf_train = get_transforms(dataset=args.dataset, kind=args.train_tf_kind, image_size=args.image_size)
    tf_val   = get_transforms(dataset=args.dataset, kind=args.val_tf_kind,   image_size=args.image_size)

    train_ds = DatasetCls(split="train",      transform=tf_train, root=root, seed=args.seed, return_masks=True)
    val_ds   = DatasetCls(split="validation", transform=tf_val,   root=root, seed=args.seed, return_masks=True)

    num_classes = getattr(DatasetCls, "NUM_CLASSES", None)

    if num_classes is None:
        raise ValueError("Dataset class must define NUM_CLASSES")

    if num_classes == 2:
        num_classes = num_classes - 1
    
    if args.subset > 0:
        train_ds = torch.utils.data.Subset(train_ds, list(range(args.subset)))
        val_ds   = torch.utils.data.Subset(val_ds,   list(range(args.subset)))

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=use_gpu, collate_fn=eomt_train_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=use_gpu, collate_fn=eomt_eval_collate
    )

    # 4. Model and Module 

    H, W = args.image_size
    if any(x in args.vit_name.lower() for x in ["16-", "16_"]):
        patch_size = 16
    elif any(x in args.vit_name.lower() for x in ["14_"]):
        patch_size = 14
    else:
        raise ValueError("Define patch_size correctly!")

    encoder = ViT(
        img_size=(H, W),
        patch_size=patch_size,
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
        num_q=args.eomt_num_q,
        num_blocks=args.eomt_num_blocks,
        masked_attn_enabled=bool(args.eomt_masked_attn_enabled),
    )

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs

    non_vit_warmup = steps_per_epoch * 1
    vit_warmup = steps_per_epoch * 2       
    warmup_steps = (non_vit_warmup, vit_warmup)


    anneal_starts = [2*steps_per_epoch, 4*steps_per_epoch, 6*steps_per_epoch, 8*steps_per_epoch]
    anneal_ends = [4*steps_per_epoch, 6*steps_per_epoch, 8*steps_per_epoch, 10*steps_per_epoch]


    print(f"[INFO] steps_per_epoch={steps_per_epoch}, total_steps={total_steps}")
    print("[INFO] anneal_starts:", anneal_starts)
    print("[INFO] anneal_ends  :", anneal_ends)
    print("[INFO] warmup_steps :", warmup_steps)


    module = MaskClassificationSemantic(
        network=network,
        img_size=(H, W),
        num_classes=num_classes,
        attn_mask_annealing_enabled=bool(args.eomt_masked_attn_enabled),
        lr=args.lr,
        llrd=args.llrd,
        lr_mult=args.lr_multi,
        warmup_steps=warmup_steps,
        attn_mask_annealing_start_steps=anneal_starts,
        attn_mask_annealing_end_steps=anneal_ends,
    )

    if not hasattr(args, "experiment_id") or args.experiment_id in [None, "", "auto"]:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        vit_name_lower = args.vit_name.lower()
        if "dinov3" in vit_name_lower:
            vit_short = "dinov3"
        elif "dinov2" in vit_name_lower:
            vit_short = "dinov2"
        elif "dino" in vit_name_lower:
            vit_short = "dino"
        else:
            vit_short = "ImageNet"
        args.experiment_id = f"eomt_{vit_short}_{args.image_size[0]}_{args.image_size[1]}_lr{args.lr}_bs{args.batch_size}_{timestamp}"
        print(f"[INFO] Auto experiment_id set to: {args.experiment_id}")


    # 5. Logging and checkpoints
    wandb_logger = WandbLogger(
        project="Internship-medical-vit-segmentation",
        name=args.experiment_id,
        log_model=False,
    )

    wandb_logger.experiment.config.update(vars(args))

    run_dir = ckpt_dir(args.dataset, args.experiment_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    
    ckpt_callback = ModelCheckpoint(
    dirpath=str(run_dir),
    filename="epoch={epoch:03d}-miou={val_iou_all:.3f}",
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
        "vit_warmup": vit_warmup
        },
    "anneal_starts": anneal_starts,
    "anneal_ends": anneal_ends
    })

    with open(run_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=4)


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

    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    try:
        import wandb
        wandb.finish()
    except Exception:
        pass


if __name__ == "__main__":
    main()
