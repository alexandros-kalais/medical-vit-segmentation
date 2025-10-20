import lightning as L
import torch
from monai.data import DataLoader
from pathlib import Path
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import json
import sys
from datetime import datetime

from medsegformers.data import get_dataset_class
from medsegformers.transforms import get_transforms                         
from medsegformers.utils.paths import get_data_root, ckpt_dir
from medsegformers.models.build import build_segmentation_model   
from medsegformers.engines.enc_dec_lightning import EncoderDecoderSegModule
from medsegformers.cli.config import load_config


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

    DatasetCls = get_dataset_class(args.dataset)
    root = DatasetCls.default_root(get_data_root())

    args.image_size = DatasetCls.get_image_size(
    vit_name=args.vit_name,
    user_size=getattr(args, "image_size", None)
    )

    # 3. Transforms and Data
    tf_train = get_transforms(dataset=args.dataset, kind=args.train_tf_kind, image_size=args.image_size)
    tf_val   = get_transforms(dataset=args.dataset, kind=args.val_tf_kind,   image_size=args.image_size)

    train_ds = DatasetCls(split="train",      transform=tf_train, root=root, seed=args.seed, return_masks=False)
    val_ds   = DatasetCls(split="validation", transform=tf_val,   root=root, seed=args.seed, return_masks=False)

    num_classes = getattr(DatasetCls, "NUM_CLASSES", None)

    if num_classes is None:
        raise ValueError("Dataset class must define NUM_CLASSES")
    
    if args.subset > 0:
        train_ds = torch.utils.data.Subset(train_ds, list(range(args.subset)))

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=use_gpu
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=use_gpu
    )

    # 4. Model
    model = build_segmentation_model(
        decoder=args.decoder,                 # e.g. "naive" | "mla" | "pup" ...
        num_classes=num_classes,
        vit_name=args.vit_name,
        pretrained=True,
        freeze_encoder=args.freeze_encoder,
        image_size=tuple(args.image_size),
        decoder_kwargs=getattr(args, "decoder_kwargs", None),
        unfreeze_last_k=args.unfreeze_last_k
    ) 

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs

    non_vit_warmup = steps_per_epoch * 1
    vit_warmup = steps_per_epoch * 2       
    warmup_steps = (non_vit_warmup, vit_warmup)

    print(f"[INFO] steps_per_epoch={steps_per_epoch}, total_steps={total_steps}")
    print(f"[INFO] warmup_steps (non_vit, vit) = {warmup_steps}")

    module = EncoderDecoderSegModule(
        network=model,
        num_classes=num_classes,
        lr=args.lr,
        weight_decay=args.weight_decay,
        llrd = args.llrd,
        lr_multi = args.lr_multi,
        poly_power = args.poly_power,
        warmup_steps=warmup_steps
    )


    if not hasattr(args, "experiment_id") or args.experiment_id in [None, "", "auto"]:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        vit_name_lower = args.vit_name.lower()
        # Extract model family + size safely
        if "dinov3" in vit_name_lower:
            vit_short = "dinov3"
        elif "dinov2" in vit_name_lower:
            vit_short = "dinov2"
        elif "dino" in vit_name_lower:
            vit_short = "dino"
        else:
            vit_short = "ImageNet"
        decoder_short = args.decoder
        args.experiment_id = f"{decoder_short}_{vit_short}_{args.image_size[0]}_{args.image_size[1]}_lr{args.lr}_bs{args.batch_size}_{timestamp}"
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
    filename="{epoch:03d}-{val_loss:.3f}-{miou:.3f}",
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
        }
    })

    with open(run_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=4)

    # 6 Trainer
    trainer = L.Trainer(
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        max_epochs=args.epochs,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        logger=wandb_logger,
        callbacks=[ckpt_callback]
    )
    
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    try:
        import wandb
        wandb.finish()
    except Exception:
        pass


if __name__ == "__main__":
    main()
