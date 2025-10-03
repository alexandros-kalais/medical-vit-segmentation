import lightning as L
import torch
from monai.data import DataLoader

from lightning.pytorch.loggers import WandbLogger

from medsegformers.config.args import get_train_args_parser
from medsegformers.data import get_dataset_class
from medsegformers.transforms import get_transforms                         
from medsegformers.utils.paths import get_data_root
from medsegformers.models.build import build_segmentation_model   
from medsegformers.engines.enc_dec_lightning import EncoderDecoderSegModule


def main():
    args = get_train_args_parser().parse_args()

    # ---- runtime (same defaults spirit as your EoMT script)
    L.seed_everything(getattr(args, "seed", 42), workers=True)
    use_gpu = torch.cuda.is_available()
    accelerator = "gpu" if use_gpu else "cpu"
    devices = 1
    precision = "16-mixed" if use_gpu else "32-true"

    DatasetCls = get_dataset_class(args.dataset)
    root = DatasetCls.default_root(get_data_root())

    tf_train = get_transforms(dataset=args.dataset, kind=args.train_tf_kind, image_size=args.image_size)
    tf_val   = get_transforms(dataset=args.dataset, kind=args.val_tf_kind,   image_size=args.image_size)

    train_ds = DatasetCls(split="train",      transform=tf_train, root=root, seed=args.seed, return_masks=False)
    val_ds   = DatasetCls(split="validation", transform=tf_val,   root=root, seed=args.seed, return_masks=False)

    num_classes = getattr(DatasetCls, "NUM_CLASSES", None)
    if num_classes is None:
        raise ValueError("Dataset class must define NUM_CLASSES")
    
    if args.subset > 0:
        train_ds = torch.utils.data.Subset(train_ds, list(range(args.subset)))
        val_ds   = torch.utils.data.Subset(val_ds,   list(range(args.subset)))

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=getattr(args, "num_workers", 0), pin_memory=use_gpu
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=getattr(args, "num_workers", 0), pin_memory=use_gpu
    )

    # ---- model (unified builder you added)
    model = build_segmentation_model(
        decoder=args.decoder,                 # e.g. "naive" | "mla" | "pup" ...
        num_classes=num_classes,
        vit_name=args.vit_name,
        pretrained=True,
        freeze_encoder=getattr(args, "freeze_encoder", True),
        image_size=tuple(args.image_size),
        patch_size=16,
        decoder_kwargs=getattr(args, "decoder_kwargs", None),
    )  # :contentReference[oaicite:11]{index=11}

    # ---- Lightning module for ED
    module = EncoderDecoderSegModule(
        network=model,
        num_classes=num_classes,
        lr=args.lr,
        weight_decay=getattr(args, "weight_decay", 0.05),
        ignore_index=getattr(args, "ignore_index", None),
    )

    # ---- W&B logger (same project/name convention as your EoMT script)
    wandb_logger = WandbLogger(
        project=getattr(args, "wandb_project", "Internship-medical-vit-segmentation"),
        name=args.experiment_id,
        log_model=False,
    )

    trainer = L.Trainer(
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        max_epochs=args.epochs,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        logger=wandb_logger,
    )

    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    try:
        import wandb
        wandb.finish()
    except Exception:
        pass


if __name__ == "__main__":
    main()
