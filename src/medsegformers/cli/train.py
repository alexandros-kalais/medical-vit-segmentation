import torch
from monai.data import DataLoader, list_data_collate
import wandb

from medsegformers.config.args import get_train_args_parser
from medsegformers.data import get_dataset_class
from medsegformers.transforms import get_transforms
from medsegformers.models import build as build_model
from medsegformers.engines.trainer import Trainer
from medsegformers.utils.paths import get_data_root

def main():
    args = get_train_args_parser().parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- dataset prep via registry + class API ---
    DatasetCls = get_dataset_class(args.dataset)

    data_root = get_data_root()
    root = DatasetCls.default_root(data_root)

    train_tf = get_transforms(dataset=args.dataset, kind=args.train_tf_kind, image_size=args.image_size)
    val_tf   = get_transforms(dataset=args.dataset, kind=args.val_tf_kind,  image_size=args.image_size)

    # Let each class handle its own defaults (e.g., split_ratio for Endoscopy)
    train_ds = DatasetCls.build_split("train",      transform=train_tf, root=root, seed=args.seed)
    val_ds   = DatasetCls.build_split("validation", transform=val_tf,   root=root, seed=args.seed)

    num_classes = getattr(DatasetCls, "NUM_CLASSES", None)
    if num_classes is None:
        raise AttributeError(f"{DatasetCls.__name__} must define class attribute NUM_CLASSES.")

    if args.subset > 0:
        train_ds = torch.utils.data.Subset(train_ds, list(range(args.subset)))
        val_ds   = torch.utils.data.Subset(val_ds,   list(range(args.subset)))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=list_data_collate,
                              pin_memory=torch.cuda.is_available())
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=list_data_collate,
                              pin_memory=torch.cuda.is_available())

    model = build_model(
        args.model,
        in_channels=3,
        out_channels=num_classes,
        vit_name="vit_base_patch16_224",
        pretrained=True,
        freeze_encoder=True,
        img_size=tuple(args.image_size) if args.image_size else (224, 224),
    ).to(device)

    wandb.login()
    run = wandb.init(project="Internship-medical-vit-segmentation",
                     name=args.experiment_id, config=vars(args))

    trainer = Trainer(args, model, train_loader, val_loader,
                      num_classes=num_classes, device=device, wandb_run=run)
    trainer.fit()
    wandb.finish()

if __name__ == "__main__":
    main()

