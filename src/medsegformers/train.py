# train.py
from pathlib import Path
import os
import numpy as np
import torch
from argparse import ArgumentParser
from tqdm import tqdm

import wandb
from torchvision.utils import make_grid
from monai.data import DataLoader, decollate_batch, list_data_collate
from monai.metrics import DiceMetric
from monai.transforms import Compose, Activations, AsDiscrete
from monai.losses import DiceCELoss

from medsegformers.data import HyperKvasirDataset, EndoscopyDataset
from medsegformers.transforms import get_transforms
from medsegformers.models import build as build_model

# ---------- color map for visualization ----------
COLOR_MAP = torch.tensor([
    [  0,   0,   0],  # 0 background
    [255,   0,   0],  # 1 cystic plate
    [  0, 255,   0],  # 2 Calot triangle
    [  0,   0, 255],  # 3 cystic artery
    [255, 255,   0],  # 4 cystic duct
    [255,   0, 255],  # 5 gallbladder
    [  0, 255, 255],  # 6 tools
], dtype=torch.uint8)

def colorize_index_map(idx_map: torch.Tensor) -> torch.Tensor:
    """
    idx_map: (B,H,W) long or (H,W)
    returns: (B,3,H,W) uint8
    """
    if idx_map.ndim == 2:
        idx_map = idx_map.unsqueeze(0)
    idx_map = idx_map.long()
    cmap = COLOR_MAP.to(idx_map.device)
    colored = cmap[idx_map]           # (B,H,W,3)
    return colored.permute(0,3,1,2).contiguous()

def to_np_uint8(grid):
    x = grid.permute(1,2,0).cpu().numpy()
    return np.clip(x, 0, 255).astype(np.uint8)

# ---------- paths ----------
def project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def get_data_root() -> Path:
    return project_root() / "data"

def ckpt_dir(args) -> Path:
    return project_root() / "experiments" / args.dataset / args.experiment_id / "checkpoints"

# ---------- args ----------
def get_args_parser():
    p = ArgumentParser("Training for medical ViT segmentation model")
    p.add_argument("--dataset", type=str, choices=["hyperkvasir","endoscopy"], required=True)
    p.add_argument("--model", type=str, default="unet")
    p.add_argument("--image-size", type=int, nargs=2, default=None)
    p.add_argument("--train-tf-kind", type=str, default="basic", choices=["basic","aug"])
    p.add_argument("--val-tf-kind", type=str, default="basic", choices=["basic","aug"])
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--experiment-id", type=str, default="unet-baseline")
    p.add_argument("--subset", type=int, default=0)
    # p.add_argument("--overfit-one-batch", action="store_true")
    return p

# ---------- dataset ----------
def create_dataset(args):
    data_root = get_data_root()

    train_tf = get_transforms(dataset=args.dataset, kind=args.train_tf_kind, image_size=args.image_size)
    val_tf   = get_transforms(dataset=args.dataset, kind=args.val_tf_kind,  image_size=args.image_size)

    if args.dataset == "hyperkvasir":
        root = data_root / "HyperKvasir"
        num_classes = 1
        train_ds = HyperKvasirDataset(root=root, split="train",       transform=train_tf)
        val_ds   = HyperKvasirDataset(root=root, split="validation",  transform=val_tf)

    elif args.dataset == "endoscopy":

        root = data_root / "endoscapes_segmentation_dataset" / "endoscapes_segmentations_processed"
        ratio = (0.7, 0.2, 0.1)
        train_ds = EndoscopyDataset(root=root, split="train", transform=train_tf, split_ratio=ratio, seed=args.seed)
        val_ds   = EndoscopyDataset(root=root, split="validation",   transform=val_tf,   split_ratio=ratio, seed=args.seed)
        num_classes = 7

    return train_ds, val_ds, num_classes

# ---------- training ----------
def train(args):
    wandb.login()
    wandb.init(project="Internship-medical-vit-segmentation", name=args.experiment_id, config=vars(args))

    out_dir = ckpt_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, val_ds, num_classes = create_dataset(args)

    if args.subset > 0:
        train_ds = torch.utils.data.Subset(train_ds, list(range(args.subset)))
        val_ds   = torch.utils.data.Subset(val_ds,   list(range(args.subset)))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=list_data_collate,
                              pin_memory=torch.cuda.is_available())
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=list_data_collate,
                              pin_memory=torch.cuda.is_available())

    # model = build_model(args.model, in_channels=3, out_channels=num_classes).to(device)
    model = build_model( 
    args.model, 
    in_channels=3, 
    out_channels=num_classes, # # You can pass extra kwargs if you want: 
    vit_name="vit_base_patch16_224", 
    pretrained=True, 
    freeze_encoder=True, 
    img_size=tuple(args.image_size) if args.image_size else (224,224), 
     ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)  # robust for multiclass
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    # post-processing for metrics
    if num_classes == 1:
        post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    else:
        post_pred  = Compose([Activations(softmax=True), AsDiscrete(argmax=True, to_onehot=num_classes)])
        post_label = Compose([AsDiscrete(to_onehot=num_classes)])


    # --------- Training loop ----------
    best_valid_loss = float('inf')
    current_best = None

    for epoch in range(args.epochs):
        model.train()
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)):
            images, labels = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            wandb.log({"train_loss": loss.item(), "lr": optimizer.param_groups[0]["lr"], "epoch": epoch+1},
                      step=epoch*len(train_loader)+i)

        # --------- Validation ----------
        model.eval()
        losses = []
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                images, labels = batch["image"].to(device), batch["label"].to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                losses.append(loss.item())

                # Log images once per epoch (first val batch)
                if i == 1:
                    if num_classes == 1:
                        preds = (outputs.sigmoid() > 0.5).float()
                        images_grid = make_grid(images, nrow=2, normalize=True, scale_each=True)
                        preds_grid  = make_grid(preds,  nrow=2, normalize=True, scale_each=True)
                        labels_grid = make_grid(labels.float(), nrow=2, normalize=True, scale_each=True)

                        wandb.log({
                            "val_images/original":   wandb.Image(images_grid.permute(1,2,0).cpu().numpy()),
                            "val_images/prediction": wandb.Image(preds_grid.permute(1,2,0).cpu().numpy()),
                            "val_images/label":      wandb.Image(labels_grid.permute(1,2,0).cpu().numpy()),
                        }, step=(epoch+1)*len(train_loader)-1)
                    else:
                        
                                                # --- make index maps ---
                        preds_idx = outputs.softmax(1).argmax(1)                  # [B,H,W]
                        labs_idx  = labels.squeeze(1).long()                      # [B,H,W]

                        # --- colorize to uint8 ---
                        pred_rgb  = colorize_index_map(preds_idx)                 # [B,3,H,W], uint8
                        lab_rgb   = colorize_index_map(labs_idx)                  # [B,3,H,W], uint8

                        preds_idx = outputs.softmax(1).argmax(1)                   # [B,H,W]
                        pred_rgb  = colorize_index_map(preds_idx)                  # uint8
                        lab_rgb   = colorize_index_map(labels.squeeze(1).long())   # uint8
                        # DO NOT normalize palette masks
                        pred_grid_u8 = make_grid(pred_rgb, nrow=2, normalize=False)
                        lab_grid_u8  = make_grid(lab_rgb,  nrow=2, normalize=False)
                        img_grid     = make_grid(images,   nrow=2, normalize=True, scale_each=True)

                        wandb.log({
                            "val_images/original":   wandb.Image(img_grid.permute(1,2,0).cpu().numpy()),
                            "val_images/prediction": wandb.Image(to_np_uint8(pred_grid_u8)),
                            "val_images/label":      wandb.Image(to_np_uint8(lab_grid_u8)),
                        }, step=(epoch+1)*len(train_loader)-1)

                # metric
                y_pred = [post_pred(x) for x in decollate_batch(outputs)]
                if num_classes == 1:
                    y_true = decollate_batch(labels)
                else:
                    y_true = [post_label(x) for x in decollate_batch(labels)]
                dice_metric(y_pred=y_pred, y=y_true)

        valid_loss = float(np.mean(losses))
        dice = dice_metric.aggregate().item()
        dice_metric.reset()
        wandb.log({"valid_loss": valid_loss, "dice_score": dice},
                  step=(epoch+1)*len(train_loader)-1)

        # checkpoints
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            if current_best and os.path.exists(current_best):
                os.remove(current_best)
            current_best = out_dir / f"best_model-epoch={epoch:04d}-val_loss={valid_loss:.4f}.pth"
            torch.save(model.state_dict(), current_best)

    print("Training completed!")
    final_path = out_dir / f"final_model-epoch={epoch:04d}-val_loss={valid_loss:.4f}.pth"
    torch.save(model.state_dict(), final_path)
    wandb.finish()

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    train(args)
