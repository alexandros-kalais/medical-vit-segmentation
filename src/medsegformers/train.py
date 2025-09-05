
from pathlib import Path
import torch
import numpy as np
from monai.data import ArrayDataset, DataLoader, decollate_batch, list_data_collate
from monai.metrics import DiceMetric
import monai
from monai.transforms import Compose, Activations, AsDiscrete
import wandb
from torchvision.utils import make_grid
import os
from argparse import ArgumentParser

from tqdm import tqdm
from medsegformers.data import HyperKvasirDataset, EndoscopyDataset
from medsegformers.transforms import get_transforms
from medsegformers.losses import FlexDiceLoss as DiceLoss
from medsegformers.models import build as build_model

def get_data_root() -> Path:
    """
    Returns the absolute path to the project's data folder.
    """
    # train.py → src/medsegformers/
    # parents[2] → medSegformers/
    project_root = Path(__file__).resolve().parents[2]
    return project_root / "data"

def project_root() -> Path:
    # src/medsegformers/train.py → repo root is two levels up
    return Path(__file__).resolve().parents[2]

def ckpt_dir(args) -> Path:
    return project_root() / "experiments" / args.dataset / args.experiment_id / "checkpoints"

def make_ckpt(epoch, model, optimizer, extra=None):
    return {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "extra": extra or {},  # e.g., {"val_dice": 0.82}
    }



COLOR_MAP = torch.tensor([
    [  0,   0,   0],  # 0 background
    [255,   0,   0],  # 1 cystic plate
    [  0, 255,   0],  # 2 Calot triangle
    [  0,   0, 255],  # 3 cystic artery
    [255, 255,   0],  # 4 cystic duct
    [255,   0, 255],  # 5 gallbladder
    [  0, 255, 255],  # 6 tools
], dtype=torch.uint8)  # (7,3)

def colorize_index_map(idx_map: torch.Tensor) -> torch.Tensor:
    """
    idx_map: (B,H,W) long in [0..6]
    returns: (B,3,H,W) uint8 colored tensor
    """
    # COLOR_MAP[idx] -> (B,H,W,3), then permute to (B,3,H,W)
    colored = COLOR_MAP.to(idx_map.device)[idx_map]                  # (B,H,W,3)
    return colored.permute(0, 3, 1, 2).contiguous()                # (B,3,H,W)


def get_args_parser():
    parser = ArgumentParser("Training for medical ViT segmentation model")
    parser.add_argument("--dataset", type=str, choices = ["hyperkvasir", "endoscopy"], required=True, help="Dataset to use for training")
    parser.add_argument("--model", type=str, default="unet", help="Model name registered in medsegformers.models (e.g., 'unet')")
    parser.add_argument("--image-size", type=int, nargs=2, default=None, help="Size of images (height width)")
    parser.add_argument("--train-tf-kind", type=str, default="basic", choices=["basic", "aug"], help= "Transformation types for training images")
    parser.add_argument("--val-tf-kind", type=str, default="basic", choices=["basic", "aug"], help= "Transformation types for validation images")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of workers for data loaders") #CHECK LATER
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--experiment-id", type=str, default="attentionUnet-test-run", help="Experiment ID for Weights & Biases")

    return parser


def create_dataset(args):

    data_root = get_data_root()

    train_tf = get_transforms(dataset=args.dataset, kind=args.train_tf_kind, image_size=args.image_size)
    val_tf = get_transforms(args.dataset, kind=args.val_tf_kind, image_size=args.image_size)

    if args.dataset == "hyperkvasir":
        dataset_root = data_root / "HyperKvasir"
        num_classes = 1
        train_ds = HyperKvasirDataset(root=dataset_root, split="train", transform=train_tf)
        val_ds   = HyperKvasirDataset(root=dataset_root, split="validation", transform=val_tf)

    elif args.dataset == "endoscopy":
        dataset_root = data_root / "endoscapes_segmentation_dataset\endoscapes_segmentations_processed"
        ratio = (0.7, 0.2, 0.1)
        train_ds = EndoscopyDataset(root=dataset_root, split="train", transform=train_tf, split_ratio=ratio, seed=args.seed)
        val_ds   = EndoscopyDataset(root=dataset_root, split="train", transform=val_tf, split_ratio=ratio, seed=args.seed)
        num_classes = 7


    return train_ds, val_ds, num_classes

def train(args):

    wandb.login()

    wandb.init(
        project="Internship-medical-vit-segmentation",
        name=args.experiment_id,
        config=vars(args),
    )

    # Create output directory if it doesn't exist
    # Build the full path
    output_dir = os.path.join(
        "experiments",          # top-level experiments folder
        args.dataset,           # dataset name (e.g., "hyperkvasir")
        args.experiment_id,     # run name (e.g., "unet-baseline")
        "checkpoints"           # store only checkpoint files here
    )
    os.makedirs(output_dir, exist_ok=True)    

    torch.manual_seed(args.seed)

    # torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, val_ds, num_classes = create_dataset(args)

    train_loader = DataLoader(train_ds,
     batch_size=args.batch_size,
     shuffle=True,
     num_workers=args.num_workers,
     collate_fn=list_data_collate,
     pin_memory=torch.cuda.is_available())

    val_loader = DataLoader(val_ds,
     batch_size=args.batch_size,
     num_workers=args.num_workers,
     collate_fn=list_data_collate,
     pin_memory=torch.cuda.is_available())

    
    model = build_model(
        args.model,
        in_channels=3,
        out_channels=num_classes,
        # channels=(16, 32, 64, 128, 256),
        # num_res_units=2,
    ).to(device)



    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    dice_metric = DiceMetric(include_background=False, reduction="mean") #See if you want background for binary or not
    criterion = DiceLoss(num_classes)

    if num_classes == 1:
        
        post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    elif num_classes > 1:
        
        post_pred = Compose([Activations(softmax=True), AsDiscrete(argmax=True, to_onehot=num_classes)])
        post_label = Compose([AsDiscrete(to_onehot=num_classes)])
 

    ### ---------Training Loop --------
    best_valid_loss = float('inf')
    current_best_model_path = None
    for epoch in range(args.epochs):

        # Training
        model.train()

        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)):

            images, labels = batch["image"].to(device), batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            wandb.log({
                "train_loss": loss.item(),
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch": epoch + 1
            }, step=epoch * len(train_loader) + i)


        # Validation
        model.eval()

        with torch.no_grad():
            losses = []
            for i, batch in enumerate(val_loader):

                images, labels = batch["image"].to(device), batch["label"].to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                losses.append(loss.item())

                if i == 0:
                    
                    if num_classes == 1:

                        predictions = torch.sigmoid(outputs)
                        predictions = (predictions > 0.5).float()
                        labels_grid = make_grid(labels, nrow=2, pad_value=1)

                    elif num_classes > 1:

                        predictions = outputs.softmax(1).argmax(1)
                        predictions = colorize_index_map(predictions)
                        labels_colorized = colorize_index_map(labels.squeeze(1).long())
                        labels_grid = make_grid(labels_colorized, nrow=2, pad_value=1)

                    images_grid = make_grid(images, nrow=2, pad_value=1)
                    predictions_grid = make_grid(predictions, nrow=2, pad_value=1)

                    wandb.log({
                        "val_images/original_image": wandb.Image(images_grid.permute(1, 2, 0).cpu().numpy()),
                        "val_images/predictions": wandb.Image(predictions_grid.permute(1, 2, 0).cpu().numpy()),
                        "val_images/ground_truth": wandb.Image(labels_grid.permute(1, 2, 0).cpu().numpy())
                    }, step=(epoch + 1) * len(train_loader) - 1)



                outputs = [post_pred(x) for x in decollate_batch(outputs)]

                if num_classes == 1:
                    labels = decollate_batch(labels)
                elif num_classes > 1:
                    labels = [post_label(x) for x in decollate_batch(labels)]

                dice_metric(y_pred=outputs, y=labels)


            valid_loss = sum(losses) / len(losses)
            dice = dice_metric.aggregate().item()
            dice_metric.reset()

            wandb.log({
                "valid_loss": valid_loss,
                "dice_score": dice
            }, step=(epoch + 1) * len(train_loader) - 1)


            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if current_best_model_path:
                    os.remove(current_best_model_path)
                current_best_model_path = os.path.join(
                    output_dir, 
                    f"best_model-epoch={epoch:04}-val_loss={valid_loss:04}.pth"
                )
                torch.save(model.state_dict(), current_best_model_path)

    print("Training completed!")

    # Save the model
    torch.save(
        model.state_dict(),
        os.path.join(
            output_dir,
            f"final_model-epoch={epoch:04}-val_loss={valid_loss:04}.pth"
        )
    )

    wandb.finish()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    train(args)



#     python your_script.py \
#   --dataset endoscopy \
#   --data-dir ./my_data \
#   --train-tf-kind aug \
#   --val-tf-kind basic

