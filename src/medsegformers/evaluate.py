# evaluating.py
import argparse
from pathlib import Path
import os
import torch
import numpy as np
from medsegformers.data import HyperKvasirDataset, EndoscopyDataset
from medsegformers.transforms import get_transforms
from medsegformers.models import build as build_model

from monai.data import DataLoader, list_data_collate, decollate_batch
from monai.transforms import Compose, Activations, AsDiscrete
from monai.metrics import DiceMetric, MeanIoU, HausdorffDistanceMetric
import monai
from tqdm import tqdm

def project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def get_data_root() -> Path:
    return project_root() / "data"

def ckpt_dir(dataset: str, experiment_id: str) -> Path:
    return project_root() / "experiments" / dataset / experiment_id / "checkpoints"


# ----- Class names for reporting (align with your COLOR_MAP / label indexing)
ENDOSCOPY_CLASS_NAMES = [
    "background",       # 0
    "cystic plate",     # 1
    "Calot triangle",   # 2
    "cystic artery",    # 3
    "cystic duct",      # 4
    "gallbladder",      # 5
    "tools",            # 6
]



def get_args_parser():
    parser = argparse.ArgumentParser("Evaluate a trained MONAI UNet")
    parser.add_argument("--dataset", type=str, choices=["hyperkvasir", "endoscopy"], required=True)
    parser.add_argument("--experiment-id", type=str, required=False, help="look in experiments/<dataset>/<experiment_id>/checkpoints")
    parser.add_argument("--model", type=str, default="unet", help="Model name registered in medsegformers.models (e.g., 'unet')")
    parser.add_argument("--image-size", type=int, nargs=2, default=None, help="(H W)")
    parser.add_argument("--tf-kind", type=str, default="basic", choices=["basic", "aug"], help="Transforms for eval")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth state_dict")
    return parser



def create_dataset(args):

    tf = get_transforms(dataset=args.dataset, kind=args.tf_kind, image_size=args.image_size)
    data_root = get_data_root()

    if args.dataset == "hyperkvasir":
        dataset_root = data_root / "HyperKvasir"
        num_classes = 1
        test_ds = HyperKvasirDataset(root=dataset_root, split="test", transform=tf)

    elif args.dataset == "endoscopy":
        dataset_root = data_root / "endoscapes_segmentation_dataset\endoscapes_segmentations_processed"
        ratio = (0.7, 0.2, 0.1)
        test_ds = EndoscopyDataset(root=dataset_root, split="test", transform=tf, split_ratio=ratio, seed=args.seed)
        num_classes = 7

    return test_ds, num_classes


def evaluate(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds, num_classes = create_dataset(args)

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(
        args.model,
        in_channels=3,
        out_channels=num_classes,
    ).to(device)


    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt)

    model.eval()

    # --- Post transforms (match training)
    if num_classes == 1:
        # For Dice we can use a single-channel binary mask.
        post_pred_dice = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
        # For IoU it's convenient to one-hot to 2 classes (background + foreground)
        post_pred_iou = post_pred_dice
        post_label_iou = Compose([AsDiscrete(to_onehot=2)])
    else:
        post_pred_dice = Compose([Activations(softmax=True), AsDiscrete(argmax=True, to_onehot=num_classes)])
        post_pred_iou  = post_pred_dice
        post_label     = Compose([AsDiscrete(to_onehot=num_classes)])

    # --- Metrics
    # Include background=False to exclude class 0 from the averages
    dice_metric = DiceMetric(include_background=False, reduction="none")  # per-class results
    miou_metric = MeanIoU(include_background=False, reduction="none")

    hd95_metric = HausdorffDistanceMetric(
        include_background=False,
        percentile=95.0,
        directed=False,
        reduction="none",
    )

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            images, labels = batch["image"].to(device), batch["label"].to(device)

            outputs = model(images)

            # ---- Dice (one-hot predictions + one-hot labels for multiclass; binary is fine as 1-channel)
            if num_classes == 1:
                dice_preds = [post_pred_dice(x) for x in decollate_batch(outputs)]
                dice_labels = decollate_batch(labels)  # (B,1,H,W)
            else:
                dice_preds = [post_pred_dice(x) for x in decollate_batch(outputs)]
                dice_labels = [post_label(x) for x in decollate_batch(labels)]

            dice_metric(y_pred=dice_preds, y=dice_labels)

            # ---- mIoU (use 1-hot for both cases; binary -> 2 classes)
            if num_classes == 1:
                iou_preds = [post_pred_iou(x) for x in decollate_batch(outputs)]      # (B,2,H,W)
                iou_labels = [post_label_iou(x) for x in decollate_batch(labels)]    # (B,2,H,W)
            else:
                iou_preds = dice_preds       # already one-hot (C,H,W)
                iou_labels = dice_labels     # already one-hot
            miou_metric(y_pred=iou_preds, y=iou_labels)

            # ---- HD95 (needs one-hot)
            hd95_metric(y_pred=iou_preds, y=iou_labels)

    # aggregate
    dice_per_class = dice_metric.aggregate().cpu().numpy()   # shape: (C-1,) when exclude_background=True
    miou_per_class = miou_metric.aggregate().cpu().numpy()
    hd95_per_class = hd95_metric.aggregate().cpu().numpy()

    dice_metric.reset()
    miou_metric.reset()
    hd95_metric.reset()

    # # Reporting helpers
    if num_classes == 1:
        class_names = ["foreground"]
    else:
        # exclude background for printing to match include_background=False
        class_names = ENDOSCOPY_CLASS_NAMES[1:]  # drop background


    # Convert to numpy and reduce over the image dimension (axis=0)

    dice_raw = np.asarray(dice_per_class)   # shape: [N, C]
    miou_raw = np.asarray(miou_per_class)   # shape: [N, C]
    hd95_raw = np.asarray(hd95_per_class)   # shape: [N, C]

    # Per-class means (ignore NaNs that come from absent classes in some images)
    dice_cls = np.nanmean(dice_raw, axis=0)    # shape [C]
    miou_cls = np.nanmean(miou_raw, axis=0)    # shape [C]
    hd95_cls = np.nanmean(hd95_raw, axis=0) if hd95_raw.size else np.array([])

    # Now print per-class scalars
    for c, cname in enumerate(class_names):
        d = dice_cls[c].item() if c < dice_cls.size else float("nan")
        j = miou_cls[c].item() if c < miou_cls.size else float("nan")
        h = hd95_cls[c].item() if hd95_cls.size and c < hd95_cls.size else float("nan")
        print(f"{c:>2} {cname:>18} | Dice: {d:0.4f} | mIoU: {j:0.4f} | HD95 (px): {h:0.3f}")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    evaluate(args)
