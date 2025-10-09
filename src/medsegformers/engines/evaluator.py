from typing import Tuple, Optional
from pathlib import Path
import numpy as np
import torch
from monai.data import decollate_batch
from monai.transforms import Compose, Activations, AsDiscrete
from monai.metrics import DiceMetric, MeanIoU, HausdorffDistanceMetric
from tqdm import tqdm

from medsegformers.utils.vis import ENDOSCOPY_CLASS_NAMES


"""

NEEDS TO BE CHANGED

"""
class Evaluator:
    def __init__(self, model: torch.nn.Module, num_classes: int, device: torch.device):
        self.model = model
        self.num_classes = num_classes
        self.device = device

        if num_classes == 1:
            self.post_pred_dice = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
            self.post_pred_iou  = self.post_pred_dice
            self.post_label_iou = Compose([AsDiscrete(to_onehot=2)])
            self.post_label     = None
        else:
            self.post_pred_dice = Compose([Activations(softmax=True), AsDiscrete(argmax=True, to_onehot=num_classes)])
            self.post_pred_iou  = self.post_pred_dice
            self.post_label     = Compose([AsDiscrete(to_onehot=num_classes)])
            self.post_label_iou = self.post_label

        # per-image, per-class metrics
        self.dice_metric = DiceMetric(include_background=True,  reduction="none")
        self.miou_metric = MeanIoU   (include_background=True,  reduction="none")
        self.hd95_metric = HausdorffDistanceMetric(include_background=True, percentile=95.0, directed=False, reduction="none")

    def load_checkpoint(self, ckpt_path: str | Path):
        ckpt = torch.load(Path(ckpt_path), map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt)
        self.model.eval()

    @torch.no_grad()
    def run(self, loader, *, dataset: str):
        device = self.device
        model  = self.model

        for batch in tqdm(loader, desc="Evaluating", leave=False):
            images, labels = batch["image"].to(device), batch["label"].to(device)
            outputs = model(images)

            # Dice inputs
            if self.num_classes == 1:
                dice_preds  = [self.post_pred_dice(x) for x in decollate_batch(outputs)]
                dice_labels = decollate_batch(labels)  # (B,1,H,W)
            else:
                dice_preds  = [self.post_pred_dice(x) for x in decollate_batch(outputs)]
                dice_labels = [self.post_label(x)     for x in decollate_batch(labels)]

            self.dice_metric(y_pred=dice_preds, y=dice_labels)

            # IoU / HD95 use one-hot for both cases
            if self.num_classes == 1:
                iou_preds  = [self.post_pred_iou(x)  for x in decollate_batch(outputs)]
                iou_labels = [self.post_label_iou(x) for x in decollate_batch(labels)]
            else:
                iou_preds, iou_labels = dice_preds, dice_labels

            self.miou_metric(y_pred=iou_preds, y=iou_labels)
            self.hd95_metric(y_pred=iou_preds, y=iou_labels)

        # aggregate
        dice_raw = self.dice_metric.aggregate().cpu().numpy()   # [N, C]
        miou_raw = self.miou_metric.aggregate().cpu().numpy()   # [N, C]
        hd95_raw = self.hd95_metric.aggregate().cpu().numpy()   # [N, C]

        self._reset()

        dice_cls = np.nanmean(dice_raw, axis=0) if dice_raw.size else np.array([])
        miou_cls = np.nanmean(miou_raw, axis=0) if miou_raw.size else np.array([])
        hd95_cls = np.nanmean(hd95_raw, axis=0) if hd95_raw.size else np.array([])

        # class names
        if self.num_classes == 1:
            class_names = ["foreground"]
        else:
            class_names = ENDOSCOPY_CLASS_NAMES

        # per-class report
        for c, cname in enumerate(class_names):
            d = dice_cls[c].item() if c < dice_cls.size else float("nan")
            j = miou_cls[c].item() if c < miou_cls.size else float("nan")
            h = hd95_cls[c].item() if hd95_cls.size and c < hd95_cls.size else float("nan")
            print(f"{c:>2} {cname:>18} | Dice: {d:0.4f} | mIoU: {j:0.4f} | HD95 (px): {h:0.3f}")

        # overall averages
        mean_dice = np.nanmean(dice_cls).item() if dice_cls.size else float("nan")
        mean_miou = np.nanmean(miou_cls).item() if miou_cls.size else float("nan")
        mean_hd95 = np.nanmean(hd95_cls).item() if hd95_cls.size else float("nan")
        print(f"{'Overall Average':>20} | Dice: {mean_dice:0.4f} | mIoU: {mean_miou:0.4f} | HD95 (px): {mean_hd95:0.3f}")

        return dice_cls, miou_cls, hd95_cls

    def _reset(self):
        self.dice_metric.reset()
        self.miou_metric.reset()
        self.hd95_metric.reset()
