
from typing import Optional

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from monai.data import decollate_batch
from monai.metrics import DiceMetric, MeanIoU
from monai.transforms import Compose, Activations, AsDiscrete

from torchvision.utils import make_grid

# Color utils used by your non-Lightning trainer for multi-class previews
# (keeps the same W&B visual parity as your current pipeline).
from medsegformers.utils.vis import colorize_index_map, to_np_uint8  # :contentReference[oaicite:0]{index=0}

# Simple Dice+CE loss wrapper you already ship (binary or multi-class).
from medsegformers.losses.dicece import FlexDiceCELoss  # :contentReference[oaicite:1]{index=1}


class EncoderDecoderSegModule(L.LightningModule):
    """
    Lightning module for classic encoder–decoder semantic segmentation.

    - Loss: Dice + CrossEntropy (MONAI-backed wrapper)
    - Metrics: Dice + mIoU (MONAI)
    - Logs: train/val loss, Dice, mIoU, and W&B images
    """

    def __init__(
        self,
        network: nn.Module,
        num_classes: int,
        lr: float = 2e-4,
        weight_decay: float = 0.05,
        ignore_index: Optional[int] = None,
        # image logging
        log_first_val_batch_images: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["network"])
        self.network = network
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.ignore_index = ignore_index
        self.log_images = log_first_val_batch_images

        # ---- loss
        self.criterion = FlexDiceCELoss(num_classes=num_classes)  # :contentReference[oaicite:2]{index=2}

        # ---- post-proc for metrics (match your Trainer)  :contentReference[oaicite:3]{index=3}
        if num_classes == 1:
            self.post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
            self.post_label = None
        else:
            self.post_pred = Compose([Activations(softmax=True), AsDiscrete(argmax=True, to_onehot=num_classes)])
            self.post_label = Compose([AsDiscrete(to_onehot=num_classes)])

        # ---- MONAI metrics (global per-epoch accumulators)
        self.dice_metric = DiceMetric(include_background=True, reduction="mean")
        self.iou_metric = MeanIoU(include_background=True, reduction="mean")

    # ----------------- Lightning hooks -----------------

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def forward(self, x):
        return self.network(x)

    # ---- Train
    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        logits = self(images)
        loss = self.criterion(logits, labels)

        # parity with your logger keys
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    # ---- Validate
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        logits = self(images)
        loss = self.criterion(logits, labels)

        # MONAI metrics accumulation (parity with your loops.py)  :contentReference[oaicite:4]{index=4}
        y_pred = [self.post_pred(x) for x in decollate_batch(logits)]
        if self.num_classes == 1:
            y_true = decollate_batch(labels)
        else:
            y_true = [self.post_label(x) for x in decollate_batch(labels)]

        self.dice_metric(y_pred=y_pred, y=y_true)
        self.iou_metric(y_pred=y_pred, y=y_true)

        # log first batch images to W&B (parity with your Trainer)  :contentReference[oaicite:5]{index=5}
        if self.log_images and batch_idx == 0 and hasattr(self.logger, "experiment"):
            self._log_wandb_images(images, labels, logits)

        self.log("valid_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)
        return loss

    def on_validation_epoch_end(self):
        dice = self.dice_metric.aggregate().item()
        miou = self.iou_metric.aggregate().item()
        self.dice_metric.reset()
        self.iou_metric.reset()

        # match trainer keys
        self.log("dice_score", dice, prog_bar=True, sync_dist=False)
        self.log("mean_iou", miou, prog_bar=True, sync_dist=False)

    # ----------------- helpers -----------------

    def _log_wandb_images(self, images: torch.Tensor, labels: torch.Tensor, logits: torch.Tensor):
        """
        Mirrors medsegformers.engines.trainer._wandb_image_logger so your dashboards look the same.
        """
        try:
            import wandb
        except Exception:
            return

        if self.num_classes == 1:
            preds = (logits.sigmoid() > 0.5).float()
            images_grid = make_grid(images, nrow=2, normalize=True, scale_each=True)
            preds_grid  = make_grid(preds,  nrow=2, normalize=True, scale_each=True)
            labels_grid = make_grid(labels.float(), nrow=2, normalize=True, scale_each=True)
            self.logger.experiment.log({
                "val_images/original":   wandb.Image(images_grid.permute(1,2,0).cpu().numpy()),
                "val_images/prediction": wandb.Image(preds_grid.permute(1,2,0).cpu().numpy()),
                "val_images/label":      wandb.Image(labels_grid.permute(1,2,0).cpu().numpy()),
            }, commit=False)
        else:
            # indices → RGB using your palette helper
            preds_idx = logits.softmax(1).argmax(1)                # [B,H,W]
            pred_rgb  = colorize_index_map(preds_idx)              # uint8 [B,3,H,W]
            lab_rgb   = colorize_index_map(labels.squeeze(1).long())  # uint8 [B,3,H,W]

            pred_grid_u8 = make_grid(pred_rgb, nrow=2, normalize=False, pad_value=255)
            lab_grid_u8  = make_grid(lab_rgb,  nrow=2, normalize=False, pad_value=255)
            img_grid     = make_grid(images,    nrow=2, normalize=True, scale_each=True, pad_value=1.0)

            self.logger.experiment.log({
                "val_images/original":   wandb.Image(img_grid.permute(1,2,0).cpu().numpy()),
                "val_images/prediction": wandb.Image(to_np_uint8(pred_grid_u8)),
                "val_images/label":      wandb.Image(to_np_uint8(lab_grid_u8)),
            }, commit=False)
