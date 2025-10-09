
from typing import Optional
import wandb
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torchvision.utils import make_grid
from monai.data import decollate_batch
from monai.metrics import DiceMetric, MeanIoU
from monai.transforms import Compose, Activations, AsDiscrete

from medsegformers.utils.vis import colorize_index_map, to_np_uint8
from medsegformers.losses.dicece import FlexDiceCELoss
from medsegformers.engines.eomt.two_stage_warmup_poly_schedule import TwoStageWarmupPolySchedule


class EncoderDecoderSegModule(L.LightningModule):

    def __init__(
        self,
        network: nn.Module,
        num_classes: int,
        lr: float = 2e-4,
        weight_decay: float = 0.05,
        llrd: float = 0.8,
        lr_multi: float = 3.0,
        poly_power: float = 0.9,
        warmup_steps: tuple[int, int] = (230, 430),
        llrd_l2_enabled: bool = True,
        log_first_val_batch_images: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["network"])
        self.network = network
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.log_images = log_first_val_batch_images
        self.llrd = llrd
        self.lr_multi = lr_multi
        self.poly_power = poly_power
        self.warmup_steps = warmup_steps
        self.llrd_l2_enabled = llrd_l2_enabled


        self.criterion = FlexDiceCELoss(num_classes=num_classes) 

        if num_classes == 1:
            self.post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
            self.post_label = None
        else:
            self.post_pred = Compose([Activations(softmax=True), AsDiscrete(argmax=True, to_onehot=num_classes)])
            self.post_label = Compose([AsDiscrete(to_onehot=num_classes)])

        self.dice_metric = DiceMetric(include_background=False, reduction="mean")
        self.iou_metric = MeanIoU(include_background=False, reduction="mean")


    def configure_optimizers(self):
        enc_backbone = self.network.encoder.encoder.backbone
        backbone_blocks = len(enc_backbone.blocks)

        backbone_fullnames = {
            f"network.encoder.encoder.backbone.{n}" for n, _ in enc_backbone.named_parameters()
        }

        backbone_param_groups = []
        other_param_groups = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            lr = self.lr

            if name in backbone_fullnames:
                block_idx = None
                parts = name.split(".")
                for i, p in enumerate(parts):
                    if p == "blocks" and i + 1 < len(parts):
                        try:
                            block_idx = int(parts[i + 1])
                        except ValueError:
                            block_idx = None
                        break

                if block_idx is not None:
                    lr *= self.llrd ** (backbone_blocks - 1 - block_idx)
                if "backbone.norm" in name:
                    lr = self.lr

                backbone_param_groups.append({"params": [param], "lr": lr, "name": name})
            else:
                if self.lr_multi != 1.0:
                    lr *= self.lr_multi
                other_param_groups.append({"params": [param], "lr": lr, "name": name})

        param_groups = backbone_param_groups + other_param_groups
        optimizer = AdamW(param_groups, weight_decay=self.weight_decay)

        scheduler = TwoStageWarmupPolySchedule(
            optimizer=optimizer,
            num_backbone_params=len(backbone_param_groups),
            warmup_steps=self.warmup_steps,
            total_steps=self.trainer.estimated_stepping_batches,
            poly_power=self.poly_power,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        logits = self(images)
        loss = self.criterion(logits, labels)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        logits = self(images)
        loss = self.criterion(logits, labels)

        y_pred = [self.post_pred(x) for x in decollate_batch(logits)]
        if self.num_classes == 1:
            y_true = decollate_batch(labels)
        else:
            y_true = [self.post_label(x) for x in decollate_batch(labels)]

        self.dice_metric(y_pred=y_pred, y=y_true)
        self.iou_metric(y_pred=y_pred, y=y_true)

        if self.log_images and batch_idx == 0 and hasattr(self.logger, "experiment"):
            self._log_wandb_images(images, labels, logits)

        self.log("valid_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)
        return loss

    def on_validation_epoch_end(self):
        dice = self.dice_metric.aggregate().item()
        miou = self.iou_metric.aggregate().item()
        self.dice_metric.reset()
        self.iou_metric.reset()
        self.log("dice_score", dice, prog_bar=True, sync_dist=False)
        self.log("mean_iou", miou, prog_bar=True, sync_dist=False)

    def _log_wandb_images(self, images: torch.Tensor, labels: torch.Tensor, logits: torch.Tensor):

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
            preds_idx = logits.softmax(1).argmax(1)       
            pred_rgb  = colorize_index_map(preds_idx)    
            lab_rgb   = colorize_index_map(labels.squeeze(1).long())

            pred_grid_u8 = make_grid(pred_rgb, nrow=2, normalize=False, pad_value=255)
            lab_grid_u8  = make_grid(lab_rgb,  nrow=2, normalize=False, pad_value=255)
            img_grid     = make_grid(images,    nrow=2, normalize=True, scale_each=True, pad_value=1.0)

            self.logger.experiment.log({
                "val_images/original":   wandb.Image(img_grid.permute(1,2,0).cpu().numpy()),
                "val_images/prediction": wandb.Image(to_np_uint8(pred_grid_u8)),
                "val_images/label":      wandb.Image(to_np_uint8(lab_grid_u8)),
            }, commit=False)


    



