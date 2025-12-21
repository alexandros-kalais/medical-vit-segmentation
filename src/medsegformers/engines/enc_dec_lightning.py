
from typing import Optional
import wandb
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torchvision.utils import make_grid
from torchmetrics.classification import MulticlassJaccardIndex
from monai.data import decollate_batch
from monai.transforms import Compose, Activations, AsDiscrete

from medsegformers.utils import colorize_index_map, to_np_uint8
from medsegformers.losses import FlexDiceCELoss
from .two_stage_warmup_poly_schedule import TwoStageWarmupPolySchedule

IGNORE_INDEX = 255

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
        self.iou_macro = MulticlassJaccardIndex(num_classes=num_classes, ignore_index=IGNORE_INDEX, average=None, validate_args=False
)

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

        preds = logits.argmax(dim=1)
        self.iou_macro.update(preds, labels.squeeze(1))
        if self.log_images and batch_idx == 0 and hasattr(self.logger, "experiment"):
            self._log_wandb_images(images, labels, logits)

        self.log("valid_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)
        return loss

    def on_validation_epoch_end(self):
        per_class = self.iou_macro.compute()
        for c, v in enumerate(per_class):
            self.log(f"metrics/val_iou_class_{c}", float(v), prog_bar=False, sync_dist=False)
        
        mean_iou = float(per_class.mean())
        self.log("metrics/val_iou_all", mean_iou, prog_bar=True, sync_dist=False)
        self.log("val_miou", mean_iou, on_epoch=True, prog_bar=False, sync_dist=True)
        self.iou_macro.reset()

    def _log_wandb_images(self, images: torch.Tensor, labels: torch.Tensor, logits: torch.Tensor):

            preds_idx = logits.softmax(1).argmax(1)          
            lab_idx   = labels.squeeze(1).long()              

            ignore_mask = (lab_idx == IGNORE_INDEX)
            preds_idx_vis = preds_idx.clone()
            preds_idx_vis[ignore_mask] = IGNORE_INDEX        

            pred_rgb = colorize_index_map(preds_idx_vis, self.num_classes)   
            lab_rgb  = colorize_index_map(lab_idx, self.num_classes)

            pred_grid_u8 = make_grid(pred_rgb, nrow=2, normalize=False, pad_value=255)
            lab_grid_u8  = make_grid(lab_rgb,  nrow=2, normalize=False, pad_value=255)
            img_grid     = make_grid(images,   nrow=2, normalize=True, scale_each=True, pad_value=1.0)

            self.logger.experiment.log({
                "val_images/original":   wandb.Image(img_grid.permute(1,2,0).cpu().numpy()),
                "val_images/prediction": wandb.Image(to_np_uint8(pred_grid_u8)),
                "val_images/label":      wandb.Image(to_np_uint8(lab_grid_u8)),
            }, commit=False)


    



