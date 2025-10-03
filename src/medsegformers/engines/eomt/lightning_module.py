# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from:
# - the torchmetrics library by the PyTorch Lightning team
# - the Mask2Former repository by Facebook, Inc. and its affiliates
# All used under the Apache 2.0 License.
# ---------------------------------------------------------------

import math
from typing import Optional, cast
import lightning
from lightning.fabric.utilities import rank_zero_info
import torch
import torch.nn as nn
from torch.optim import AdamW
from torchmetrics.classification import MulticlassJaccardIndex
import wandb
from PIL import Image
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import io
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.functional import interpolate
import logging

from medsegformers.engines.eomt.two_stage_warmup_poly_schedule import TwoStageWarmupPolySchedule

bold_green = "\033[1;32m"
reset = "\033[0m"


class LightningModule(lightning.LightningModule):
    def __init__(
        self,
        network: nn.Module,
        img_size: tuple[int, int],
        num_classes: int,
        attn_mask_annealing_enabled: bool,
        attn_mask_annealing_start_steps: Optional[list[int]],
        attn_mask_annealing_end_steps: Optional[list[int]],
        lr: float,
        llrd: float,
        llrd_l2_enabled: bool,
        lr_mult: float,
        weight_decay: float,
        poly_power: float,
        warmup_steps: tuple[int, int],
        ckpt_path=None,
        delta_weights=False,
        load_ckpt_class_head=True,
    ):
        super().__init__()

        self.network = network
        self.img_size = img_size
        self.num_classes = num_classes
        self.attn_mask_annealing_enabled = attn_mask_annealing_enabled
        self.attn_mask_annealing_start_steps = attn_mask_annealing_start_steps
        self.attn_mask_annealing_end_steps = attn_mask_annealing_end_steps
        self.lr = lr
        self.llrd = llrd
        self.lr_mult = lr_mult
        self.weight_decay = weight_decay
        self.poly_power = poly_power
        self.warmup_steps = warmup_steps
        self.llrd_l2_enabled = llrd_l2_enabled

        self.strict_loading = False

        if delta_weights and ckpt_path:
            logging.info("Delta weights mode")
            self._zero_init_outside_encoder()
            current_state_dict = self.state_dict()
            ckpt = self._load_ckpt(ckpt_path, load_ckpt_class_head)
            combined_state_dict = self._add_state_dicts(current_state_dict, ckpt)
            incompatible_keys = self.load_state_dict(combined_state_dict, strict=False)
            self._raise_on_incompatible(incompatible_keys, load_ckpt_class_head)
        elif ckpt_path:
            ckpt = self._load_ckpt(ckpt_path, load_ckpt_class_head)
            incompatible_keys = self.load_state_dict(ckpt, strict=False)
            self._raise_on_incompatible(incompatible_keys, load_ckpt_class_head)

        self.log = torch.compiler.disable(self.log)  # type: ignore

    def configure_optimizers(self):
        encoder_param_names = {
            n for n, _ in self.network.encoder.backbone.named_parameters()
        }
        backbone_param_groups = []
        other_param_groups = []
        backbone_blocks = len(self.network.encoder.backbone.blocks)
        block_i = backbone_blocks

        l2_blocks = torch.arange(
            backbone_blocks - self.network.num_blocks, backbone_blocks
        ).tolist()

        for name, param in reversed(list(self.named_parameters())):
            lr = self.lr

            if name.replace("network.encoder.backbone.", "") in encoder_param_names:
                name_list = name.split(".")

                is_block = False
                for i, key in enumerate(name_list):
                    if key == "blocks":
                        block_i = int(name_list[i + 1])
                        is_block = True

                if is_block or block_i == 0:
                    lr *= self.llrd ** (backbone_blocks - 1 - block_i)
                    
                elif (is_block or block_i == 0) and self.lr_mult != 1.0:
                    lr *= self.lr_mult

                if "backbone.norm" in name:
                    lr = self.lr

                if is_block and (block_i in l2_blocks) and ((not self.llrd_l2_enabled) or (self.lr_mult != 1.0)):
                    lr = self.lr

                backbone_param_groups.append(
                    {"params": [param], "lr": lr, "name": name}
                )
            else:
                other_param_groups.append(
                    {"params": [param], "lr": self.lr, "name": name}
                )

        param_groups = backbone_param_groups + other_param_groups
        optimizer = AdamW(param_groups, weight_decay=self.weight_decay)

        scheduler = TwoStageWarmupPolySchedule(
            optimizer,
            num_backbone_params=len(backbone_param_groups),
            warmup_steps=self.warmup_steps,
            total_steps=self.trainer.estimated_stepping_batches,
            poly_power=self.poly_power,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def forward(self, imgs):
        return self.network(imgs)

    def training_step(self, batch, batch_idx):
        imgs, targets = batch

        mask_logits_per_block, class_logits_per_block = self(imgs)

        losses_all_blocks = {}
        for i, (mask_logits, class_logits) in enumerate(
            list(zip(mask_logits_per_block, class_logits_per_block))
        ):
            losses = self.criterion(
                masks_queries_logits=mask_logits,
                class_queries_logits=class_logits,
                targets=targets,
            )
            block_postfix = self.block_postfix(i)
            losses = {f"{key}{block_postfix}": value for key, value in losses.items()}
            losses_all_blocks |= losses

        return self.criterion.loss_total(losses_all_blocks, self.log)

    def validation_step(self, batch, batch_idx=0):
        return self.eval_step(batch, batch_idx, "val")

    def mask_annealing(self, start_iter, current_iter, final_iter):
        device = self.device
        dtype = self.network.attn_mask_probs[0].dtype
        if current_iter < start_iter:
            return torch.ones(1, device=device, dtype=dtype)
        elif current_iter >= final_iter:
            return torch.zeros(1, device=device, dtype=dtype)
        else:
            progress = (current_iter - start_iter) / (final_iter - start_iter)
            progress = torch.tensor(progress, device=device, dtype=dtype)
            return (1.0 - progress).pow(self.poly_power)

    def on_train_batch_end(
        self,
        outputs,
        batch,
        batch_idx=None,
        dataloader_idx=None,
    ):
        if self.attn_mask_annealing_enabled:
            for i in range(self.network.num_blocks):
                self.network.attn_mask_probs[i] = self.mask_annealing(
                    self.attn_mask_annealing_start_steps[i],
                    self.global_step,
                    self.attn_mask_annealing_end_steps[i],
                )

            for i, attn_mask_prob in enumerate(self.network.attn_mask_probs):
                self.log(
                    f"attn_mask_prob_{i}",
                    attn_mask_prob,
                    on_step=True,
                )

    def init_metrics_semantic(self, ignore_idx, num_blocks):
        self.metrics = nn.ModuleList(
            [
                MulticlassJaccardIndex(
                    num_classes=self.num_classes,
                    validate_args=False,
                    ignore_index=ignore_idx,
                    average=None,
                )
                for _ in range(num_blocks)
            ])
        
        # NEW: micro & weighted (FWIoU)
        self.metrics_micro = nn.ModuleList([
            MulticlassJaccardIndex(
                num_classes=self.num_classes,
                validate_args=False,
                ignore_index=ignore_idx,
                average="micro",
            ) for _ in range(num_blocks)
        ])
        self.metrics_weighted = nn.ModuleList([
            MulticlassJaccardIndex(
                num_classes=self.num_classes,
                validate_args=False,
                ignore_index=ignore_idx,
                average="weighted",
            ) for _ in range(num_blocks)
        ])

    @torch.compiler.disable
    def update_metrics_semantic(
        self,
        preds: list[torch.Tensor],
        targets: list[torch.Tensor],
        block_idx,
    ):
        for i in range(len(preds)):
            self.metrics[block_idx].update(preds[i][None, ...], targets[i][None, ...])
            # NEW: micro & weighted
            self.metrics_micro[block_idx].update(preds[i][None, ...], targets[i][None, ...])
            self.metrics_weighted[block_idx].update(preds[i][None, ...], targets[i][None, ...])

    

    def block_postfix(self, block_idx):
        if not self.network.masked_attn_enabled:
            return ""
        return (
            f"_block_{-len(self.metrics) + block_idx + 1}"
            if block_idx != self.network.num_blocks
            else ""
        )

    def _on_eval_epoch_end_semantic(self, log_prefix, log_per_class=True):
        for i, metric in enumerate(self.metrics):  # type: ignore
            iou_per_class = metric.compute()
            metric.reset()

            block_postfix = self.block_postfix(i)
            if log_per_class:
                for class_idx, iou in enumerate(iou_per_class):
                    self.log(
                        f"metrics/{log_prefix}_iou_class_{class_idx}{block_postfix}",
                        iou,
                    )

            iou_all = float(iou_per_class.mean())
            self.log(
                f"metrics/{log_prefix}_iou_all{block_postfix}",
                iou_all,
            )

            # NEW: micro IoU (global) & weighted IoU (FWIoU)
            iou_micro = float(self.metrics_micro[i].compute())
            self.metrics_micro[i].reset()
            self.log(f"metrics/{log_prefix}_iou_micro{block_postfix}", iou_micro)

            iou_weighted = float(self.metrics_weighted[i].compute())
            self.metrics_weighted[i].reset()
            self.log(f"metrics/{log_prefix}_iou_weighted{block_postfix}", iou_weighted)

    def _on_eval_end_semantic(self, log_prefix):
        if not self.trainer.sanity_checking:
            cb = self.trainer.callback_metrics
            # prefer FWIoU for imbalanced datasets
            try:
                fw = cb[f"metrics/{log_prefix}_iou_weighted"] * 100
                ma = cb[f"metrics/{log_prefix}_iou_all"] * 100
                mi = cb[f"metrics/{log_prefix}_iou_micro"] * 100
                rank_zero_info(
                    f"{bold_green}FWIoU: {fw:.1f} | mIoU: {ma:.1f} | microIoU: {mi:.1f}{reset}"
                )
            except KeyError:

                rank_zero_info(
                    f"{bold_green}mIoU: {self.trainer.callback_metrics[f'metrics/{log_prefix}_iou_all'] * 100:.1f}{reset}"
                )


    @torch.compiler.disable
    def plot_semantic(
        self,
        img,
        target,
        logits,
        log_prefix,
        block_idx,
        batch_idx,
        cmap="tab20",
    ):
        fig, axes = plt.subplots(1, 3, figsize=[15, 5], sharex=True, sharey=True)

        axes[0].imshow(img.cpu().numpy().transpose(1, 2, 0))
        axes[0].axis("off")

        target = target.cpu().numpy()
        unique_classes = np.unique(target)

        preds = torch.argmax(logits, dim=0).cpu().numpy()
        unique_classes = np.unique(np.concatenate((unique_classes, np.unique(preds))))

        num_classes = len(unique_classes)
        colors = plt.get_cmap(cmap, num_classes)(np.linspace(0, 1, num_classes))  # type: ignore

        if self.ignore_idx in unique_classes:
            colors[unique_classes == self.ignore_idx] = [0, 0, 0, 1]  # type: ignore

        custom_cmap = mcolors.ListedColormap(colors)  # type: ignore
        norm = mcolors.Normalize(0, num_classes - 1)

        axes[1].imshow(
            np.digitize(target, unique_classes) - 1,
            cmap=custom_cmap,
            norm=norm,
            interpolation="nearest",
        )
        axes[1].axis("off")

        if preds is not None:
            axes[2].imshow(
                np.digitize(preds, unique_classes, right=True),
                cmap=custom_cmap,
                norm=norm,
                interpolation="nearest",
            )
            axes[2].axis("off")

        patches = [
            Line2D([0], [0], color=colors[i], lw=4, label=str(unique_classes[i]))
            for i in range(num_classes)
        ]

        fig.legend(handles=patches, loc="upper left")

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, facecolor="black")
        plt.close(fig)
        buf.seek(0)

        block_postfix = self.block_postfix(block_idx)
        name = f"{log_prefix}_pred_{batch_idx}{block_postfix}"
        self.trainer.logger.experiment.log({name: [wandb.Image(Image.open(buf))]})

    @torch.compiler.disable
    def scale_img_size_semantic(self, size: tuple[int, int]):
        factor = max(
            self.img_size[0] / size[0],
            self.img_size[1] / size[1],
        )

        return [round(s * factor) for s in size]

    @torch.compiler.disable
    def window_imgs_semantic(self, imgs):
        crops, origins = [], []

        for i in range(len(imgs)):
            img = imgs[i]
            new_h, new_w = self.scale_img_size_semantic(img.shape[-2:])
            img_np = img.permute(1, 2, 0).detach().cpu().numpy()
            # ensure uint8 for PIL
            if img_np.dtype != "uint8":
                # assume [0,1] float or arbitrary float; clamp + scale
                img_np = (img_np.clip(0.0, 1.0) * 255.0).astype("uint8")
            pil_img = Image.fromarray(img_np)
            resized_img = pil_img.resize((new_w, new_h), Image.BILINEAR)
            resized_img = (
                torch.from_numpy(np.array(resized_img)).permute(2, 0, 1).to(img.device)
            )
            resized_img = resized_img.float() / 255.0
            num_crops = math.ceil(max(resized_img.shape[-2:]) / min(self.img_size))
            overlap = num_crops * min(self.img_size) - max(resized_img.shape[-2:])
            overlap_per_crop = (overlap / (num_crops - 1)) if overlap > 0 else 0

            for j in range(num_crops):
                start = int(j * (min(self.img_size) - overlap_per_crop))
                end = start + min(self.img_size)
                if resized_img.shape[-2] > resized_img.shape[-1]:
                    crop = resized_img[:, start:end, :]
                else:
                    crop = resized_img[:, :, start:end]

                crops.append(crop)
                origins.append((i, start, end))

        return torch.stack(crops), origins

    def revert_window_logits_semantic(self, crop_logits, origins, img_sizes):
        logit_sums, logit_counts = [], []
        for size in img_sizes:
            h, w = self.scale_img_size_semantic(size)
            logit_sums.append(
                torch.zeros((crop_logits.shape[1], h, w), device=crop_logits.device)
            )
            logit_counts.append(
                torch.zeros((crop_logits.shape[1], h, w), device=crop_logits.device)
            )

        for crop_i, (img_i, start, end) in enumerate(origins):
            if img_sizes[img_i][0] > img_sizes[img_i][1]:
                logit_sums[img_i][:, start:end, :] += crop_logits[crop_i]
                logit_counts[img_i][:, start:end, :] += 1
            else:
                logit_sums[img_i][:, :, start:end] += crop_logits[crop_i]
                logit_counts[img_i][:, :, start:end] += 1

        return [
            interpolate(
                (sums / counts)[None, ...],
                img_sizes[i],
                mode="bilinear",
            )[0]
            for i, (sums, counts) in enumerate(zip(logit_sums, logit_counts))
        ]

    @staticmethod
    def to_per_pixel_logits_semantic(
        mask_logits: torch.Tensor, class_logits: torch.Tensor
    ):
        return torch.einsum(
            "bqhw, bqc -> bchw",
            mask_logits.sigmoid(),
            class_logits.softmax(dim=-1)[..., :-1],
        )

    @staticmethod
    @torch.compiler.disable
    def to_per_pixel_targets_semantic(
        targets: list[dict],
        ignore_idx,
    ):
        per_pixel_targets = []
        for target in targets:
            per_pixel_target = torch.full(
                target["masks"].shape[-2:],
                ignore_idx,
                dtype=target["labels"].dtype,
                device=target["labels"].device,
            )

            for i, mask in enumerate(target["masks"]):
                per_pixel_target[mask] = target["labels"][i]

            per_pixel_targets.append(per_pixel_target)

        return per_pixel_targets

    def on_save_checkpoint(self, checkpoint):
        checkpoint["state_dict"] = {
            k.replace("._orig_mod", ""): v for k, v in checkpoint["state_dict"].items()
        }

    def _zero_init_outside_encoder(self, encoder_prefix="network.encoder."):
        with torch.no_grad():
            total, zeroed = 0, 0
            for name, p in self.named_parameters():
                total += p.numel()
                if not name.startswith(encoder_prefix):
                    p.zero_()
                    zeroed += p.numel()
            logging.info(
                f"Zeroed {zeroed:,} / {total:,} parameters (everything not under '{encoder_prefix}')"
            )

    def _add_state_dicts(self, state_dict1, state_dict2):
        summed = {}
        for k in state_dict1.keys():
            if k not in state_dict2:
                raise KeyError(f"Key {k} not found in second state_dict")

            if state_dict1[k].shape != state_dict2[k].shape:
                raise ValueError(
                    f"Shape mismatch at {k}: "
                    f"{state_dict1[k].shape} vs {state_dict2[k].shape}"
                )

            summed[k] = state_dict1[k] + state_dict2[k]

        return summed

    def _load_ckpt(self, ckpt_path, load_ckpt_class_head):
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        ckpt = {k: v for k, v in ckpt.items() if "criterion.empty_weight" not in k}
        if not load_ckpt_class_head:
            ckpt = {
                k: v
                for k, v in ckpt.items()
                if "class_head" not in k and "class_predictor" not in k
            }
        logging.info(f"Loaded {len(ckpt)} keys")
        return ckpt

    def _raise_on_incompatible(self, incompatible_keys, load_ckpt_class_head):
        if incompatible_keys.missing_keys:
            if not load_ckpt_class_head:
                missing_keys = [
                    key
                    for key in incompatible_keys.missing_keys
                    if "class_head" not in key and "class_predictor" not in key
                ]
            else:
                missing_keys = incompatible_keys.missing_keys
            if missing_keys:
                raise ValueError(f"Missing keys: {missing_keys}")
        if incompatible_keys.unexpected_keys:
            raise ValueError(f"Unexpected keys: {incompatible_keys.unexpected_keys}")