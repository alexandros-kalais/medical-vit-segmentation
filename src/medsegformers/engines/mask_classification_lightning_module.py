import math
import io
import logging
from typing import List, Optional
import lightning
from lightning.fabric.utilities import rank_zero_info
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torchmetrics.classification import MulticlassJaccardIndex
import wandb
from PIL import Image
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.functional import interpolate

from medsegformers.losses.mask_classification_loss import MaskClassificationLoss
from .two_stage_warmup_poly_schedule import TwoStageWarmupPolySchedule

bold_green = "\033[1;32m"
reset = "\033[0m"


class MaskClassificationSemantic(lightning.LightningModule):
    def __init__(
        self,
        network: nn.Module,
        img_size: tuple[int, int],
        num_classes: int,
        attn_mask_annealing_enabled: bool,
        attn_mask_annealing_start_steps: Optional[list[int]] = None,
        attn_mask_annealing_end_steps: Optional[list[int]] = None,
        ignore_idx: int = 255,
        lr: float = 1e-4,
        llrd: float = 0.8,
        llrd_l2_enabled: bool = True,
        lr_mult: float = 1.0,
        weight_decay: float = 0.05,
        num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
        poly_power: float = 0.9,
        warmup_steps: List[int] = [500, 1000],
        no_object_coefficient: float = 0.1,
        mask_coefficient: float = 5.0,
        dice_coefficient: float = 5.0,
        class_coefficient: float = 2.0,
        mask_thresh: float = 0.8,
        overlap_thresh: float = 0.8,
        ckpt_path: Optional[str] = None,
        delta_weights: bool = False,
        load_ckpt_class_head: bool = True,
    ):
        super().__init__()

        self.network = network
        self.img_size = img_size
        self.num_classes = num_classes
        self.attn_mask_annealing_enabled = attn_mask_annealing_enabled
        self.attn_mask_annealing_start_steps = attn_mask_annealing_start_steps
        self.attn_mask_annealing_end_steps = attn_mask_annealing_end_steps
        self.ignore_idx = ignore_idx
        self.mask_thresh = mask_thresh
        self.overlap_thresh = overlap_thresh
        self.stuff_classes = range(num_classes)
        self.lr = lr
        self.llrd = llrd
        self.llrd_l2_enabled = llrd_l2_enabled
        self.lr_mult = lr_mult
        self.weight_decay = weight_decay
        self.poly_power = poly_power
        self.warmup_steps = warmup_steps

        self.save_hyperparameters(ignore=["_class_path"])

        self.strict_loading = False

        self.criterion = MaskClassificationLoss(
            num_points=num_points,
            oversample_ratio=oversample_ratio,
            importance_sample_ratio=importance_sample_ratio,
            mask_coefficient=mask_coefficient,
            dice_coefficient=dice_coefficient,
            class_coefficient=class_coefficient,
            num_labels=num_classes,
            no_object_coefficient=no_object_coefficient,
        )

        num_blocks = self.network.num_blocks + 1 if self.network.masked_attn_enabled else 1
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

        self.log = torch.compiler.disable(self.log)

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
            block_postfix = self._block_postfix(i)
            losses = {f"{key}{block_postfix}": value for key, value in losses.items()}
            losses_all_blocks |= losses

        return self.criterion.loss_total(losses_all_blocks, self.log)

    def validation_step(self, batch, batch_idx=0):
        imgs, targets = batch

        img_sizes = [img.shape[-2:] for img in imgs]
        crops, origins = self._window_imgs(imgs)
        mask_logits_per_layer, class_logits_per_layer = self(crops)

        targets = self._to_per_pixel_targets(targets)

        for i, (mask_logits, class_logits) in enumerate(
            list(zip(mask_logits_per_layer, class_logits_per_layer))
        ):
            mask_logits = F.interpolate(mask_logits, self.img_size, mode="bilinear")
            crop_logits = self._to_per_pixel_logits(mask_logits, class_logits)
            logits = self._revert_window_logits(crop_logits, origins, img_sizes)

            self._update_metrics(logits, targets, i)

            if batch_idx == 0:
                self._plot_predictions(
                    imgs[0], targets[0], logits[0], "val", i, batch_idx
                )

    def on_validation_epoch_end(self):
        for i, metric in enumerate(self.metrics):
            iou_per_class = metric.compute()
            metric.reset()

            block_postfix = self._block_postfix(i)
            for class_idx, iou in enumerate(iou_per_class):
                self.log(
                    f"metrics/val_iou_class_{class_idx}{block_postfix}",
                    iou,
                )

            iou_all = float(iou_per_class.mean())
            self.log(
                f"metrics/val_iou_all{block_postfix}",
                iou_all,
            )
        self.log("val_miou", iou_all, on_epoch=True, prog_bar=False, sync_dist=True)

    def on_validation_end(self):
        if not self.trainer.sanity_checking:
            cb = self.trainer.callback_metrics
            val = cb[f"metrics/val_iou_all"] * 100
            rank_zero_info(f"{bold_green}mIoU: {val:.1f}{reset}")

    def on_train_batch_end(
        self,
        outputs,
        batch,
        batch_idx=None,
        dataloader_idx=None,
    ):
        if self.attn_mask_annealing_enabled:
            for i in range(self.network.num_blocks):
                self.network.attn_mask_probs[i] = self._mask_annealing(
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

    def on_save_checkpoint(self, checkpoint):
        checkpoint["state_dict"] = {
            k.replace("._orig_mod", ""): v for k, v in checkpoint["state_dict"].items()
        }

    def _mask_annealing(self, start_iter, current_iter, final_iter):
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

    @torch.compiler.disable
    def _update_metrics(
        self,
        preds: list[torch.Tensor],
        targets: list[torch.Tensor],
        block_idx,
    ):
        for i in range(len(preds)):
            self.metrics[block_idx].update(preds[i][None, ...], targets[i][None, ...])

    def _block_postfix(self, block_idx):
        if not self.network.masked_attn_enabled:
            return ""
        return (
            f"_block_{-len(self.metrics) + block_idx + 1}"
            if block_idx != self.network.num_blocks
            else ""
        )

    @torch.compiler.disable
    def _plot_predictions(
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

        target_np = target.detach().cpu().numpy().astype(np.int64)
        preds_np = torch.argmax(logits, dim=0).detach().cpu().numpy().astype(np.int64)

        ignore_mask = (target_np == self.ignore_idx)
        preds_vis = preds_np.copy()
        preds_vis[ignore_mask] = self.ignore_idx

        unique_classes = np.unique(
            np.concatenate((np.unique(target_np), np.unique(preds_vis)))
        )

        num_classes = len(unique_classes)
        colors = plt.get_cmap(cmap, num_classes)(np.linspace(0, 1, num_classes))

        if (self.ignore_idx in unique_classes):
            colors[unique_classes == self.ignore_idx] = [0, 0, 0, 1]

        custom_cmap = mcolors.ListedColormap(colors)
        norm = mcolors.Normalize(0, num_classes - 1)

        target_mapped = np.digitize(target_np, unique_classes) - 1
        preds_mapped = np.digitize(preds_vis, unique_classes) - 1

        axes[1].imshow(target_mapped, cmap=custom_cmap, norm=norm, interpolation="nearest")
        axes[1].axis("off")

        axes[2].imshow(preds_mapped, cmap=custom_cmap, norm=norm, interpolation="nearest")
        axes[2].axis("off")

        labels = [("ignore" if cls == self.ignore_idx else str(cls)) for cls in unique_classes]
        patches = [Line2D([0], [0], color=colors[i], lw=4, label=labels[i]) for i in range(num_classes)]
        fig.legend(handles=patches, loc="upper left")

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, facecolor="black")
        plt.close(fig)
        buf.seek(0)

        block_postfix = self._block_postfix(block_idx)
        name = f"{log_prefix}_pred_{batch_idx}{block_postfix}"
        self.trainer.logger.experiment.log({name: [wandb.Image(Image.open(buf))]})

    @torch.compiler.disable
    def _scale_img_size(self, size: tuple[int, int]):
        factor = max(
            self.img_size[0] / size[0],
            self.img_size[1] / size[1],
        )
        return [round(s * factor) for s in size]

    @torch.compiler.disable
    def _window_imgs(self, imgs):
        crops, origins = [], []

        for i in range(len(imgs)):
            img = imgs[i]
            new_h, new_w = self._scale_img_size(img.shape[-2:])
            img_np = img.permute(1, 2, 0).detach().cpu().numpy()
            if img_np.dtype != "uint8":
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

    def _revert_window_logits(self, crop_logits, origins, img_sizes):
        logit_sums, logit_counts = [], []
        for size in img_sizes:
            h, w = self._scale_img_size(size)
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

    def _to_per_pixel_logits(self, 
        mask_logits: torch.Tensor, class_logits: torch.Tensor
    ):
        per_pixel_fg = torch.einsum(
            "bqhw, bqc -> bchw",
            mask_logits.sigmoid(),
            class_logits.softmax(dim=-1)[..., :-1],
        )

        if self.num_classes == 2:
            p_bg = (1.0 - per_pixel_fg.sum(dim=1, keepdim=True)).clamp(0.0, 1.0)
            per_pixel = torch.cat([p_bg, per_pixel_fg], dim=1)
        else:
            per_pixel = per_pixel_fg
        return per_pixel

    @torch.compiler.disable
    def _to_per_pixel_targets(self, targets: list[dict]):
        per_pixel_targets = []
        for target in targets:
            if self.num_classes == 2:
                per_pixel_target = torch.zeros(
                    target["masks"].shape[-2:],
                    dtype=target["labels"].dtype,
                    device=target["labels"].device,
                )
                for mask in target["masks"]:
                    per_pixel_target[mask] = 1
            else:
                per_pixel_target = torch.full(
                    target["masks"].shape[-2:],
                    self.ignore_idx,
                    dtype=target["labels"].dtype,
                    device=target["labels"].device,
                )
                for i, mask in enumerate(target["masks"]):
                    per_pixel_target[mask] = target["labels"][i]

            per_pixel_targets.append(per_pixel_target)

        return per_pixel_targets

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