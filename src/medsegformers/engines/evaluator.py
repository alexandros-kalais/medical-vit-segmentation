import numpy as np
import torch
from typing import Dict, Tuple, Optional
from monai.data import decollate_batch
from monai.transforms import Compose, Activations, AsDiscrete
from monai.metrics import DiceMetric, MeanIoU, HausdorffDistanceMetric
from medsegformers.utils.vis import ENDOSCOPY_CLASS_NAMES

import copy, time
import numpy as np
import torch
from fvcore.nn import FlopCountAnalysis

def _count_params_total(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def _compute_flops(model: torch.nn.Module, input_shape) -> int:

    model_cpu = copy.deepcopy(model).cpu().eval()
    x_cpu = torch.randn(*input_shape)  # CPU tensor
    flops = FlopCountAnalysis(model_cpu, x_cpu).total()
    return int(flops)

def _measure_fps_gpu(
    model: torch.nn.Module,
    input_shape,
    device: torch.device,
    warmup: int = 10,
    runs: int = 50,
    amp_dtype: torch.dtype | None = torch.bfloat16,  # good default for A100/H100
) -> float:

    assert device.type == "cuda", "GPU timing requires CUDA device"
    model.eval().to(device)
    x = torch.randn(*input_shape, device=device)

    torch.backends.cudnn.benchmark = True

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=amp_dtype) if amp_dtype else torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
    torch.cuda.synchronize()

    times = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event   = torch.cuda.Event(enable_timing=True)

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=amp_dtype) if amp_dtype else torch.no_grad():
        for _ in range(runs):
            start_event.record()
            _ = model(x)
            end_event.record()
            torch.cuda.synchronize()
            ms = start_event.elapsed_time(end_event)
            times.append(ms / 1000.0)

    med = float(np.median(times))
    return 1.0 / med if med > 0 else float("nan")


class Evaluator:
    def __init__(self, model: torch.nn.Module, num_classes: int, device: torch.device,
                 compute_hd95: bool = True, include_background: bool = False):
        self.model = model.to(device)
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

        red = "none"
        self.dice_metric = DiceMetric(include_background=include_background, reduction=red)
        self.miou_metric = MeanIoU   (include_background=include_background, reduction=red)
        self.compute_hd95 = compute_hd95
        self.hd95_metric = None
        if compute_hd95:
            self.hd95_metric = HausdorffDistanceMetric(
                include_background=include_background,
                percentile=95.0, directed=False, reduction=red
            )

    def efficiency(
        self,
        *,
        input_shape: tuple,  # (1, 3, H, W)
        warmup: int = 10,
        runs: int = 50,
        amp_dtype: torch.dtype | None = torch.bfloat16,
        use_gpu: bool = True,
    ):

        total_params = sum(p.numel() for p in self.model.parameters())

        flops = _compute_flops(self.model, input_shape)
        gflops = float(flops) / 1e9 if flops is not None else None

        if use_gpu and self.device.type == "cuda":
            fps = _measure_fps_gpu(
                self.model, input_shape, self.device, warmup=warmup, runs=runs, amp_dtype=amp_dtype
            )
        else:
            fps = float("nan")

        return {
            "total_params": int(total_params),
            "gflops": gflops,
            "fps": float(fps),
        }

    @torch.no_grad()
    def run(self, loader, *, dataset_name: str) -> Dict:
        self.model.eval()
        for batch in loader:
            images, labels = batch["image"].to(self.device), batch["label"].to(self.device)
            logits = self.model(images)

            if self.num_classes == 1:
                dice_preds  = [self.post_pred_dice(x) for x in decollate_batch(logits)]
                dice_labels = decollate_batch(labels)
            else:
                dice_preds  = [self.post_pred_dice(x) for x in decollate_batch(logits)]
                dice_labels = [self.post_label(x)     for x in decollate_batch(labels)]

            self.dice_metric(y_pred=dice_preds, y=dice_labels)

            if self.num_classes == 1:
                iou_preds  = [self.post_pred_iou(x)  for x in decollate_batch(logits)]
                iou_labels = [self.post_label_iou(x) for x in decollate_batch(labels)]
            else:
                iou_preds, iou_labels = dice_preds, dice_labels

            self.miou_metric(y_pred=iou_preds, y=iou_labels)
            if self.hd95_metric:
                self.hd95_metric(y_pred=iou_preds, y=iou_labels)

        dice_raw = self.dice_metric.aggregate().cpu().numpy()
        miou_raw = self.miou_metric.aggregate().cpu().numpy()
        hd95_raw = self.hd95_metric.aggregate().cpu().numpy() if self.hd95_metric else None

        self.dice_metric.reset()
        self.miou_metric.reset()
        if self.hd95_metric:
            self.hd95_metric.reset()

        dice_cls = np.nanmean(dice_raw, axis=0) if dice_raw.size else np.array([])
        miou_cls = np.nanmean(miou_raw, axis=0) if miou_raw.size else np.array([])
        hd95_cls = np.nanmean(hd95_raw, axis=0) if (hd95_raw is not None and hd95_raw.size) else np.array([])

        if self.num_classes == 1:
            class_names = ["foreground"]
        else:
            if self.include_background == False:
                class_names = ENDOSCOPY_CLASS_NAMES[1:]

        result = {
            "class_names": class_names,
            "dice_per_class": dice_cls.tolist() if dice_cls.size else [],
            "miou_per_class": miou_cls.tolist() if miou_cls.size else [],
            "hd95_per_class": hd95_cls.tolist() if hd95_cls.size else [],
            "mean_dice": float(np.nanmean(dice_cls)) if dice_cls.size else float("nan"),
            "mean_miou": float(np.nanmean(miou_cls)) if miou_cls.size else float("nan"),
            "mean_hd95": float(np.nanmean(hd95_cls)) if hd95_cls.size else float("nan"),
        }
        return result
