import numpy as np
import torch
from typing import Dict, Tuple, Optional
from torchmetrics.classification import MulticlassJaccardIndex
from monai.data import decollate_batch
from monai.metrics import HausdorffDistanceMetric
from medsegformers.utils.vis import ENDOSCOPY_CLASS_NAMES

import copy, time
import numpy as np
import torch
from fvcore.nn import FlopCountAnalysis


IGNORE_INDEX = 255

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

        self.iou_per_class = MulticlassJaccardIndex(
            num_classes=num_classes,
            ignore_index=IGNORE_INDEX,
            average=None,
            validate_args=False
        ).to(self.device)

        self.compute_hd95 = compute_hd95
        self.hd95_metric = None
        if compute_hd95:
            self.hd95_metric = HausdorffDistanceMetric(
                include_background=include_background,
                percentile=95.0, directed=False, reduction="none"
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

        hd95_class_sums = None  # accumulate per-class sums
        hd95_class_counts = None

        for batch in loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)          # (B,1,H,W) with {0..C-1, 255}

            logits = self.model(images)                      # (B,C,H,W)
            preds  = logits.argmax(dim=1)                    # (B,H,W) in {0..C-1}
            target = labels.squeeze(1).long()                # (B,H,W) in {0..C-1, 255}


            # IoU (torchmetrics) — ignores 255 automatically
            self.iou_per_class.update(preds, target)

            # HD95 (optional): build one-hot WITH ignored pixels suppressed
            if self.hd95_metric is not None:
                # Map ignored labels to class 0 temporarily (to keep shapes), but then zero-out both pred & gt at those positions
                t_tmp = target.clone()
                t_tmp[target == IGNORE_INDEX] = 0

                # discrete preds for HD: if we confidence-masked to 255, also send them to 0 temp, then zero-out with mask
                p_tmp = preds.clone()
                p_tmp[p_tmp == IGNORE_INDEX] = 0

                # one-hot
                # Shapes: list of (1,C,H,W) expected by monai metrics if using decollate
                # We’ll build tensors and decollate explicitly.
                p_oh = torch.nn.functional.one_hot(p_tmp, num_classes=self.num_classes)  \
                        .permute(0,3,1,2).float()
                t_oh = torch.nn.functional.one_hot(t_tmp, num_classes=self.num_classes)  \
                        .permute(0,3,1,2).float()

                # Zero-out ignored pixels in both p_oh and t_oh
                ignore = (target == IGNORE_INDEX)            # (B,H,W)
                p_oh[ignore.unsqueeze(1).expand_as(p_oh)] = 0.0
                t_oh[ignore.unsqueeze(1).expand_as(t_oh)] = 0.0

                # decollate to lists for MONAI hd95
                p_list = list(torch.unbind(p_oh, dim=0))
                t_list = list(torch.unbind(t_oh, dim=0))

                # compute hd95 per sample (returns tensor [B, C])
                hd = self.hd95_metric(y_pred=p_list, y=t_list)  # type: ignore
                # accumulate across batch
                hd = torch.stack(hd) if isinstance(hd, list) else hd  # (B,C)
                if hd95_class_sums is None:
                    hd95_class_sums = torch.zeros(self.num_classes, device=hd.device)
                    hd95_class_counts = torch.zeros(self.num_classes, device=hd.device)
                # Only count classes that appear (non-zero gt or pred) to avoid inflating with zeros
                present = (t_oh.sum(dim=(0,2,3)) > 0)  # which classes present in gt across batch
                hd95_class_sums[present] += hd[:, present].mean(dim=0)
                hd95_class_counts[present] += 1

        # Final IoU
        miou_per_class = self.iou_per_class.compute().cpu().numpy()
        self.iou_per_class.reset()
        miou_macro = float(np.nanmean(miou_per_class))   # macro mIoU over 6 classes

        # HD95 aggregate
        if self.hd95_metric is not None and hd95_class_sums is not None:
            eps = 1e-9
            hd95_per_class = (hd95_class_sums / (hd95_class_counts + eps)).cpu().numpy()
            mean_hd95 = float(np.nanmean(hd95_per_class))
        else:
            hd95_per_class = [float("nan")] * self.num_classes
            mean_hd95 = float("nan")

        # We removed Dice as a tracked metric; if you still want it, compute similarly on masked one-hots.
        result = {
            "miou_per_class": miou_per_class.tolist(),
            "mean_miou": miou_macro,
            "hd95_per_class": (hd95_per_class.tolist() if isinstance(hd95_per_class, np.ndarray) else hd95_per_class),
            "mean_hd95": mean_hd95,
        }
        return result

