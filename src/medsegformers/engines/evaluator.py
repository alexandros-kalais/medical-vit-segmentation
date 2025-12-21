import numpy as np
import torch
from typing import Dict, Tuple, Optional
from torchmetrics.classification import MulticlassJaccardIndex
from monai.data import decollate_batch
from monai.metrics import HausdorffDistanceMetric
from medsegformers.utils import ENDOSCOPY_CLASS_NAMES
import copy, time
import numpy as np
import torch
from fvcore.nn import FlopCountAnalysis


IGNORE_INDEX = 255

def _count_params_total(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def _compute_flops(model: torch.nn.Module, input_shape) -> int:

    dev = next(model.parameters()).device
    mdl = copy.deepcopy(model).to(dev).eval()
    x = torch.randn(*input_shape, device=dev)

    with torch.autocast(device_type=dev.type, enabled=False):
        flops = FlopCountAnalysis(mdl, x).total()
    return int(flops)

def _measure_fps_gpu(
    model: torch.nn.Module,
    input_shape,
    device: torch.device,
    warmup: int = 10,
    runs: int = 50,
    amp_dtype: torch.dtype | None = torch.bfloat16,
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
        input_shape: tuple,
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

        hd95_class_sums = None
        hd95_class_counts = None

        for batch in loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)         

            logits = self.model(images)                     
            preds  = logits.argmax(dim=1)                    
            target = labels.squeeze(1).long()                

            self.iou_per_class.update(preds, target)

            if self.hd95_metric is not None:
                t_tmp = target.clone()
                t_tmp[target == IGNORE_INDEX] = 0

                p_tmp = preds.clone()
                p_tmp[p_tmp == IGNORE_INDEX] = 0

                p_oh = torch.nn.functional.one_hot(p_tmp, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
                t_oh = torch.nn.functional.one_hot(t_tmp, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

                ignore = (target == IGNORE_INDEX)
                p_oh[ignore.unsqueeze(1).expand_as(p_oh)] = 0.0
                t_oh[ignore.unsqueeze(1).expand_as(t_oh)] = 0.0

                p_list = list(torch.unbind(p_oh, dim=0))
                t_list = list(torch.unbind(t_oh, dim=0))

                hd = self.hd95_metric(y_pred=p_list, y=t_list)
                hd = torch.stack(hd) if isinstance(hd, list) else hd  # [B, C]

                tgt_present = (t_oh.sum(dim=(2, 3)) > 0)              # [B, C]
                hd = hd.masked_fill(~tgt_present, float("nan"))        # [B, C]

                if hd95_class_sums is None:
                    hd95_class_sums = torch.zeros(self.num_classes, device=hd.device)
                    hd95_class_counts = torch.zeros(self.num_classes, device=hd.device, dtype=torch.long)

                hd95_class_sums   += torch.nan_to_num(hd, nan=0.0).sum(dim=0)      # [C]
                hd95_class_counts += torch.isfinite(hd).sum(dim=0)                  # [C]


        miou_per_class = self.iou_per_class.compute().cpu().numpy()
        self.iou_per_class.reset()
        miou_macro = float(np.nanmean(miou_per_class))   

        if self.hd95_metric is not None and hd95_class_sums is not None:

            hd95_per_class = hd95_class_sums / hd95_class_counts.clamp_min(1)
            hd95_per_class[hd95_class_counts == 0] = float("nan")
            mean_hd95 = float(torch.nanmean(hd95_per_class))
            hd95_per_class = hd95_per_class.cpu().numpy()
        else:
            hd95_per_class = [float("nan")] * self.num_classes
            mean_hd95 = float("nan")


        result = {
            "miou_per_class": miou_per_class.tolist(),
            "mean_miou": miou_macro,
            "hd95_per_class": (hd95_per_class.tolist() if isinstance(hd95_per_class, np.ndarray) else hd95_per_class),
            "mean_hd95": mean_hd95,
        }
        return result

