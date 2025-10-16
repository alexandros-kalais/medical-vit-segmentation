import argparse, json
from pathlib import Path
import torch
from typing import Optional
from monai.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

from medsegformers.models.vit import ViT
from medsegformers.models.eomt import EoMT
from medsegformers.transforms import get_transforms
from medsegformers.utils.paths import get_experiments_root, get_data_root
from medsegformers.data import get_dataset_class
from medsegformers.engines.evaluator import Evaluator  

def _read_runs_txt(p: Path) -> list[str]:
    with p.open("r") as f:
        return [ln.strip() for ln in f.readlines() if ln.strip()]

def _exp_dir_from_id(exp_id: str, dataset: str) -> Path:
    return get_experiments_root() / dataset / exp_id

def _pick_ckpt(ckpt_dir: Path) -> Optional[Path]:

    if not ckpt_dir.is_dir():
        return None

    non_last = [p for p in ckpt_dir.glob("*.ckpt") if p.name != "last.ckpt"]

    return non_last[0]

def _load_network_weights_from_lightning_ckpt(model: torch.nn.Module, ckpt_path: Path, device: torch.device):
    # If you trust the checkpoint source, just disable the safety gate:
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    sd = ckpt.get("state_dict", {})
    net_sd = {k[len("network."):]: v for k, v in sd.items() if k.startswith("network.")}
    missing, unexpected = model.load_state_dict(net_sd, strict=False)
    if missing or unexpected:
        print(f"[WARN] load_state_dict mismatches -> missing: {missing}, unexpected: {unexpected}")

class EoMTSemanticWrapper(nn.Module):
    """Wraps EoMT to output per-pixel class logits (B,C,H,W) from the FINAL block only."""
    def __init__(self, eomt: EoMT, img_size: tuple[int, int]):
        super().__init__()
        self.eomt = eomt
        self.img_size = img_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # EoMT returns lists per block: ([mask_logits_l], [class_logits_l])  (last item is final block)
        mask_logits_list, class_logits_list = self.eomt(x)  # lists length = num_blocks (+1 if masked attn enabled) :contentReference[oaicite:5]{index=5}
        mask_logits = mask_logits_list[-1]                  # (B, Q, h, w)
        class_logits = class_logits_list[-1]                # (B, Q, C+1)  (+1 = "no object") :contentReference[oaicite:6]{index=6}

        # Match Lightning eval behavior: upsample mask logits to the training image size before combining. :contentReference[oaicite:7]{index=7}
        mask_logits = F.interpolate(mask_logits, size=self.img_size, mode="bilinear")

        # Convert to per-pixel class logits (drop the "no object" column). :contentReference[oaicite:8]{index=8}
        per_pixel = torch.einsum(
            "bqhw,bqc->bchw",
            mask_logits.sigmoid(),
            class_logits.softmax(dim=-1)[..., :-1],
        )
        return per_pixel



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="e.g. endoscopy")
    ap.add_argument("--experiments_file", required=True, help="Text file with one experiment ID per line")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp_dtype", default="bf16", choices=["bf16", "fp16", "none"], help="for FPS timing on GPU")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--runs", type=int, default=50)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}

    # read ids from txt
    exp_ids = _read_runs_txt(Path(args.experiments_file))
    if not exp_ids:
        print("Empty experiments file.")
        return


    for exp_id in exp_ids:
        exp_dir = _exp_dir_from_id(exp_id, args.dataset)
        ckpt_dir = exp_dir / "checkpoints"
        ckpt = _pick_ckpt(ckpt_dir)
        cfg_path = ckpt_dir / "run_config.json"

        if ckpt is None or not cfg_path.exists():
            print(f"[SKIP] Missing ckpt or run_config.json in: {exp_dir}")
            continue

        with open(cfg_path, "r") as f:
            cfg = json.load(f)

        image_size = tuple(cfg["image_size"])
        vit_name   = cfg["vit_name"] 
        seed       = cfg.get("seed", 42)
        eomt_num_blocks = cfg.get("eomt_num_blocks")
        eomt_num_q = cfg.get("eomt_num_q")

        DatasetCls = get_dataset_class(args.dataset)
        root = DatasetCls.default_root(get_data_root())

        tf = get_transforms(dataset=args.dataset, kind="basic", image_size=image_size)
        ds = DatasetCls(split="test",transform=tf, root=root, return_masks=False)

        num_classes = getattr(DatasetCls, "NUM_CLASSES", None)

        loader =  DataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers)

        # 4. Model and Module 

        H, W = image_size
        if any(x in vit_name.lower() for x in ["16-", "16_"]):
            patch_size = 16
        elif any(x in vit_name.lower() for x in ["14_"]):
            patch_size = 14
        else:
            raise ValueError("Define patch_size correctly!")

        encoder = ViT(
            img_size=(H, W),
            patch_size=patch_size,
            backbone_name=vit_name,
            ckpt_path=getattr(args, "vit_ckpt", None),
        )

        network = EoMT(
            encoder=encoder,
            num_classes=num_classes,
            num_q=eomt_num_q,
            num_blocks=eomt_num_blocks
    )

        _load_network_weights_from_lightning_ckpt(network, ckpt, device)
        
        model = EoMTSemanticWrapper(network, img_size=(H, W)).to(device).eval()

        # evaluate seg + efficiency
        evaluator = Evaluator(
            model, num_classes=num_classes, device=device, include_background=False
        )
        seg = evaluator.run(loader, dataset_name=args.dataset)
        eff = evaluator.efficiency(
            input_shape=(1, 3, image_size[0], image_size[1]),
            warmup=args.warmup,
            runs=args.runs,
            amp_dtype=amp_map[args.amp_dtype],
            use_gpu=(device.type == "cuda"),
        )

        # write JSON inside the experiment folder
        out = {
            "experiment_id": exp_dir.name,
            "dataset": args.dataset,
            "vit_name": vit_name,
            "image_size": list(image_size),
            "mean_dice": seg["mean_dice"],
            "mean_miou": seg["mean_miou"],
            "mean_hd95": seg["mean_hd95"],
            "class_names": seg["class_names"],
            "dice_per_class": seg["dice_per_class"],
            "miou_per_class": seg["miou_per_class"],
            "hd95_per_class": seg["hd95_per_class"],
            "total_params": eff["total_params"],
            "gflops": eff["gflops"],
            "fps": eff["fps"],
            "ckpt": str(ckpt),
        }
        out_json = exp_dir / f"eval.json"
        with open(out_json, "w") as f:
            json.dump(out, f, indent=2)

        print(f"[OK] {exp_dir.name}  mDice={out['mean_dice']:.4f}  mIoU={out['mean_miou']:.4f} mHD95={out['mean_hd95']:.4f}  GFLOPs={out['gflops']:.2f}  FPS={out['fps']:.1f} params={out['total_params']}")

if __name__ == "__main__":
    main()
