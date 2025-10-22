import argparse
import json
from pathlib import Path
import torch
from typing import Optional
from monai.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

from medsegformers.models import ViT
from medsegformers.dataset import get_dataset_class, get_transforms
from medsegformers.utils import get_experiments_root, get_data_root, _read_runs_txt, _exp_dir_from_id, _pick_ckpt
from medsegformers.engines import Evaluator
from . import infer_patch_size, build_model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="e.g. endoscopy")
    ap.add_argument("--experiments_file", required=True, help="Text file with one experiment ID per line")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp_dtype", default="bf16", choices=["bf16", "fp16", "none"], 
                    help="AMP dtype for FPS timing on GPU")
    ap.add_argument("--warmup", type=int, default=10, help="Warmup iterations for FPS measurement")
    ap.add_argument("--runs", type=int, default=50, help="Number of runs for FPS measurement")
    args = ap.parse_args()

    device = torch.device(args.device)
    amp_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}

    exp_ids = _read_runs_txt(Path(args.experiments_file))
    if not exp_ids:
        print("Empty experiments file.")
        return

    for idx, exp_id in enumerate(exp_ids, 1):
 
        exp_dir = _exp_dir_from_id(exp_id, args.dataset)
        ckpt_dir = exp_dir
        ckpt = _pick_ckpt(ckpt_dir)
        cfg_path = ckpt_dir / "run_config.json"

        if ckpt is None or not cfg_path.exists():
            print(f"[SKIP] Missing checkpoint or run_config.json in: {exp_dir}")
            continue

        with open(cfg_path, "r") as f:
            cfg = json.load(f)

        image_size = tuple(cfg["image_size"])
        vit_name = cfg["vit_name"]
        model_type = cfg.get("model_type", "enc_dec")
        
        print(f"[INFO] Model type: {model_type}")
        print(f"[INFO] ViT: {vit_name}")
        print(f"[INFO] Image size: {image_size}")

        DatasetCls = get_dataset_class(args.dataset)
        root = DatasetCls.default_root(get_data_root())
        
        tf = get_transforms(dataset=args.dataset, kind="basic", image_size=image_size)
        ds = DatasetCls(split="test", transform=tf, root=root, return_masks=False)
        
        num_classes = getattr(DatasetCls, "NUM_CLASSES", None)
        if num_classes is None:
            print(f"[SKIP] Dataset {args.dataset} does not define NUM_CLASSES")
            continue

        loader = DataLoader(
            ds, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=args.num_workers
        )

        patch_size = infer_patch_size(vit_name)
        encoder = ViT(
            img_size=image_size,
            patch_size=patch_size,
            backbone_name=vit_name,
            ckpt_path=None,  
        )

       
        model = build_model(model_type=model_type, vit=encoder, num_classes=num_classes,image_size=image_size, config=cfg, mode="eval", device=device)


        ckpt = torch.load(str(ckpt), map_location=device, weights_only=False)

        model.eval()

        evaluator = Evaluator(
            model, 
            num_classes=num_classes, 
            device=device, 
            include_background=True
        )
        
        seg = evaluator.run(loader, dataset_name=args.dataset)
        
        eff = evaluator.efficiency(
            input_shape=(1, 3, image_size[0], image_size[1]),
            warmup=args.warmup,
            runs=args.runs,
            amp_dtype=amp_map[args.amp_dtype],
            use_gpu=(device.type == "cuda"),
        )

        out = {
            "experiment_id": exp_dir.name,
            "dataset": args.dataset,
            "model_type": model_type,
            "vit_name": vit_name,
            "image_size": list(image_size),
            "mean_miou": seg["mean_miou"],
            "mean_hd95": seg["mean_hd95"],
            "miou_per_class": seg["miou_per_class"],
            "hd95_per_class": seg["hd95_per_class"],
            "total_params": eff["total_params"],
            "gflops": eff["gflops"],
            "fps": eff["fps"],
            "ckpt": str(ckpt),
        }
        
        if model_type == "enc_dec":
            out["decoder"] = cfg.get("decoder", "unknown")
        elif model_type == "mask2former":
            out["num_queries"] = cfg.get("num_queries")
            out["nheads"] = cfg.get("nheads")
        elif model_type == "eomt":
            out["eomt_num_blocks"] = cfg.get("eomt_num_blocks")
            out["eomt_num_q"] = cfg.get("eomt_num_q")

        out_json = exp_dir / "eval.json"
        with open(out_json, "w") as f:
            json.dump(out, f, indent=2)

        print(f"\n[OK] {exp_dir.name}")
        print(f"     mIoU:   {out['mean_miou']:.4f}")
        print(f"     mHD95:  {out['mean_hd95']:.4f}")
        print(f"     GFLOPs: {out['gflops']:.2f}")
        print(f"     FPS:    {out['fps']:.1f}")
        print(f"     Params: {out['total_params']:,}")
        print(f"     Saved to: {out_json}")

if __name__ == "__main__":
    main()