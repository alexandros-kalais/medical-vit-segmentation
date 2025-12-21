import argparse
import json
from pathlib import Path
import torch
import numpy as np
from typing import Dict, List
from monai.data import DataLoader

from medsegformers.models import ViT
from medsegformers.dataset import get_dataset_class, get_transforms
from medsegformers.utils import get_data_root, ckpt_dir
from medsegformers.engines import Evaluator
from . import infer_patch_size, build_model
from medsegformers.cli.helpers import MaskClassificationSemanticWrapper


def find_best_checkpoint(fold_dir: Path) -> Path:
    """Find the best checkpoint in a fold directory."""
    ckpts = list(fold_dir.glob("best-*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found in {fold_dir}")
    
    # If multiple, take the one with highest miou in filename
    return max(ckpts, key=lambda p: float(p.stem.split('-')[1]) if len(p.stem.split('-')) > 1 and p.stem.split('-')[1].replace('.', '').isdigit() else 0.0)

def evaluate_single_fold(
    fold_idx: int,
    fold_dir: Path,
    test_loader: DataLoader,
    num_classes: int,
    dataset_name: str,
    device: torch.device,
    image_size: tuple,
    amp_dtype_str: str,
) -> Dict:
    
    # Load fold config
    cfg_path = fold_dir / "run_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing run_config.json in {fold_dir}")
    
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    
    vit_name = cfg["vit_name"]
    model_type = cfg.get("model_type", "enc_dec")
    
    # Find checkpoint
    ckpt_path = find_best_checkpoint(fold_dir)
    
    # Load checkpoint weights
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        if any(k.startswith("network.") for k in state_dict.keys()):
            state_dict = {k.replace("network.", "", 1): v for k, v in state_dict.items()}
    else:
        state_dict = ckpt
    
    # Build model
    patch_size = infer_patch_size(vit_name)
    encoder = ViT(
        img_size=image_size,
        patch_size=patch_size,
        backbone_name=vit_name,
        ckpt_path=None,
    ).to(device)
    
    model = build_model(
        model_type=model_type,
        vit=encoder,
        num_classes=num_classes,
        image_size=image_size,
        config=cfg,
        mode="eval",
        device=device
    )
    
    model.load_state_dict(state_dict, strict=False)

    model = model.to(device)

    if model_type in ["mask2former", "eomt"]:
        model = MaskClassificationSemanticWrapper(
            model, img_size=image_size, num_classes=num_classes
        )
    model = model.to(device)

    model.eval()
    
    # Evaluate
    evaluator = Evaluator(
        model,
        num_classes=num_classes,
        device=device,
        include_background=True
    )
    
    seg_metrics = evaluator.run(test_loader, dataset_name=dataset_name)
    
    # Efficiency metrics (only compute once, not per fold)
    amp_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}
    eff_metrics = evaluator.efficiency(
        input_shape=(1, 3, image_size[0], image_size[1]),
        warmup=10,
        runs=50,
        amp_dtype=amp_map[amp_dtype_str],
        use_gpu=(device.type == "cuda"),
    )
    
    result = {
        "fold": fold_idx,
        "checkpoint": str(ckpt_path),
        "mean_miou": seg_metrics["mean_miou"],
        "mean_hd95": seg_metrics["mean_hd95"],
        "miou_per_class": seg_metrics["miou_per_class"],
        "hd95_per_class": seg_metrics["hd95_per_class"],
        "total_params": eff_metrics["total_params"],
        "gflops": eff_metrics["gflops"],
        "fps": eff_metrics["fps"],
    }
        
    return result


def compute_statistics(fold_results: List[Dict], metric_name: str) -> Dict:
    values = [r[metric_name] for r in fold_results]
    return {
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "values": values,
    }


def main():
    ap = argparse.ArgumentParser(description="Evaluate 5-fold cross-validation results on test set")
    ap.add_argument("--dataset", required=True, help="e.g., endoscopy")
    ap.add_argument("--experiment_id", required=True, 
                    help="CV experiment ID (e.g., linear_dino_448x448_lr0.0001_bs4_2025-10-23-14-38-15_5fold_cv)")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp_dtype", default="bf16", choices=["bf16", "fp16", "none"])
    args = ap.parse_args()
    
    device = torch.device(args.device)
    
    # Find CV experiment directory
    cv_dir = ckpt_dir(args.dataset, args.experiment_id)
    if not cv_dir.exists():
        raise FileNotFoundError(f"CV experiment directory not found: {cv_dir}")
        
    # Find all fold directories
    fold_dirs = sorted(cv_dir.glob("fold_*"))
    if not fold_dirs:
        raise FileNotFoundError(f"No fold directories found in {cv_dir}")
    
    n_folds = len(fold_dirs)
    print(f"Found {n_folds} folds\n")
    
    # Load config from first fold to get dataset parameters
    first_cfg_path = fold_dirs[0] / "run_config.json"
    with open(first_cfg_path, "r") as f:
        cfg = json.load(f)
    
    image_size = tuple(cfg["image_size"])
    seed = cfg.get("seed", 42)
    
    # Create test dataset (using same seed as training)
    DatasetCls = get_dataset_class(args.dataset)
    root = DatasetCls.default_root(get_data_root())
    
    tf = get_transforms(dataset=args.dataset, kind="basic", image_size=image_size)
    test_ds = DatasetCls(split="test", transform=tf, root=root, seed=seed, return_masks=False)
    
    num_classes = getattr(DatasetCls, "NUM_CLASSES", None)
    if num_classes is None:
        raise ValueError(f"Dataset {args.dataset} does not define NUM_CLASSES")
    
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
       
    # Evaluate each fold
    fold_results = []
    for fold_dir in fold_dirs:
        fold_idx = int(fold_dir.name.split("_")[1])
        
        try:
            result = evaluate_single_fold(
                fold_idx=fold_idx,
                fold_dir=fold_dir,
                test_loader=test_loader,
                num_classes=num_classes,
                dataset_name=args.dataset,
                device=device,
                image_size=image_size,
                amp_dtype_str=args.amp_dtype,
            )
            fold_results.append(result)
        except Exception as e:
            print(f"[ERROR] Failed to evaluate fold {fold_idx}: {e}")
            continue
    
    if not fold_results:
        print("\n[ERROR] No folds were successfully evaluated!")
        return
    
    miou_stats = compute_statistics(fold_results, "mean_miou")
    hd95_stats = compute_statistics(fold_results, "mean_hd95")
            
    # Compute per-class statistics
    per_class_miou_stats = []
    per_class_hd95_stats = []
    
    for cls_idx in range(num_classes):
        cls_mious = [r["miou_per_class"][cls_idx] for r in fold_results]
        cls_hd95s = [r["hd95_per_class"][cls_idx] for r in fold_results]
        
        per_class_miou_stats.append({
            "class_idx": cls_idx,
            "median": float(np.median(cls_mious)),
            "min": float(np.min(cls_mious)),
            "max": float(np.max(cls_mious)),
            "mean": float(np.mean(cls_mious)),
            "std": float(np.std(cls_mious)),
        })
        
        per_class_hd95_stats.append({
            "class_idx": cls_idx,
            "median": float(np.median(cls_hd95s)),
            "min": float(np.min(cls_hd95s)),
            "max": float(np.max(cls_hd95s)),
            "mean": float(np.mean(cls_hd95s)),
            "std": float(np.std(cls_hd95s)),
        })
    
    # Use efficiency metrics from first fold (they're the same across folds)
    efficiency = {
        "total_params": fold_results[0]["total_params"],
        "gflops": fold_results[0]["gflops"],
        "fps": fold_results[0]["fps"],
    }
    
    # Save comprehensive results
    output = {
        "experiment_id": args.experiment_id,
        "dataset": args.dataset,
        "n_folds": len(fold_results),
        "config": cfg,
        "test_set_size": len(test_ds),
        "summary": {
            "miou": miou_stats,
            "hd95": hd95_stats,
        },
        "per_class_miou": per_class_miou_stats,
        "per_class_hd95": per_class_hd95_stats,
        "efficiency": efficiency,
        "fold_results": fold_results,
    }
    
    output_path = cv_dir / "test_eval_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
if __name__ == "__main__":
    main()