import argparse, json
from pathlib import Path
import torch
from typing import Optional
from monai.data import DataLoader

from medsegformers.models.build import build_segmentation_model
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
    ckpt = torch.load(str(ckpt_path), map_location=device)
    sd = ckpt.get("state_dict", {})
    net_sd = {k[len("network."):]: v for k, v in sd.items() if k.startswith("network.")}
    missing, unexpected = model.load_state_dict(net_sd, strict=False)
    if missing or unexpected:
        print(f"[WARN] load_state_dict mismatches -> missing: {missing}, unexpected: {unexpected}")


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
        decoder    = cfg["decoder"]
        seed       = cfg.get("seed", 42)

        DatasetCls = get_dataset_class(args.dataset)
        root = DatasetCls.default_root(get_data_root())

        tf = get_transforms(dataset=args.dataset, kind="basic", image_size=image_size)
        ds = DatasetCls(split="test",transform=tf, root=root, return_masks=False)

        num_classes = getattr(DatasetCls, "NUM_CLASSES", None)

        loader =  DataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers)

        # build & load weights
        model = build_segmentation_model(
            decoder=decoder,
            num_classes=num_classes,
            vit_name=vit_name,
            pretrained=True,
            freeze_encoder=False,
            image_size=image_size,
            decoder_kwargs=cfg.get("decoder_kwargs", None),
            unfreeze_last_k=cfg.get("unfreeze_last_k", 0),
        ).to(device)
        _load_network_weights_from_lightning_ckpt(model, ckpt, device)
        model.eval()

        

        # evaluate seg + efficiency
        evaluator = Evaluator(
            model, num_classes=num_classes, device=device, include_background=True
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
            "decoder": decoder,
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

        out_json = exp_dir / f"eval.json"
        with open(out_json, "w") as f:
            json.dump(out, f, indent=2)

        print(f"[OK] {exp_dir.name} mIoU={out['mean_miou']:.4f} mHD95={out['mean_hd95']:.4f}  GFLOPs={out['gflops']:.2f}  FPS={out['fps']:.1f} params={out['total_params']}")

if __name__ == "__main__":
    main()
