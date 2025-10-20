# probe_timm_intermediate_min.py
import json
import inspect
from typing import Any, Dict, List, Tuple

import torch
import timm


def tensor_shape(x):
    return list(x.shape) if isinstance(x, torch.Tensor) else None


def summarize_item(item) -> Dict[str, Any]:
    if isinstance(item, torch.Tensor):
        return {"kind": "tensor", "shape": tensor_shape(item)}
    if isinstance(item, tuple):
        shapes = []
        kinds = []
        for t in item:
            shapes.append(tensor_shape(t) if isinstance(t, torch.Tensor) else None)
            kinds.append(type(t).__name__)
        return {"kind": "tuple", "elem_types": kinds, "shapes": shapes}
    return {"kind": type(item).__name__}


@torch.no_grad()
def probe(model_name: str, img_size: Tuple[int, int], layers: List[int], batch: int = 2) -> None:
    print(f"[probe] model={model_name} img_size={img_size} layers={layers} batch={batch}")

    model = timm.create_model(model_name, pretrained=False, img_size=img_size, num_classes=0).eval()

    C = 3
    H, W = img_size
    x = torch.randn(batch, C, H, W)

    fn = getattr(model, "get_intermediate_layers", None)
    if fn is None:
        raise RuntimeError("Model has no `get_intermediate_layers`.")

    try:
        params = tuple(inspect.signature(fn).parameters.keys())
    except Exception:
        params = ()
    print(f"[probe] signature params={params}")

    calls = [
        ("plain", dict()),  # fn(x, n)
        ("return_prefix_tokens", dict(return_prefix_tokens=True)),
        ("return_cls_token", dict(return_cls_token=True)),
        ("return_class_token", dict(return_class_token=True)),  # what your adapter uses
    ]

    results = {}
    for tag, kw in calls:
        if not all(k in params for k in kw.keys()):
            results[tag] = {"skipped": True, "reason": "unsupported kwarg(s)"}
            continue
        try:
            out = fn(x, layers, **kw)
            summary = [summarize_item(it) for it in out]
            results[tag] = {"skipped": False, "ok": True, "items": summary}
        except TypeError as e:
            results[tag] = {"skipped": False, "ok": False, "error": f"TypeError: {e}"}
        except Exception as e:
            results[tag] = {"skipped": False, "ok": False, "error": f"{type(e).__name__}: {e}"}

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    # hard-coded per your request
    model_name = "vit_small_patch16_224.dino"
    img_size = (512, 512)
    layers = [0, 11, 23]
    probe(model_name, img_size, layers, batch=2)
