from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd,
    RandFlipd, RandRotate90d, RandAffined, Resized, Lambdad, EnsureTyped
)
import numpy as np
import torch

def binary_mask_preprocess(x: np.ndarray):
    """
    Input x can be (H,W) or (C,H,W). Ensure single channel and binarize.
    Returns float32 {0,1} as [1,H,W].
    """
    if x.ndim == 2:
        x = x[None, ...]
    if x.shape[0] > 1:  # e.g. RGB mask -> take first channel
        x = x[:1, ...]
    return (x > 0).astype(np.float32)

PALETTE = np.array([
    [  0,   0,   0],  # 0 background
    [255,   0,   0],  # 1 cystic plate
    [  0, 255,   0],  # 2 Calot triangle
    [  0,   0, 255],  # 3 cystic artery
    [255, 255,   0],  # 4 cystic duct
    [255,   0, 255],  # 5 gallbladder
    [  0, 255, 255],  # 6 tools
], dtype=np.uint8)

def mask_to_indices_endoscopy(x: np.ndarray) -> np.ndarray:

    if x.ndim == 2:
        x = x[None, ...]
    c, h, w = x.shape

    if c == 1:
        y = x.astype(np.int64)
        return y  # [1,H,W], int64

    if c == 3:
        rgb = np.moveaxis(x, 0, -1).astype(np.uint8)  # (H,W,3)
        y = np.zeros((h, w), dtype=np.int64)
        for idx, color in enumerate(PALETTE):
            matches = np.all(rgb == color, axis=-1)
            y[matches] = idx
        return y[None, :]

    raise ValueError(f"Unexpected mask shape {x.shape} in mask_to_indices_endoscopy")

def get_transforms(dataset: str, kind="basic", image_size=None):
    """
    dataset: "hyperkvasir" (binary) | "endoscopy" (multi-class)
    kind: "none" | "basic" | "aug"
    """
    keys_imglab = ["image", "label"]
    tfs = [
        LoadImaged(keys=keys_imglab, image_only=True),
        EnsureChannelFirstd(keys=keys_imglab),
    ]

    # dataset-specific mask preprocessing
    if dataset == "hyperkvasir":  # binary masks
        tfs += [
            Lambdad(keys="label", func=binary_mask_preprocess),
        ]
    elif dataset == "endoscopy":  # multi-class masks
        tfs += [
            Lambdad(keys="label", func=mask_to_indices_endoscopy),
        ]
    else:
        raise ValueError(f"Unknown dataset {dataset}")

    # shared extras
    if kind in ("basic", "aug"):
        tfs += [ScaleIntensityd(keys="image")]

    if image_size:
        tfs += [
            Resized(
                keys=keys_imglab,
                spatial_size=image_size,
                mode=("bilinear", "nearest"),  # image bilinear, label nearest
            ),
        ]

    if kind == "aug":
        tfs += [
            RandFlipd(keys=keys_imglab, prob=0.5, spatial_axis=1),
            RandRotate90d(keys=keys_imglab, prob=0.5, max_k=3),
            RandAffined(
                keys=keys_imglab,
                prob=0.5,
                rotate_range=(0, 0, 0.1),
                scale_range=(0.1, 0.1, 0.0),
                mode=("bilinear", "nearest"),
                padding_mode="zeros",
            ),
        ]

    tfs += [EnsureTyped(keys=("image", "label"), dtype=(torch.float32, torch.long))]
    return Compose(tfs)
