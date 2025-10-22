from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd,
    RandFlipd, RandRotate90d, RandAffined, Resized, Lambdad, EnsureTyped
)
from medsegformers.utils import COLOR_MAP
import numpy as np
import torch

IGNORE_INDEX = 255
COLOR_MAP = COLOR_MAP.cpu().numpy().astype(np.uint8)

def mask_to_indices_endoscopy(x: np.ndarray) -> np.ndarray:

    if x.ndim == 2:
        x = x[None, ...]
    c, h, w = x.shape

    if c == 3:
        rgb = np.moveaxis(x, 0, -1).astype(np.uint8)
        y = np.full((h, w), IGNORE_INDEX, dtype=np.int64)
        for idx, color in enumerate(COLOR_MAP):
            matches = np.all(rgb == color, axis=-1)
            y[matches] = idx
        return y[None, :]

    raise ValueError(f"Unexpected mask shape {x.shape} in mask_to_indices_endoscopy")

def get_transforms(dataset: str, kind="basic", image_size=None):
    """
    kind: "none" | "basic" | "aug"
    """
    keys_imglab = ["image", "label"]
    tfs = [
        LoadImaged(keys=keys_imglab, image_only=True),
        EnsureChannelFirstd(keys=keys_imglab),
    ]


    tfs += [
        Lambdad(keys="label", func=mask_to_indices_endoscopy),
    ]

    if kind in ("basic", "aug"):
        tfs += [ScaleIntensityd(keys="image")]

    if kind == "aug":
        tfs += [
            RandFlipd(keys=keys_imglab, prob=0.5, spatial_axis=1),
            # RandRotate90d(keys=keys_imglab, prob=0.5, max_k=3),
            RandAffined(
                keys=keys_imglab,
                prob=0.5,
                rotate_range=(0, 0, 0.1),
                scale_range=(0.1, 0.1, 0.0),
                mode=("bilinear", "nearest"),
                padding_mode="zeros",
            ),
        ]
    
    if image_size:
        tfs += [
            Resized(
                keys=keys_imglab,
                spatial_size=image_size,
                mode=("bilinear", "nearest"),
            ),
        ]

    tfs += [EnsureTyped(keys=("image", "label"), dtype=(torch.float32, torch.long))]
    return Compose(tfs)
