from monai.transforms import (
    Compose,
    LoadImaged, EnsureChannelFirstd, ScaleIntensityd, AsDiscreted,
    RandFlipd, RandRotate90d, RandAffined, Resized, Lambdad, EnsureTyped
)
import numpy as np

# ----- dataset-specific mask preprocessors -----
def binary_mask_preprocess(x: np.ndarray):
    # x is (H,W) or (C,H,W); ensure single channel and binarize
    if x.ndim == 2:
        x = x[None, ...]
    if x.shape[0] > 1:  # RGB mask -> take first channel
        x = x[:1, ...]
    return (x > 0).astype(np.float32)

def rgb_mask_to_class_indices(x):
    # example for multi-class: map RGB -> integer class indices
    COLOR_MAP = {
        0: (  0,   0,   0), # background
        1: (255,   0,   0), # cystic plate
        2: (  0, 255,   0), # Calot triangle
        3: (  0,   0, 255), # cystic artery
        4: (255, 255,   0), # cystic duct
        5: (255,   0, 255), # gallbladder
        6: (  0, 255, 255), # tools
    }
    x = x.permute(1, 2, 0).astype(np.uint8)  # (C,H,W) -> (H,W,3)
    h, w, _ = x.shape
    y = np.zeros((h, w), dtype=np.int64)
    for idx, color in COLOR_MAP.items():
        matches = np.all(x == np.array(color, dtype=np.uint8), axis=-1)
        y[matches] = idx
    return y[None, ...]

def get_transforms(dataset: str, kind="basic", image_size=None):
    """
    dataset: "hyperkvasir" (binary) | "endoscopy" (multi-class)
    kind: "none" | "basic" | "aug"
    """
    keys_imglab = ["image", "label"]

    tfs = [
        LoadImaged(keys=keys_imglab, image_only=True),  # loads both
        EnsureChannelFirstd(keys=keys_imglab),
    ]

    # dataset-specific mask preprocessing
    if dataset == "hyperkvasir":  # binary masks
        tfs += [
            Lambdad(keys="label", func=binary_mask_preprocess),
            AsDiscreted(keys="label", threshold=0.5),
        ]
    elif dataset == "endoscopy":  # multi-class masks
        tfs += [
            Lambdad(keys="label", func=rgb_mask_to_class_indices),
        ]
    else:
        raise ValueError(f"Unknown dataset {dataset}")

    # shared extras
    if kind in ("basic", "aug"):
        tfs += [
            ScaleIntensityd(keys="image"),
        ]

    if image_size:
        # You can give per-key modes via dict:
        tfs += [
            Resized(keys=keys_imglab, spatial_size=image_size,
                    mode=("bilinear", "nearest")),
        ]

    if kind == "aug":
        # One random decision shared across both keys for each transform:
        tfs += [
            RandFlipd(keys=keys_imglab, prob=0.9, spatial_axis=1),
            RandRotate90d(keys=keys_imglab, prob=0.9, max_k=3),
            RandAffined(
                keys=keys_imglab,
                prob=0.9,
                rotate_range=(0, 0, 0.1),
                scale_range=(0.1, 0.1, 0.0),
                mode=("bilinear", "nearest"),
                padding_mode="zeros",
            ),
        ]

    # tfs += [EnsureTyped(keys=keys_imglab)]
    return Compose(tfs)

