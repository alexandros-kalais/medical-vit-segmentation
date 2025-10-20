from pathlib import Path
from typing import Tuple, Union
import numpy as np
from PIL import Image

import torch
from monai.data import Dataset
from monai.transforms import Compose

from . import register_dataset

"""
NEED TO CHANGE THIS!
"""

@register_dataset
class HyperKvasirDataset(Dataset):
    DATASET_NAME = "hyperkvasir"
    NUM_CLASSES = 2
    REL_ROOT = Path("HyperKvasir")

    def __init__(
        self,
        root: Union[str, Path],
        split: str,
        transform: Compose,
        seed: int = 42,
        return_masks: bool = False):

        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.return_masks = return_masks
        self.seed = seed

        if split not in {"train", "validation", "test"}:
            raise ValueError(f"split must be 'train'|'validation'|'test', got {split!r}")

        img_dir = self.root / "Images" / split
        msk_dir = self.root / "Masks"

        if not img_dir.is_dir():
            raise RuntimeError(f"Images dir not found: {img_dir}")
        if not msk_dir.is_dir():
            raise RuntimeError(f"Masks dir not found: {msk_dir}")

        mask_dict = {p.name: p for p in msk_dir.glob("*.jpg")}
        image_paths = list(img_dir.glob("*.jpg"))

        pairs: list[tuple[Path, Path]] = []
        for img_path in image_paths:
            mask_path = mask_dict.get(img_path.name)
            if mask_path and self._is_non_empty(mask_path):
                pairs.append((img_path, mask_path))
        
        if not pairs:
            raise RuntimeError("No valid image–mask pairs found (after empty-mask filtering).")

        self.items = pairs

        
    def _is_non_empty(self, mask_path: Path) -> bool:
        """Check if mask contains more than just background"""
        with Image.open(mask_path) as img:
            mask = np.array(img)
        return np.max(mask) > 0  # Has at least one non-background pixel
    
    def _semantic_to_targets(self, semantic_mask: torch.Tensor):
        sm = semantic_mask[0].long()  # (H,W) in {0,1}
        H, W = sm.shape
        masks, labels = [], []

        fg = (sm == 1)
        if fg.any():
            masks.append(fg)          
            labels.append(0)          

        target = {
            "masks": torch.stack(masks, dim=0) if masks else torch.zeros((0, H, W), dtype=torch.bool),
            "labels": torch.tensor(labels, dtype=torch.long) if labels else torch.zeros(0, dtype=torch.long),
            "is_crowd": torch.zeros(len(labels), dtype=torch.bool) if labels else torch.zeros(0, dtype=torch.bool),
        }
        return target


    def __len__(self):
        return len(self.items)

    def __getitem__(self, index: int):
        img_path, mask_path = self.items[index]

        data = {"image": str(img_path), "label": str(mask_path)}
        if self.transform is not None:
            data = self.transform(data)

        image = data["image"]           # tensor (3,H,W), float
        semantic_mask = data["label"]   # tensor (1,H,W), expected {0,1} after transforms

        if self.return_masks:
            targets = self._semantic_to_targets(semantic_mask)
            return image, targets
        else:
            return {"image": image, "label": semantic_mask}

    @classmethod
    def default_root(cls, data_root: Path) -> Path:
        return Path(data_root) / cls.REL_ROOT

    @classmethod
    def get_image_size(cls, vit_name: str, user_size=None) -> Tuple[int, int]:
        # mirror EndoscopyDataset behavior
        if user_size not in (None, "auto") and len(user_size) == 2:
            return tuple(user_size)

        name = vit_name.lower()
        if "dinov2" in name:
            return (476, 854)
        elif "dinov3" in name:
            return (480, 848)
        else:
            return (224, 224)