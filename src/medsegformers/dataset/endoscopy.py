from pathlib import Path
from typing import Tuple, Union
import numpy as np
from PIL import Image
import torch
from monai.data import Dataset
from monai.transforms import Compose
from . import register_dataset

@register_dataset
class EndoscopyDataset(Dataset):
    DATASET_NAME = "endoscopy"
    NUM_CLASSES = 6
    REL_ROOT = Path("endoscapes_segmentation_dataset") / "endoscapes_segmentations_processed"

    def __init__(
        self,
        root: Union[str, Path],
        split: str,  # ["train", "validation", "test"]
        transform: Compose,
        return_masks: bool = False,
        split_ratio: Tuple[float, float, float] = (0.7, 0.2, 0.1),
        seed: int = 42,
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.return_masks = return_masks

        if split not in {"train", "validation", "test"}:
            raise ValueError(f"split must be 'train'|'validation'|'test', got {split!r}")

        img_dir = self.root / "images"
        msk_dir = self.root / "masks"
        if not img_dir.is_dir():
            raise RuntimeError(f"Images dir not found: {img_dir}")
        if not msk_dir.is_dir():
            raise RuntimeError(f"Masks dir not found: {msk_dir}")

        mask_dict = {p.name: p for p in sorted(msk_dir.glob("*.png"))}
        image_paths = sorted(img_dir.glob("*.png"))

        pairs = []
        
        for img_path in image_paths:
            mask_path = mask_dict.get(img_path.name)
            if mask_path and self._is_non_empty(mask_path):
                pairs.append((img_path, mask_path))

        if not pairs:
            raise RuntimeError("No valid image-mask pairs found")

        n = len(pairs)
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n)

        tr, va, te = split_ratio
        n_train = int(tr * n)
        n_val = int(va * n)

        train_ids = perm[:n_train]
        val_ids = perm[n_train:n_train + n_val]
        test_ids = perm[n_train + n_val:]

        if split == "train":
            idxs = train_ids
        elif split == "validation":
            idxs = val_ids
        else:
            idxs = test_ids

        self.items = [pairs[i] for i in idxs]

    def _is_non_empty(self, mask_path: Path) -> bool:
        """Check if mask contains more than just background"""
        with Image.open(mask_path) as img:
            mask = np.array(img)
        return np.max(mask) > 0  # Has at least one non-background pixel
    
    def _semantic_to_targets(self, semantic_mask: torch.Tensor):

        sm = semantic_mask[0].long()  # [H, W]
        masks, labels = [], []

        for cls_id in range(self.NUM_CLASSES):  
            m = (sm == cls_id)
            if m.any():
                masks.append(m)
                labels.append(cls_id) 

        H, W = semantic_mask.shape[-2:]
        target = {
            "masks": torch.stack(masks, dim=0) if masks else 
                     torch.zeros((0, H, W), dtype=torch.bool),
            "labels": torch.tensor(labels, dtype=torch.long) if labels else 
                     torch.zeros(0, dtype=torch.long),
            "is_crowd": torch.zeros(len(labels), dtype=torch.bool) if labels else 
                       torch.zeros(0, dtype=torch.bool),
        }

        return target

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        img_path, mask_path = self.items[index]

        data = {"image": str(img_path), "label": str(mask_path)}
        if self.transform is not None:
            data = self.transform(data)

        image = data["image"]    
        semantic_mask = data["label"] 

        if self.return_masks:
            targets = self._semantic_to_targets(semantic_mask)
            return image, targets
        else:
            return {
                "image": image,
                "label": semantic_mask
            }

    @classmethod
    def default_root(cls, data_root: Path) -> Path:
        return Path(data_root) / cls.REL_ROOT