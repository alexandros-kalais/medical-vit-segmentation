from pathlib import Path
from typing import Tuple, Union
import numpy as np
from PIL import Image
import torch
from monai.data import Dataset
from monai.transforms import Compose
from sklearn.model_selection import KFold
from . import register_dataset

@register_dataset
class EndoscopyDataset(Dataset):
    DATASET_NAME = "endoscopy"
    NUM_CLASSES = 6
    REL_ROOT = Path("endoscapes_segmentation_dataset") / "endoscapes_segmentations_processed"

    def __init__(
        self,
        root: Union[str, Path],
        split: str,  # ["cross-val", "test"]
        transform: Compose,
        return_masks: bool = False,
        split_ratio: Tuple[float,float] = (0.9, 0.1),
        seed: int = 42,
        n_folds: int = 5,
        fold_idx: int = 0,
        train: bool = True
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.return_masks = return_masks

        if split not in {"test", "cross-val"}:
            raise ValueError(f"split must be 'train'|'validation'|'test'|'cross-val', got {split!r}")

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
        
        _ , te = split_ratio
        n_test = int(te * n)
        
        trainval_ids = perm[:-n_test] if n_test > 0 else perm
        test_ids = perm[-n_test:] if n_test > 0 else np.array([])
        
        if split == "cross-val":

            n_trainval = len(trainval_ids)
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
            folds = list(kf.split(trainval_ids))
            fold_train_indices, fold_val_indices = folds[fold_idx]
            
            train_ids = trainval_ids[fold_train_indices]
            val_ids = trainval_ids[fold_val_indices]
            
            if train:
                idxs = train_ids
            else:
                idxs = val_ids
        else:
            idxs = test_ids

        idxs = np.sort(idxs)
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