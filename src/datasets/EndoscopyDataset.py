from pathlib import Path
from typing import List, Dict, Tuple, Union
from monai.data import Dataset
from monai.transforms import Compose
import numpy as np


class EndoscopyDataset(Dataset):

    def __init__(
        self,
        root: Union[str, Path],
        split: str,                                 # "train" | "validation" | "test"
        transform: Compose,                         # REQUIRED, dict-style
        split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42,
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform

        if split not in {"train", "validation", "test"}:
            raise ValueError(f"split must be 'train'|'validation'|'test', got {split!r}")

        img_dir = self.root / "images"
        msk_dir = self.root / "masks"
        if not img_dir.is_dir():
            raise RuntimeError(f"Images dir not found: {img_dir}")
        if not msk_dir.is_dir():
            raise RuntimeError(f"Masks dir not found: {msk_dir}")

        # Build dict of masks
        mask_dict = {p.name: p for p in sorted(msk_dir.glob("*.png"))}

        # Collect and sort image paths 
        image_paths = sorted(img_dir.glob("*.png"))

        # Paired list
        pairs = [(img, mask_dict[img.name]) for img in image_paths]
        n = len(pairs)
        if n == 0:
            raise RuntimeError("No paired .png items found.")

        # Seeded permutation -> deterministic split slices
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n)

        tr, va, te = split_ratio 
        n_train = int(tr * n)
        n_val   = int(va * n)

        train_ids = perm[:n_train]
        val_ids   = perm[n_train:n_train + n_val]
        test_ids  = perm[n_train + n_val:]

        if split == "train":
            idxs = train_ids
        elif split == "validation":
            idxs = val_ids
        else:
            idxs = test_ids

        # Build MONAI dict items
        items: List[Dict[str, str]] = [
            {"image": str(pairs[i][0]), "label": str(pairs[i][1])} for i in idxs
        ]
        if not items:
            raise RuntimeError(
                f"No items selected for split='{split}' with ratio={split_ratio} (n={n})."
            )

        super().__init__(data=items, transform=transform)
