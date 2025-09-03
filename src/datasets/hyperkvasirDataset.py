from pathlib import Path
from typing import List, Dict, Union
from monai.data import Dataset
from monai.transforms import Compose


class HyperKvasirDataset(Dataset):

    def __init__(
        self,
        root: Union[str, Path],
        split: str,            # "train" | "validation" | "test"
        transform: Compose, 
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform

        img_dir = self.root / "Images" / split
        msk_dir = self.root / "Masks"

        if not img_dir.is_dir():
            raise RuntimeError(f"Images dir not found: {img_dir}")
        if not msk_dir.is_dir():
            raise RuntimeError(f"Masks dir not found: {msk_dir}")

        # Build dictionary of masks
        mask_dict = {p.name: p for p in msk_dir.glob(f"*.jpg")}

        # Gather images
        image_paths = list(img_dir.glob(f"*.jpg"))

        # Build paired dataset
        items: List[Dict[str, str]] = [
            {"image": str(img_path), "label": str(mask_dict[img_path.name])}
            for img_path in image_paths
        ]

        if not items:
            raise RuntimeError(
                f"No paired items found for split='{split}' with ext .jpg."
                f"Checked {img_dir} vs {msk_dir}."
            )

        super().__init__(data=items, transform=transform)
