# debug_grid_fix.py
import torch
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from medsegformers.data import HyperKvasirDataset, EndoscopyDataset
from medsegformers.transforms import get_transforms

COLOR_MAP = torch.tensor([
    [0, 0, 0],
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 255, 0],
    [255, 0, 255],
    [0, 255, 255],
], dtype=torch.uint8)

def colorize_index_map(idx_map: torch.Tensor) -> torch.Tensor:
    if idx_map.ndim == 2:
        idx_map = idx_map.unsqueeze(0)
    cmap = COLOR_MAP.to(idx_map.device)
    return cmap[idx_map].permute(0, 3, 1, 2).contiguous()  # (B,3,H,W)

def grid_to_uint8(grid: torch.Tensor) -> np.ndarray:
    """
    Accepts a grid (C,H,W) tensor that may be float in [0,1] or uint8 [0,255],
    returns HxWx3 uint8.
    """
    g = grid.detach().cpu()
    if g.dtype.is_floating_point:
        # assume [0,1] if normalize=True was used
        g = torch.clamp(g, 0, 1) * 255.0
    else:
        # uint8 or int
        g = torch.clamp(g, 0, 255)
    g = g.permute(1, 2, 0).numpy().astype(np.uint8)
    return g

def show_stats(batch, name):
    print(f"\n{name} stats per image (C,H,W):")
    for i, img in enumerate(batch):
        vmin = float(img.min())
        vmax = float(img.max())
        vmean = float(img.mean())
        dtype = img.dtype
        shape = tuple(img.shape)
        print(f"  [{i}] dtype={dtype} shape={shape} min={vmin:.6f} max={vmax:.6f} mean={vmean:.6f}")

def main():
    dataset_name = "endoscopy"  # or "hyperkvasir"
    image_size = (224, 224)
    B = 4

    tf = get_transforms(dataset=dataset_name, kind="basic", image_size=image_size)

    if dataset_name == "hyperkvasir":
        ds = HyperKvasirDataset(root=Path(__file__).resolve().parents[2] / "data" / "HyperKvasir",
                                split="train", transform=tf)
        num_classes = 1
    else:
        ds = EndoscopyDataset(
            root=Path(__file__).resolve().parents[2] / "data" / "endoscapes_segmentation_dataset" / "endoscapes_segmentations_processed",
            split="validation", transform=tf, split_ratio=(0.7, 0.2, 0.1), seed=42
        )
        num_classes = 7

    # pull first batch (matching your subset=4 case)
    imgs = []
    labs = []
    for i in range(B):
        s = ds[i]
        imgs.append(s["image"])
        labs.append(s["label"])
    images = torch.stack(imgs)            # (B, C, H, W)
    labels = torch.stack(labs)            # (B, 1, H, W) or (B, H, W) depending on dataset

    # stats
    show_stats(images, "images (raw)")
    show_stats(labels.float(), "labels (as float for stats)")

    # colorize labels
    labs_idx = labels.squeeze(1).long()
    lab_rgb  = colorize_index_map(labs_idx)  # (B,3,H,W)

    # --- GRID with normalize=True (per-image min/max) ---
    img_grid_norm = make_grid(images, nrow=2, normalize=True, scale_each=True)
    lab_grid = make_grid(lab_rgb, nrow=2, normalize=False)

    # --- GRID with normalize=False (assumes images already in [0,1] or uint8) ---
    img_grid_no = make_grid(images, nrow=2, normalize=False)

    # Save with torchvision (handles float nicely)
    save_image(img_grid_norm, "debug_images_grid_norm_saveimage.png")
    save_image(img_grid_no,   "debug_images_grid_no_norm_saveimage.png")
    save_image(lab_grid.float()/255.0, "debug_labels_grid_saveimage.png")  # convert uint8→float

    # Save via corrected uint8 conversion
    import matplotlib.pyplot as plt
    plt.imsave("debug_images_grid_norm_uint8.png", grid_to_uint8(img_grid_norm))
    plt.imsave("debug_images_grid_no_norm_uint8.png", grid_to_uint8(img_grid_no))
    plt.imsave("debug_labels_grid_uint8.png", grid_to_uint8(lab_grid))

    print("\nWrote:")
    print("  debug_images_grid_norm_saveimage.png")
    print("  debug_images_grid_no_norm_saveimage.png")
    print("  debug_labels_grid_saveimage.png")
    print("  debug_images_grid_norm_uint8.png")
    print("  debug_images_grid_no_norm_uint8.png")
    print("  debug_labels_grid_uint8.png")

    # Optional: also dump the 4 individual images for sanity
    for i in range(B):
        save_image(images[i], f"sample_image_{i}.png")             # auto-handles float [0,1]
        save_image(lab_rgb[i].float()/255.0, f"sample_label_{i}.png")

if __name__ == "__main__":
    main()
