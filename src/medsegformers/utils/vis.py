import numpy as np
import torch

COLOR_MAP = torch.tensor([
    [  0,   0,   0],  # 0 background
    [255,   0,   0],  # 1 cystic plate
    [  0, 255,   0],  # 2 Calot triangle
    [  0,   0, 255],  # 3 cystic artery
    [255, 255,   0],  # 4 cystic duct
    [255,   0, 255],  # 5 gallbladder
    [  0, 255, 255],  # 6 tools
], dtype=torch.uint8)

# Class names aligned with your palette / label indexing
ENDOSCOPY_CLASS_NAMES = [
    "background",       # 0
    "cystic plate",     # 1
    "Calot triangle",   # 2
    "cystic artery",    # 3
    "cystic duct",      # 4
    "gallbladder",      # 5
    "tools",            # 6
]


def colorize_index_map(idx_map: torch.Tensor) -> torch.Tensor:
    """
    idx_map: (B,H,W) long or (H,W) -> returns (B,3,H,W) uint8
    """
    if idx_map.ndim == 2:
        idx_map = idx_map.unsqueeze(0)
    idx_map = idx_map.long()
    cmap = COLOR_MAP.to(idx_map.device)
    colored = cmap[idx_map]           # (B,H,W,3)
    return colored.permute(0,3,1,2).contiguous()

def to_np_uint8(grid: torch.Tensor) -> np.ndarray:
    x = grid.permute(1,2,0).cpu().numpy()
    return np.clip(x, 0, 255).astype(np.uint8)
