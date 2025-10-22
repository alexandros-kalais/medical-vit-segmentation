import numpy as np
import torch

IGNORE_INDEX = 255

COLOR_MAP = torch.tensor([
    [255,   0,   0],  # 0 cystic plate
    [  0, 255,   0],  # 1 Calot triangle
    [  0,   0, 255],  # 2 cystic artery
    [255, 255,   0],  # 3 cystic duct
    [255,   0, 255],  # 4 gallbladder
    [  0, 255, 255],  # 5 tools
], dtype=torch.uint8)

ENDOSCOPY_CLASS_NAMES = [
    "cystic plate",     # 0
    "Calot triangle",   # 1
    "cystic artery",    # 2
    "cystic duct",      # 3
    "gallbladder",      # 4
    "tools",            # 5
]

def colorize_index_map(idx_map: torch.Tensor, num_classes: int | None = None) -> torch.Tensor:

    if idx_map.ndim == 2:
        idx_map = idx_map.unsqueeze(0)
    idx_map = idx_map.long()

    device = idx_map.device

    B, H, W = idx_map.shape
    colored = torch.zeros(B, H, W, 3, dtype=torch.uint8, device=device)

    

    valid = (idx_map >= 0) & (idx_map < len(COLOR_MAP.to(device)))
    colored[valid] = COLOR_MAP.to(device)[idx_map[valid]]

    return colored.permute(0, 3, 1, 2).contiguous()


def to_np_uint8(grid: torch.Tensor) -> np.ndarray:

    x = grid.permute(1, 2, 0).cpu().numpy()
    return np.clip(x, 0, 255).astype(np.uint8)
