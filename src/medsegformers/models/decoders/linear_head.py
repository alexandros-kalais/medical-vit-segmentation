import torch.nn as nn
import torch.nn.functional as F
import torch

class LinearHead(nn.Module):
    """
    1x1 conv to map (B, D, H', W') -> (B, C, H, W) via bilinear upsample.
    """
    def __init__(self, in_channels: int, num_classes: int, upsample_factor: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        self.ups = upsample_factor

    def forward(self, fmap):
        logits_low = self.proj(fmap)  # (B, C, H', W')
        logits = F.interpolate(
            logits_low,
            scale_factor=self.ups,
            mode="bilinear",
            align_corners=False,
        )
        return logits

