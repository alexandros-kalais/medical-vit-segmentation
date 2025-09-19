import torch.nn as nn
import torch.nn.functional as F
from . import register

@register("linear", input_kind="single")
class LinearHead(nn.Module):
    DECODER_NAME = "linear"
    def __init__(self, in_channels: int, num_classes: int, upsample_factor: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        self.ups = upsample_factor
        
    def forward(self, fmap):
        lo = self.proj(fmap)
        return F.interpolate(lo, scale_factor=self.ups, mode="bilinear", align_corners=False)
