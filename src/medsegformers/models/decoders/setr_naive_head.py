import torch.nn as nn
import torch.nn.functional as F
from . import register

@register("naive", input_kind="single")
class NaiveHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, upsample_factor: int, mid_channels: int = 256):
        super().__init__()
        self.ups = upsample_factor

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1   = nn.SyncBatchNorm(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, num_classes, kernel_size=1, bias=True)

    def forward(self, fmap):
        x = self.conv1(fmap)
        x = F.relu(self.bn1(x), inplace=True)
        x = self.conv2(x)
        return F.interpolate(x, scale_factor=self.ups, mode="bilinear", align_corners=False)
