import torch.nn as nn
import torch.nn.functional as F
from . import register

@register("pup", input_kind="single")
class PUPHead(nn.Module):
    """
    PUP (Progressive UPsampling) for ViT/patch size:
      fmap (N, C, H', W') with H' = H/patch size, W' = W/patch size
      -> [3x3 Conv -> SyncBN -> ReLU -> 2x upsample] x 3  (H': *8)
      -> 3x3 Conv -> SyncBN -> ReLU -> 1x1 Conv
      -> final upsample to (H, W) computed as (H'*patch, W'*patch)

    """

    def __init__(self, in_channels: int, num_classes: int,
                 upsample_factor: int,          # patch size
                 mid_channels: int = 256,
                 align_corners: bool = False):
        super().__init__()
        self.patch = int(upsample_factor)
        self.align_corners = align_corners

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.SyncBatchNorm(out_c),
                nn.ReLU(inplace=True)
            )

        # 3 stages: conv -> 2x upsample
        self.stage1 = block(in_channels, mid_channels)
        self.stage2 = block(mid_channels, mid_channels)
        self.stage3 = block(mid_channels, mid_channels)

        # final conv stack (no immediate 2x), then classifier
        self.final_conv = block(mid_channels, mid_channels)
        self.classifier = nn.Conv2d(mid_channels, num_classes, kernel_size=1, bias=True)

    def forward(self, fmap):
        # fmap: (N, C, H', W'), where H' = H/patch, W' = W/patch
        h0, w0 = fmap.shape[-2:]                   # H', W'
        target_h, target_w = h0 * self.patch, w0 * self.patch  # -> H, W

        x = self.stage1(fmap)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=self.align_corners)  # *2

        x = self.stage2(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=self.align_corners)  # *4

        x = self.stage3(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=self.align_corners)  # *8

        x = self.final_conv(x)
        x = self.classifier(x)

        # final jump to exact (H, W); handles the leftover factor (e.g., 14/8 = 1.75)
        x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=self.align_corners)
        return x
