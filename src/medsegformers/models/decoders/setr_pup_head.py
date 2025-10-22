import torch.nn as nn
import torch.nn.functional as F
from . import register

@register("pup", input_kind="single")
class PUPHead(nn.Module):

    def __init__(self, in_channels: int, num_classes: int,
                 upsample_factor: int,  
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

        self.stage1 = block(in_channels, mid_channels)
        self.stage2 = block(mid_channels, mid_channels)
        self.stage3 = block(mid_channels, mid_channels)
        self.final_conv = block(mid_channels, mid_channels)
        self.classifier = nn.Conv2d(mid_channels, num_classes, kernel_size=1, bias=True)

    def forward(self, fmap):

        h0, w0 = fmap.shape[-2:]                
        target_h, target_w = h0 * self.patch, w0 * self.patch  

        x = self.stage1(fmap)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=self.align_corners)  

        x = self.stage2(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=self.align_corners) 

        x = self.stage3(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=self.align_corners)  

        x = self.final_conv(x)
        x = self.classifier(x)

        x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=self.align_corners)
        return x
