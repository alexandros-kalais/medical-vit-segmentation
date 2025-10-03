import torch
import torch.nn as nn
import torch.nn.functional as F
from . import register

@register("mla", input_kind="multi")
class MLAHead(nn.Module):
    """
    SETR-MLA
    inputs: List[Tensor] of length M, each (N, C_in, H', W')  # same H', W' for all streams
    Steps (per stream):
      1) 1x1 conv (C_in -> mla_channels)
      2) top-down elementwise add across streams after step 1
      3) 3x3 conv (mla_channels -> mla_channels)
      4) 3x3 conv (mla_channels -> mlahead_channels)   # halved again
      5) upsample x4  -> (H/4, W/4)
    Fuse:
      concat streams -> 1x1 classifier -> upsample x4 -> (H, W)
    """

    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 upsample_factor: int,
                 mla_channels: int = 256,
                 mlahead_channels: int = 128,
                 num_levels: int = 4,
                 align_corners: bool = False):
        super().__init__()
        self.patch = int(upsample_factor)
        self.num_levels = int(num_levels)
        self.align_corners = align_corners

        # per-stream conv stacks
        self.reduce_1x1 = nn.ModuleList([
            nn.Conv2d(in_channels, mla_channels, kernel_size=1, bias=False)
            for _ in range(self.num_levels)
        ])
        self.bn_reduce = nn.ModuleList([nn.SyncBatchNorm(mla_channels) for _ in range(self.num_levels)])

        self.after_add_3x3 = nn.ModuleList([
            nn.Conv2d(mla_channels, mla_channels, kernel_size=3, padding=1, bias=False)
            for _ in range(self.num_levels)
        ])
        self.bn_after = nn.ModuleList([nn.SyncBatchNorm(mla_channels) for _ in range(self.num_levels)])

        self.to_head_3x3 = nn.ModuleList([
            nn.Conv2d(mla_channels, mlahead_channels, kernel_size=3, padding=1, bias=False)
            for _ in range(self.num_levels)
        ])
        self.bn_head = nn.ModuleList([nn.SyncBatchNorm(mlahead_channels) for _ in range(self.num_levels)])

        self.classifier = nn.Conv2d(self.num_levels * mlahead_channels, num_classes, kernel_size=1, bias=True)

    def forward(self, feats):
        assert isinstance(feats, (list, tuple)) and len(feats) == self.num_levels
        N, _, Hp, Wp = feats[0].shape
        for f in feats:
            assert f.shape[-2:] == (Hp, Wp), "All streams must have identical H', W'"

        H, W = Hp * self.patch, Wp * self.patch
        H4, W4 = H // 4, W // 4

        #1x1 + BN + ReLU per stream
        red = []
        for i in range(self.num_levels):
            x = self.reduce_1x1[i](feats[i])
            x = F.relu(self.bn_reduce[i](x), inplace=True)
            red.append(x)

        #top-down adds
        for i in range(self.num_levels - 2, -1, -1):
            red[i] = red[i] + red[i + 1]

        #3x3 -> 3x3 -> upsample to (H/4, W/4)
        ups = []
        for i in range(self.num_levels):
            x = self.after_add_3x3[i](red[i])
            x = F.relu(self.bn_after[i](x), inplace=True)
            x = self.to_head_3x3[i](x)
            x = F.relu(self.bn_head[i](x), inplace=True)
            x = F.interpolate(x, size=(H4, W4), mode="bilinear", align_corners=self.align_corners)
            ups.append(x)

        fused = torch.cat(ups, dim=1)
        logits = self.classifier(fused)
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=self.align_corners)
        return logits

