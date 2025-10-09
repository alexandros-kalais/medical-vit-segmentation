import torch
import torch.nn as nn
import torch.nn.functional as F
from . import register

@register("mla", input_kind="multi")
class MLAHead(nn.Module):
    """
    SETR-MLA (Official Implementation Style)
    
    inputs: List[Tensor] of length M, each (N, C_in, H', W')  # same H', W' for all streams
    Steps (per stream):
      1) 1x1 conv (C_in -> mla_channels)
      2) top-down elementwise add across streams after step 1
      3) TWO 3x3 convs: mla_channels -> mlahead_channels -> mlahead_channels
      4) upsample x4  -> (H/4, W/4)
    Fuse:
      concat streams -> 3x3 classifier -> upsample x4 -> (H, W)
    """

    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 upsample_factor: int,
                 num_levels: int = 4,
                 align_corners: bool = False):
        super().__init__()
        self.patch = int(upsample_factor)
        self.num_levels = int(num_levels)
        self.align_corners = align_corners
        mla_channels = in_channels // 2
        mlahead_channels = in_channels // 4
        # Stage 1: 1x1 reduction per stream
        self.reduce_1x1 = nn.ModuleList([
            nn.Conv2d(in_channels, mla_channels, kernel_size=1, bias=False)
            for _ in range(self.num_levels)
        ])
        self.bn_reduce = nn.ModuleList([
            nn.SyncBatchNorm(mla_channels) for _ in range(self.num_levels)
        ])

        # Stage 2: Two 3x3 conv refinement per stream (matching official repo)
        # Pattern: 256->128->128 (reduce then refine)
        self.refine_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(mla_channels, mla_channels, kernel_size=3, padding=1, bias=False),  # keep
                nn.SyncBatchNorm(mla_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mla_channels, mlahead_channels, kernel_size=3, padding=1, bias=False),  # reduce
                nn.SyncBatchNorm(mlahead_channels),
                nn.ReLU(inplace=True),
            )
            for _ in range(self.num_levels)
        ])

        # Final classifier: 3x3 conv (matching official repo, not 1x1)
        self.classifier = nn.Conv2d(
            self.num_levels * mlahead_channels, 
            num_classes, 
            kernel_size=3, 
            padding=1, 
            bias=True
        )

    def forward(self, feats):
        assert isinstance(feats, (list, tuple)) and len(feats) == self.num_levels
        N, _, Hp, Wp = feats[0].shape
        for f in feats:
            assert f.shape[-2:] == (Hp, Wp), "All streams must have identical H', W'"

        H, W = Hp * self.patch, Wp * self.patch
        H4, W4 = H // 4, W // 4

        # Step 1: 1x1 + BN + ReLU per stream
        red = []
        for i in range(self.num_levels):
            x = self.reduce_1x1[i](feats[i])
            x = F.relu(self.bn_reduce[i](x), inplace=True)
            red.append(x)

        # Step 2: Top-down aggregation (element-wise add)
        for i in range(self.num_levels - 2, -1, -1):
            red[i] = red[i] + red[i + 1]

        # Step 3: Two 3x3 convs per stream + upsample to (H/4, W/4)
        ups = []
        for i in range(self.num_levels):
            x = self.refine_heads[i](red[i])  # Two 3x3 convs: 256->128->128
            x = F.interpolate(x, size=(H4, W4), mode="bilinear", align_corners=self.align_corners)
            ups.append(x)

        # Step 4: Concatenate all streams
        fused = torch.cat(ups, dim=1)  # [B, 4*128=512, H/4, W/4]

        # Step 5: Final 3x3 classifier + upsample to full resolution
        logits = self.classifier(fused)
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=self.align_corners)
        
        return logits