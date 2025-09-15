# medsegformers/models/vit_linear.py
import torch.nn as nn
from . import register
from .encoders.vit_encoder import ViTEncoder
from .decoders.linear_head import LinearHead

@register
class ViT_Linear(nn.Module):
    """
    Plain ViT encoder + linear segmentation head (EoMT-style).
    """
    MODEL_NAME = "vit_linear"

    def __init__(
        self,
        # required by your trainer's build() call
        in_channels: int = 3,          # ignored but kept for API compat
        out_channels: int = 2,
        # model-specific
        vit_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        freeze_encoder: bool = True,
        img_size: tuple[int, int] = (224, 224),
    ):
        super().__init__()
        self.encoder = ViTEncoder(vit_name=vit_name, pretrained=pretrained, freeze=freeze_encoder)
        up = self.encoder.patch_size
        self.head = LinearHead(in_channels=self.encoder.embed_dim, num_classes=out_channels, upsample_factor=up)

    def forward(self, x):
        fmap = self.encoder(x)  # (B, D, H', W')
        logits = self.head(fmap)
        return logits
