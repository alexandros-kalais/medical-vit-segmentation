import torch.nn as nn
from medsegformers.models.encoders.dinov2_encoder import DINOv2Encoder
from medsegformers.models.decoders import build as build_decoder

def build_segmentation_model(
    *,
    decoder: str,
    num_classes: int,
    vit_name: str = "vit_base_patch14_dinov2",
    pretrained: bool = True,
    freeze_encoder: bool = True,
    decoder_kwargs: dict | None = None,
    image_size: int = 518
) -> nn.Module:
    enc = DINOv2Encoder(vit_name=vit_name, pretrained=pretrained, freeze=freeze_encoder, image_size=image_size)
    up = enc.patch_size
    in_ch = enc.embed_dim
    dec = build_decoder(decoder, in_channels=in_ch, num_classes=num_classes, upsample_factor=up, **(decoder_kwargs or {}))

    class _SegModel(nn.Module):
        def __init__(self, encoder, decoder):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder
        def forward(self, x):
            fmap = self.encoder(x)
            return self.decoder(fmap)

    return _SegModel(enc, dec)
