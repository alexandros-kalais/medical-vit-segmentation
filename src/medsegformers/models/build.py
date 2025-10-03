from typing import Optional, Tuple
import torch.nn as nn
from medsegformers.models.encoder import Encoder
from medsegformers.models.vit import ViT
from medsegformers.models.decoders import build as build_decoder, get_decoder_info

def build_segmentation_model(
    *,
    decoder: str,  # ["linear", "naive", "pup", "mla", "masktfm"]
    num_classes: int,
    vit_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
    pretrained: bool = True,
    freeze_encoder: bool = True,
    unfreeze_last_k: int = 0,
    decoder_kwargs: Optional[dict] = None,
    indices: Tuple[int, ...] = (1, 4, 7, 12),
    image_size: Tuple[int, int] = (224, 224),
) -> nn.Module:
    # Read decoder information
    spec = get_decoder_info(decoder)

    # Build ViT (don't reference `vit` before it's created)
    vit = ViT(
        img_size=image_size,
        patch_size=16,  # or parse from vit_name if needed
        backbone_name=vit_name,
        ckpt_path=None if pretrained else "",  # if you want manual ckpt loading
    )

    if freeze_encoder:
        n = len(vit.backbone.blocks) - unfreeze_last_k
        for blk in vit.backbone.blocks[:n]:
            for p in blk.parameters():
                p.requires_grad_(False)
            for mod in blk.modules():
                if isinstance(mod, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
                    for p in mod.parameters():
                        p.requires_grad_(False)

    enc = Encoder(
        encoder=vit,
        return_intermediate=(spec.input_kind == "multi"),
        indices=indices,
    )

    # Build decoder
    dec = build_decoder(
        decoder,
        in_channels=vit.backbone.embed_dim,
        num_classes=num_classes,
        upsample_factor=vit.backbone.patch_embed.patch_size[0],
        **(decoder_kwargs or {}),
    )

    class _SegModel(nn.Module):
        def __init__(self, encoder, decoder):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder

        def forward(self, x):
            feats = self.encoder(x)  # fmap or list of fmaps
            return self.decoder(feats)

    return _SegModel(enc, dec)
