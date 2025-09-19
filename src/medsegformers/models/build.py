import torch.nn as nn
from medsegformers.models.encoders.dinov2_encoder import DINOv2Encoder
from medsegformers.models.decoders import build as build_decoder, get_decoder_info

def build_segmentation_model(
    *,
    decoder: str,                  # ["linear", "naive", "pup", "mla", "msktfm"]
    num_classes: int,
    vit_name: str = "vit_base_patch16_224",
    pretrained: bool = True,
    freeze_encoder: bool = True,
    decoder_kwargs: dict | None = None,
    mla_indices: tuple[int, ...] = (2, 5, 8, 11)  # used only if decoder wants multi
) -> nn.Module:

    #Read decoder information
    spec = get_decoder_info(decoder)            # has fields: name, cls, input_kind
    wants_multi = (spec.input_kind == "multi")  # "multi" for MLA; "single" otherwise

    # Configure encoder
    enc = DINOv2Encoder(
        vit_name=vit_name,
        pretrained=pretrained,
        freeze=freeze_encoder,
        return_intermediate=wants_multi,  # encoder should return list of fmaps if True
        mla_indices=mla_indices           # used only when wants_multi=True
    )

    # Build decoder
    dec = build_decoder(
        decoder,
        in_channels=enc.embed_dim,
        num_classes=num_classes,
        upsample_factor=enc.patch_size,         # heads treat this as the ViT patch size
        **(decoder_kwargs or {})
    )

    class _SegModel(nn.Module):
        def __init__(self, encoder, decoder):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder

        def forward(self, x):
            feats = self.encoder(x)  # fmap (linear, naive, pup, masktfm) or list of fmaps (mla)
            return self.decoder(feats)

    return _SegModel(enc, dec)
