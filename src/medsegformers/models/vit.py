from typing import Optional
import torch
import torch.nn as nn
import timm
from transformers import AutoModel
import inspect



class ViT(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int],
        patch_size=16,
        backbone_name="vit_large_patch14_reg4_dinov2",
        ckpt_path: Optional[str] = None,
    ):
        super().__init__()

        if "/" in backbone_name:
            self.backbone = self.transformers_to_timm(
                AutoModel.from_pretrained(
                    backbone_name,
                ),
                img_size,
            )
        else:
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=ckpt_path is None,
                img_size=img_size,
                patch_size=patch_size,
                num_classes=0,
            )

            self.backbone.patch_size = patch_size
            self._orig_gil = self.backbone.get_intermediate_layers
            self.backbone.get_intermediate_layers = self._get_intermediate_layers_timm

        pixel_mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
        pixel_std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)

        self.register_buffer("pixel_mean", pixel_mean)
        self.register_buffer("pixel_std", pixel_std)
    
    def _get_intermediate_layers_timm(self, x, n, **_):
        return self._orig_gil(x, n, return_prefix_tokens=True)

    def transformers_to_timm(self, backbone, img_size: tuple[int, int]):
        backbone.patch_embed = backbone.embeddings
        backbone.patch_embed.patch_size = (
            backbone.embeddings.config.patch_size,
            backbone.embeddings.config.patch_size,
        )
        backbone.patch_size = backbone.embeddings.config.patch_size
        backbone.patch_embed.grid_size = (
            img_size[0] // backbone.embeddings.config.patch_size,
            img_size[1] // backbone.embeddings.config.patch_size,
        )

        backbone.embed_dim = backbone.embeddings.config.hidden_size
        backbone.num_prefix_tokens = backbone.patch_embed.config.num_register_tokens + 1
        backbone.blocks = backbone.layer
        backbone.get_intermediate_layers = self._get_intermediate_layers_hf

        del (
            backbone.patch_embed.mask_token
        )

        return backbone

    def _get_intermediate_layers_hf(self, x, n, return_class_token=True, **kwargs):
        
        out = self.backbone(pixel_values=x, output_hidden_states=True)
        hidden = out.hidden_states
        idxs = list(n)
        picks = []
        npt = self.backbone.num_prefix_tokens
        for idx in idxs:
            hs = hidden[idx]
            cls_tok = hs[:, :1, :]
            patch_tok = hs[:, npt:, :]
            picks.append((patch_tok, cls_tok) if return_class_token else patch_tok)
        return picks
