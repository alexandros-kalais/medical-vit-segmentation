from typing import Dict, Optional, Tuple, Union, List
import torch
import torch.nn as nn
from .vit import ViT
import torch.nn.functional as F
from .decoders import build as build_decoder, get_decoder_info

class Encoder(nn.Module):

    def __init__(self, encoder: nn.Module,
                 return_intermediate: bool = False,
                 indices: tuple[int, ...] = (5, 11, 17, 23)):
        super().__init__()
        self.encoder = encoder 
        self.return_intermediate = return_intermediate  
        self.indices = indices                   
        patch_size = encoder.backbone.patch_embed.patch_size
        

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        
        x = (x - self.encoder.pixel_mean) / self.encoder.pixel_std

        # 0) optional RoPE handle 
        rope = None
        if hasattr(self.encoder.backbone, "rope_embeddings"):
            rope = self.encoder.backbone.rope_embeddings(x)  # kept for parity; not further used

        # 1) patch embedding → tokens
        x = self.encoder.backbone.patch_embed(x)

        # 2) positional embeddings
        if hasattr(self.encoder.backbone, "_pos_embed"):
            x = self.encoder.backbone._pos_embed(x)

        outputs = []
        for i, block in enumerate(self.encoder.backbone.blocks):
            if rope != None:
                residual = x
                hidden_states = block.norm1(x)
                
                # Call attention with RoPE
                hidden_states, _ = block.attention(
                    hidden_states,
                    attention_mask=None,
                    position_embeddings=rope
                )
                
                if hasattr(block, 'layer_scale1'):
                    hidden_states = block.layer_scale1(hidden_states)
                elif hasattr(block, "ls1"):
                    hidden_states = block.ls1(hidden_states)
                x = residual + hidden_states
                
                residual = x
                hidden_states = block.norm2(x)
                hidden_states = block.mlp(hidden_states)
                if hasattr(block, 'layer_scale2'):
                    hidden_states = block.layer_scale2(hidden_states)
                elif hasattr(block, "ls2"):
                    hidden_states = block.ls2(hidden_states)
                x = residual + hidden_states
            else:
                x = block(x)

            if (self.return_intermediate and i in self.indices) or (i == len(self.encoder.backbone.blocks) - 1):
                x = self.encoder.backbone.norm(x)
                fmap = x[:, self.encoder.backbone.num_prefix_tokens:, :].transpose(1, 2).reshape(
                    x.shape[0], -1, *self.encoder.backbone.patch_embed.grid_size
                )       
                outputs.append(fmap)
        
        if self.return_intermediate:
            return outputs
        else:
            return outputs[0]


class EncDecModel(nn.Module):
    def __init__(
        self,
        vit: ViT,
        *,
        num_classes: int,
        decoder: str,
        decoder_kwargs: Optional[Dict] = None,
    ) -> None:
        super().__init__()


        spec = get_decoder_info(decoder)

        if len(vit.backbone.blocks) > 12:
            indices = (5, 11, 17, 23)
        else:
            indices = (1, 4, 7, 11)

        self.encoder = Encoder(
            encoder=vit,
            return_intermediate=(spec.input_kind == "multi"),
            indices=indices,
        )
        
        embed_dim = vit.backbone.embed_dim
        patch_size = vit.backbone.patch_size

        self.decoder = build_decoder(
            decoder,
            in_channels=embed_dim,
            num_classes=num_classes,
            upsample_factor=patch_size,
            **(decoder_kwargs or {}),
        )

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        feats = self.encoder(x)
        return self.decoder(feats)
