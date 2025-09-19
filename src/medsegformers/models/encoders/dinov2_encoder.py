import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model

def _to_hw(n: int) -> tuple[int, int]:
    s = int(math.sqrt(n))
    assert s * s == n, f"N={n} is not a perfect square; ensure H,W are multiples of patch_size"
    return s, s

def _resize_pos_embed(pos_embed: torch.Tensor, new_hw: tuple[int,int]) -> torch.Tensor:
    """pos_embed: (1, 1+N, D) with CLS at idx 0  →  returns (1, 1+H'*W', D)"""
    cls_pos = pos_embed[:, :1, :]
    grid_pos = pos_embed[:, 1:, :]
    old_h, old_w = _to_hw(grid_pos.shape[1])
    D = grid_pos.shape[-1]
    grid_pos = grid_pos.reshape(1, old_h, old_w, D).permute(0,3,1,2)         # (1,D,old_h,old_w)
    grid_pos = F.interpolate(grid_pos, size=new_hw, mode="bicubic", align_corners=False)
    grid_pos = grid_pos.permute(0,2,3,1).reshape(1, new_hw[0]*new_hw[1], D)  # (1,H'*W',D)
    return torch.cat([cls_pos, grid_pos], dim=1)

class DINOv2Encoder(nn.Module):
    """
    Generic ViT/DINOv2 encoder (timm) that can return:
      - a single fmap (B, D, H', W')  [return_intermediate=False]
      - a list of fmaps at selected block indices  [return_intermediate=True]
    """
    def __init__(
        self,
        vit_name: str = "vit_base_patch16_224",  # or "vit_base_patch14_dinov2"
        pretrained: bool = True,
        freeze: bool = True,
        return_intermediate: bool = False,
        mla_indices: tuple[int, ...] = (5, 11, 17, 23),
    ):
        super().__init__()
        self.vit = create_model(vit_name, pretrained=pretrained)


        # Remove the fixed size constraint from the patch embed
        if hasattr(self.vit.patch_embed, 'img_size'):
            self.vit.patch_embed.img_size = None 


        # infer patch size + embed dim
        ps = self.vit.patch_embed.patch_size
        self._patch = ps if isinstance(ps, int) else ps[0]
        self._embed = getattr(self.vit, "embed_dim", getattr(self.vit, "num_features", None))
        assert self._embed is not None, "Could not infer ViT embed_dim"

        self.return_intermediate = return_intermediate
        self.mla_indices = tuple(sorted(mla_indices))

        if freeze:
            for p in self.vit.parameters():
                p.requires_grad = False
            self.vit.eval()

    @property
    def patch_size(self) -> int:
        return self._patch

    @property
    def embed_dim(self) -> int:
        return self._embed

    @torch.no_grad()
    def _forward_tokens(self, x: torch.Tensor):
        # patch embeddings to determine Hp, Wp for this input
        tok = self.vit.patch_embed(x)                 # (B, N, D), N = Hp*Wp
        B, N, D = tok.shape
        Hp = x.shape[-2] // self._patch              # height/patch
        Wp = x.shape[-1] // self._patch              # width/patch

        # add cls + (resized) pos emb
        cls_tok = self.vit.cls_token.expand(B, -1, -1)  # (B,1,D)
        z = torch.cat((cls_tok, tok), dim=1)            # (B, 1+N, D)
        pos = self.vit.pos_embed
        if pos.shape[1] != z.shape[1]:
            pos = _resize_pos_embed(pos, (Hp, Wp))
        z = self.vit.pos_drop(z + pos)

        # run blocks
        outs = []
        for i, blk in enumerate(self.vit.blocks):
            z = blk(z)
            if self.return_intermediate and i in self.mla_indices:
                zi = self.vit.norm(z)[:, 1:, :]                            # drop CLS
                fmap = zi.transpose(1, 2).reshape(B, self._embed, Hp, Wp)  # (B,D,Hp,Wp)
                outs.append(fmap)

        if self.return_intermediate:
            assert len(outs) == len(self.mla_indices), "Check mla_indices vs ViT depth"
            return outs, Hp, Wp
        else:
            z = self.vit.norm(z)[:, 1:, :]                                  # (B,N,D)
            fmap = z.transpose(1, 2).reshape(B, self._embed, Hp, Wp)        # (B,D,Hp,Wp)
            return fmap, Hp, Wp

    def forward(self, x: torch.Tensor):
        return self._forward_tokens(x)[0]  # return just feats (single fmap or list of fmaps)
