import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model

def _resize_pos_embed(pos_embed: torch.Tensor, new_hw: tuple[int,int]) -> torch.Tensor:
    cls = pos_embed[:, :1, :]
    grid = pos_embed[:, 1:, :]
    n, d = grid.shape[1], grid.shape[2]
    h = w = int(n ** 0.5)
    grid = grid.reshape(1, h, w, d).permute(0,3,1,2)
    grid = F.interpolate(grid, size=new_hw, mode="bicubic", align_corners=False)
    grid = grid.permute(0,2,3,1).reshape(1, new_hw[0]*new_hw[1], d)
    return torch.cat([cls, grid], dim=1)

class DINOv2Encoder(nn.Module):
    """
    ViT DINOv2 encoder wrapper that returns a feature map (B, D, H', W').
    Default: 'vit_base_patch14_dinov2'
    """
    def __init__(self, vit_name: str = "vit_base_patch14_dinov2", pretrained: bool = True, freeze: bool = True, image_size: int = 518):
        super().__init__()
        self.vit = create_model(vit_name, pretrained=pretrained, img_size=image_size)
        ps = self.vit.patch_embed.patch_size
        self._patch = ps if isinstance(ps, int) else ps[0]
        self._embed = getattr(self.vit, "embed_dim", getattr(self.vit, "num_features", None))
        assert self._embed is not None, "Could not infer embed_dim"

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
        tok = self.vit.patch_embed(x)   # (B,N,D)
        B, N, D = tok.shape
        Hp = x.shape[-2] // self._patch
        Wp = x.shape[-1] // self._patch

        cls_tok = self.vit.cls_token.expand(B, -1, -1)
        z = torch.cat((cls_tok, tok), dim=1)

        if self.vit.pos_embed.shape[1] != z.shape[1]:
            pos = _resize_pos_embed(self.vit.pos_embed, (Hp, Wp))
        else:
            pos = self.vit.pos_embed
        z = self.vit.pos_drop(z + pos)

        for blk in self.vit.blocks:
            z = blk(z)
        z = self.vit.norm(z)
        return z, Hp, Wp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, Hp, Wp = self._forward_tokens(x)  # (B,1+N,D)
        z = z[:, 1:, :]
        fmap = z.transpose(1, 2).reshape(x.size(0), self.embed_dim, Hp, Wp)
        return fmap
