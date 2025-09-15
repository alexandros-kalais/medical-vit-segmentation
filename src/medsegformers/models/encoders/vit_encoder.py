
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
    """
    pos_embed: (1, 1+N, D) with cls at index 0
    new_hw: (H', W') for new grid
    returns: (1, 1+H'*W', D)
    """
    cls_pos = pos_embed[:, :1, :]              # (1,1,D)
    grid_pos = pos_embed[:, 1:, :]             # (1,N,D)
    old_h, old_w = _to_hw(grid_pos.shape[1])
    D = grid_pos.shape[-1]

    grid_pos = grid_pos.reshape(1, old_h, old_w, D).permute(0, 3, 1, 2)  # (1,D,old_h,old_w)
    grid_pos = F.interpolate(grid_pos, size=new_hw, mode="bicubic", align_corners=False)
    grid_pos = grid_pos.permute(0, 2, 3, 1).reshape(1, new_hw[0] * new_hw[1], D)  # (1,H'*W',D)
    return torch.cat([cls_pos, grid_pos], dim=1)

class ViTEncoder(nn.Module):
    """
    Plain ViT encoder wrapper that returns a feature map (B, D, H', W').
    Works with timm ViTs like 'vit_base_patch16_224'.
    """
    def __init__(
        self,
        vit_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        freeze: bool = True,
    ):
        super().__init__()
        self.vit = create_model(vit_name, pretrained=pretrained)
        # infer patch size and embed dim from timm model
        ps = self.vit.patch_embed.patch_size
        self._patch_size = ps if isinstance(ps, int) else ps[0]
        self._embed_dim = getattr(self.vit, "embed_dim", getattr(self.vit, "num_features", None))
        assert self._embed_dim is not None, "Could not infer ViT embed dim"

        if freeze:
            for p in self.vit.parameters():
                p.requires_grad = False
            self.vit.eval()

    @property
    def patch_size(self) -> int:
        return self._patch_size

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @torch.no_grad()
    def _tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """
        Return tokens after transformer (with CLS kept), plus patch grid size (Hp, Wp).
        """
        # get patch embeddings to know Hp,Wp for *this* input size
        tokens = self.vit.patch_embed(x)            # (B, N, D), N = H' * W'

        B, N, D = tokens.shape
        Hp = x.shape[-2] // self._patch_size
        Wp = x.shape[-2] // self._patch_size

        # prepend CLS and add pos embed (interpolated if needed)
        cls_tok = self.vit.cls_token.expand(B, -1, -1)     # (B,1,D)
        tok_plus_cls = torch.cat((cls_tok, tokens), dim=1) # (B, N+1, D)
        # interpolate pos embed if grid differs
        if self.vit.pos_embed.shape[1] != tok_plus_cls.shape[1]:
            pos = _resize_pos_embed(self.vit.pos_embed, (Hp, Wp))
        else:
            pos = self.vit.pos_embed
        z = self.vit.pos_drop(tok_plus_cls + pos)

        # transformer encoder blocks
        for blk in self.vit.blocks:
            z = blk(z)
        z = self.vit.norm(z)                          # (B, N+1, D)
        return z, Hp, Wp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,3,H,W) with H,W multiples of patch_size
        returns: feature map (B, D, H/ps, W/ps)
        """
        z, Hp, Wp = self._tokens(x)       # (B, N+1, D)
        z = z[:, 1:, :]                   # drop CLS → (B, N, D)
        fmap = z.transpose(1, 2).reshape(x.size(0), self.embed_dim, Hp, Wp)
        return fmap



# # 1) Instantiate encoder (frozen, pretrained)
# enc = ViTEncoder(
#     vit_name="vit_base_patch16_224",
#     pretrained=True,
#     freeze=True,
# )
# print(f"patch_size={enc.patch_size}  embed_dim={enc.embed_dim}")

# # 2) Confirm parameters are frozen
# n_total = sum(p.numel() for p in enc.parameters())
# n_trainable = sum(p.numel() for p in enc.parameters() if p.requires_grad)
# print(f"params total={n_total:,}  trainable={n_trainable:,}")
# assert n_trainable == 0, "Encoder should be frozen for the smoke test."

# # 3) Forward on 224×224 (no pos-embed resize needed)
# x224 = torch.randn(2, 3, 224, 224)  # B=2
# with torch.no_grad():
#     fmap224 = enc(x224)  # (B, D, H', W') = (2, 768, 14, 14) for ViT-B/16
# print("fmap224:", tuple(fmap224.shape))
# assert fmap224.shape[1] == enc.embed_dim
# assert fmap224.shape[2] == 224 // enc.patch_size
# assert fmap224.shape[3] == 224 // enc.patch_size