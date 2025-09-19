import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_

from . import register

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout, out_dim=None):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        if out_dim is None:
            out_dim = dim
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        head_dim = dim // heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.heads, C // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]               # (B, h, N, d)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, h, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class Block(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout, drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, mask=None, return_attention=False):
        y, attn = self.attn(self.norm1(x), mask)
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# ---------------------------
# Segmenter-style Mask Transformer Head
# ---------------------------

@register("masktfm", input_kind="single")
class MaskTransformerHead(nn.Module):
    """
    Segmenter-style transformer decoder head using your Block/Attention/FFN.

    Input:
      fmap: (B, C_in, H', W') from encoder (single fmap)

    Output:
      logits: (B, K, H, W)
    """
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        upsample_factor: int,
        *,
        depth: int = 2,
        num_heads: int = 8,
        d_model: int | None = None,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        normalize_dot: bool = True,
        align_corners: bool = False,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.patch = int(upsample_factor)
        self.align_corners = align_corners
        self.normalize_dot = bool(normalize_dot)

        dim_in = int(in_channels)
        dim = int(d_model) if d_model is not None else dim_in

        # project encoder width -> decoder width (paper: proj_dec)
        self.proj_dec = nn.Linear(dim_in, dim) if dim_in != dim else nn.Identity()

        # class embeddings (paper: cls_emb)
        self.cls_emb = nn.Parameter(torch.zeros(1, self.num_classes, dim))
        trunc_normal_(self.cls_emb, std=0.02)

        # transformer blocks with linear drop-path schedule
        dprs = torch.linspace(0, drop_path_rate, steps=depth).tolist() if depth > 0 else []
        self.blocks = nn.ModuleList([
            Block(dim=dim,
                  heads=num_heads,
                  mlp_dim=int(dim * mlp_ratio),
                  dropout=dropout,
                  drop_path=dprs[i] if i < len(dprs) else 0.0)
            for i in range(depth)
        ])
        self.decoder_norm = nn.LayerNorm(dim)

        # learned projections before similarity (paper: proj_patch/classes)
        scale = dim ** -0.5
        self.proj_patch = nn.Parameter(scale * torch.randn(dim, dim))
        self.proj_classes = nn.Parameter(scale * torch.randn(dim, dim))

        # LayerNorm over class dimension for masks (B, N, K)
        self.mask_norm = nn.LayerNorm(self.num_classes)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_emb"}

    def _flatten(self, fmap: torch.Tensor):
        # (B, C, H', W') -> (B, N, C), N = H'*W'
        B, C, Hp, Wp = fmap.shape
        tokens = fmap.flatten(2).transpose(1, 2)
        return tokens, Hp, Wp

    def forward(self, fmap: torch.Tensor) -> torch.Tensor:
        """
        fmap: (B, C_in, H', W')
        returns: (B, K, H, W)
        """
        B, _, Hp, Wp = fmap.shape
        H, W = Hp * self.patch, W * self.patch if (W := Wp) else None  # small trick to keep Wp

        # 1) flatten + 2) project
        z, Hp_, Wp_ = self._flatten(fmap)          # (B, N, C_in)
        assert (Hp_, Wp_) == (Hp, Wp)
        x = self.proj_dec(z)                       # (B, N, dim)

        # 3) concat [patch tokens, class tokens]
        cls = self.cls_emb.expand(B, -1, -1)       # (B, K, dim)
        x = torch.cat([x, cls], dim=1)             # (B, N+K, dim)

        # 4) transformer stack
        for blk in self.blocks:
            x = blk(x)
        x = self.decoder_norm(x)                   # (B, N+K, dim)

        # 5) split back
        N = Hp * Wp
        patches = x[:, :N, :]                      # (B, N, dim)
        cls_out = x[:, N:, :]                      # (B, K, dim)

        # 6) learned projections
        patches = patches @ self.proj_patch        # (B, N, dim)
        cls_out = cls_out @ self.proj_classes      # (B, K, dim)

        # 7) L2 normalization before dot product
        if self.normalize_dot:
            patches = F.normalize(patches, dim=-1)
            cls_out = F.normalize(cls_out, dim=-1)

        # 8) similarity -> (B, N, K), then LayerNorm across classes
        masks = patches @ cls_out.transpose(1, 2)  # (B, N, K)
        masks = self.mask_norm(masks)

        # 9) reshape to (B, K, H', W') and upsample to (H, W)
        masks = masks.transpose(1, 2).reshape(B, self.num_classes, Hp, Wp)
        masks_up = F.interpolate(masks, size=(H, W), mode="bilinear", align_corners=self.align_corners)
        return masks_up

    @torch.no_grad()
    def get_attention_map(self, fmap: torch.Tensor, layer_id: int):
        """
        Return attention weights from a specific block.
        """
        if layer_id < 0 or layer_id >= len(self.blocks):
            raise ValueError(f"layer_id {layer_id} out of range [0, {len(self.blocks)})")

        B, _, Hp, Wp = fmap.shape
        z, _, _ = self._flatten(fmap)              # (B, N, C_in)
        x = self.proj_dec(z)                       # (B, N, dim)
        cls = self.cls_emb.expand(B, -1, -1)       # (B, K, dim)
        x = torch.cat([x, cls], dim=1)             # (B, N+K, dim)

        for i, blk in enumerate(self.blocks):
            if i < layer_id:
                x = blk(x)
            else:
                attn = blk(x, return_attention=True)  # (B, heads, N+K, N+K)
                return attn
