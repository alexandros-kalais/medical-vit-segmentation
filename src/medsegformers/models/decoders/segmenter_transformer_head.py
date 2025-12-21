import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from einops import rearrange
from . import register

def init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)

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
        q, k, v = qkv[0], qkv[1], qkv[2]               
        attn = (q @ k.transpose(-2, -1)) * self.scale  
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  
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

@register("masktfm", input_kind="single")
class MaskTransformerHead(nn.Module):
    def __init__(
        self,
        num_classes: int,
        upsample_factor: int = 16,
        in_channels: int = 384,
        n_layers: int = 2,
        n_heads: int = None,
        d_model: int = None,
        d_ff: int | None = None,
        drop_path_rate=0.0,
        dropout= 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.upsample_factor = upsample_factor
        self.n_layers = n_layers
        self.num_classes = num_classes
        self.d_model = int(self.in_channels if d_model is None else d_model)
        self.n_heads = int((self.d_model // 64) if n_heads is None else n_heads)
        self.d_ff = 4 * self.d_model
        self.scale = self.d_model ** -0.5

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
            [Block(self.d_model, self.n_heads, self.d_ff, dropout, dpr[i]) for i in range(n_layers)]
        )

        self.cls_emb = nn.Parameter(torch.randn(1, num_classes, self.d_model))
        self.proj_dec = nn.Linear(in_channels, self.d_model)

        self.proj_patch = nn.Parameter(self.scale * torch.randn(self.d_model, self.d_model))
        self.proj_classes = nn.Parameter(self.scale * torch.randn(self.d_model, self.d_model))

        self.decoder_norm = nn.LayerNorm(self.d_model)
        self.mask_norm = nn.LayerNorm(num_classes)

        self.apply(init_weights)
        trunc_normal_(self.cls_emb, std=0.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_emb"}

    def forward(self, feat):
        B, C, Hp, Wp = feat.shape

        H = Hp * self.upsample_factor
        W = Wp * self.upsample_factor

        x = feat.flatten(2).transpose(1, 2)           
        x = self.proj_dec(x)
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = torch.cat((x, cls_emb), 1)
        for blk in self.blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        patches, cls_seg_feat = x[:, : -self.num_classes], x[:, -self.num_classes :] 
        patches = patches @ self.proj_patch
        cls_seg_feat = cls_seg_feat @ self.proj_classes
        masks = (patches @ cls_seg_feat.transpose(1, 2)) * self.scale
        masks = rearrange(masks, "b (h w) n -> b n h w", h=Hp, w=Wp)
        masks = F.interpolate(masks, size=(H, W), mode="bilinear")

        return masks

    def get_attention_map(self, feat, layer_id):
        if layer_id >= self.n_layers or layer_id < 0:
            raise ValueError(
                f"Provided layer_id: {layer_id} is not valid. 0 <= {layer_id} < {self.n_layers}."
            )
        B, C, Hp, Wp = feat.shape
        x = feat.flatten(2).transpose(1, 2)  
        x = self.proj_dec(x)
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = torch.cat((x, cls_emb), 1)
        for i, blk in enumerate(self.blocks):
            if i < layer_id:
                x = blk(x)
            else:
                return blk(x, return_attention=True)
