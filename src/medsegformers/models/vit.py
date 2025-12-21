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
            # Check if ckpt_path points to a .pth file (raw weights)
            if ckpt_path and ckpt_path.endswith('.pth'):
                
                # Create model from pretrained config (downloads config only)
                hf_model = AutoModel.from_pretrained(backbone_name)
                
                # Load the custom weights
                state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)

                # Handle different state dict formats
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                elif 'model' in state_dict:
                    state_dict = state_dict['model']

                # Fix key naming to match HuggingFace format
                new_state_dict = {}
                for key, value in state_dict.items():
                    new_key = key
                    
                    # Remove 'backbone.' prefix if present
                    if new_key.startswith('backbone.'):
                        new_key = new_key.replace('backbone.', '', 1)
                    
                    # Rename to match HF naming convention
                    new_key = new_key.replace('patch_embed.', 'embeddings.')
                    new_key = new_key.replace('blocks.', 'layer.')
                    
                    # Skip pixel_mean and pixel_std (you register these separately)
                    if new_key not in ['pixel_mean', 'pixel_std']:
                        new_state_dict[new_key] = value
                
                # Load weights into the HF model
                missing_keys, unexpected_keys = hf_model.load_state_dict(new_state_dict, strict=False)
                
                if missing_keys:
                    print(f"[INFO] Missing keys when loading checkpoint: {missing_keys}")
                if unexpected_keys:
                    print(f"[INFO] Unexpected keys when loading checkpoint: {unexpected_keys}")

                # After loading, check that weights are different from random initialization
                sample_param = None
                for name, param in hf_model.named_parameters():
                    if 'weight' in name and len(param.shape) >= 2:
                        sample_param = (name, param)
                        break

                if sample_param:
                    name, param = sample_param
                
                # Convert to timm format
                self.backbone = self.transformers_to_timm(hf_model, img_size)
            else:
                # Original path: load from HF hub
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

        # --- Normalization buffers (conditional on checkpoint) ---
        if ckpt_path is not None:
            # Custom stats for checkpointed models SurgeNet
            mean = [0.46888983, 0.29536288, 0.28712815]
            std  = [0.24689102, 0.21034359, 0.21188641]
        else:
            # Default (ImageNet)
            mean = [0.485, 0.456, 0.406]
            std  = [0.229, 0.224, 0.225]
            
        pixel_mean = torch.tensor(mean, dtype=torch.float32).reshape(1, -1, 1, 1)
        pixel_std  = torch.tensor(std,  dtype=torch.float32).reshape(1, -1, 1, 1)

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