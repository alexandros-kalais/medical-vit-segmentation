from typing import Dict, Optional, Tuple, Union, List
import torch
import torch.nn as nn

from .vit import ViT

from medsegformers.external import DINOv3_Adapter
from medsegformers.external import MSDeformAttnPixelDecoder as PixelDecoder
from medsegformers.external import MultiScaleMaskedTransformerDecoder


class Mask2FormerModel(nn.Module):

    def __init__(
        self,
        encoder: ViT,
        *,
        num_classes: int,
        adapter_kwargs: Optional[Dict] = None,
        pixel_decoder_kwargs: Optional[Dict] = None,
        transformer_kwargs: Optional[Dict] = None,
        masked_attn_enabled: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes

        self._adapter_kwargs = {
            "backbone": self.encoder.backbone,
            "deform_num_heads": 8,
            "n_points": 4,
            "with_cp": False,
            "add_vit_feature": True,
            "deform_ratio": 0.5,
            "drop_path_rate": 0.0,
        }

        if len(encoder.backbone.blocks) > 12:
            self._adapter_kwargs["interaction_indexes"] = (5, 11, 17, 23)
        else:
            self._adapter_kwargs["interaction_indexes"] = (1, 4, 7, 11)

        if adapter_kwargs:
            self._adapter_kwargs.update(adapter_kwargs)

        self._pixel_decoder_kwargs = {
            "transformer_in_features": ["1", "2", "3"],  
            "transformer_dropout": 0.0,
            "transformer_nheads": 8,
            "transformer_dim_feedforward": 2048,
            "transformer_enc_layers": 6,
            "conv_dim": None,
            "mask_dim": None,
            "common_stride": 4,
        }
        if pixel_decoder_kwargs:
            self._pixel_decoder_kwargs.update(pixel_decoder_kwargs)

        self._transformer_kwargs = {
            "mask_classification": True,
            "num_classes": self.num_classes,
            "num_queries": 100,
            "nheads": 8,
            "dim_feedforward": 2048,
            "dec_layers": 9,
            "pre_norm": False,
            "enforce_input_project": False,
        }
        if transformer_kwargs:
            self._transformer_kwargs.update(transformer_kwargs)

        self.adapter = DINOv3_Adapter(**self._adapter_kwargs)

        self.pixel_decoder: Optional[PixelDecoder] = None
        self.transformer_decoder: Optional[MultiScaleMaskedTransformerDecoder] = None
        self._built = False

        self.masked_attn_enabled = masked_attn_enabled
        self.num_blocks = 1
        self.attn_mask_probs = nn.ParameterList([nn.Parameter(torch.tensor(1.0), requires_grad=False)])

    @staticmethod
    def _infer_input_shape(feats: Dict[str, torch.Tensor]) -> Dict[str, Tuple[int, int, int, int]]:
        input_shape = {
        "1": (feats["1"].shape[1], feats["1"].shape[-2], feats["1"].shape[-1], 4),
        "2": (feats["2"].shape[1], feats["2"].shape[-2], feats["2"].shape[-1], 8),
        "3": (feats["3"].shape[1], feats["3"].shape[-2], feats["3"].shape[-1], 16),
        "4": (feats["4"].shape[1], feats["4"].shape[-2], feats["4"].shape[-1], 32),
        }
        return input_shape

    def _build_heads_from_feats(self, feats: Dict[str, torch.Tensor], device: torch.device) -> None:
        """
        Instantiate PixelDecoder and TransformerDecoder given one adapter pass.
        """
       
        EMB = feats["1"].shape[1]

        input_shape = self._infer_input_shape(feats)

        pd_args = dict(self._pixel_decoder_kwargs)
        pd_args["input_shape"] = input_shape
        pd_args["conv_dim"] = EMB
        pd_args["mask_dim"] = EMB

        self.pixel_decoder = PixelDecoder(**pd_args).to(device)

        td_args = dict(self._transformer_kwargs)
        td_args["in_channels"] = EMB
        td_args["hidden_dim"] = EMB
        td_args["mask_dim"] = EMB

        self.transformer_decoder = MultiScaleMaskedTransformerDecoder(**td_args).to(device)


        self._built = True

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:

        amp_enabled = torch.is_autocast_enabled()

        with torch.autocast(device_type=x.device.type, enabled=False):
            feats = self.adapter(x.float())

            if not self._built:
                self._build_heads_from_feats(feats, device=x.device)

            mask_features, _memory, multi_scale = self.pixel_decoder.forward_features(feats)

        if amp_enabled:
            target_dtype = x.dtype
            mask_features = mask_features.to(target_dtype)
            multi_scale   = [t.to(target_dtype) for t in multi_scale]

        out = self.transformer_decoder(multi_scale, mask_features, mask=None)

        pred_logits = out["pred_logits"] if isinstance(out, dict) else out[0]
        pred_masks  = out["pred_masks"]  if isinstance(out, dict) else out[1]

        return [pred_masks], [pred_logits]


