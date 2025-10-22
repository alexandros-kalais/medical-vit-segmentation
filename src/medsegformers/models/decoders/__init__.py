from dataclasses import dataclass
from typing import Dict, Type

@dataclass(frozen=True)
class DecoderSpec:
    name: str
    cls: Type
    input_kind: str = "single"

_DECODER_REGISTRY: Dict[str, DecoderSpec] = {}

def register(name: str = None, input_kind: str = "single"):

    def _decorator(cls):
        key = (name or cls.__name__).lower()
        if key in _DECODER_REGISTRY and _DECODER_REGISTRY[key].cls is not cls:
            raise ValueError(f"Decoder '{key}' already registered")

        _DECODER_REGISTRY[key] = DecoderSpec(name=key, cls=cls, input_kind=input_kind)

        setattr(cls, "DECODER_NAME", key)
        setattr(cls, "INPUT_KIND", input_kind)
        return cls
    return _decorator

def build(name: str, **kwargs):
    key = name.lower()
    if key not in _DECODER_REGISTRY:
        avail = ", ".join(sorted(_DECODER_REGISTRY.keys())) or "<none>"
        raise KeyError(f"Unknown decoder '{name}'. Available: {avail}")
    return _DECODER_REGISTRY[key].cls(**kwargs)

def get_decoder_info(name: str) -> DecoderSpec:
    return _DECODER_REGISTRY[name.lower()]

from .linear_head import LinearHead
from .setr_naive_head import NaiveHead
from .setr_pup_head import PUPHead
from .setr_mla_head import MLAHead
from .segmenter_transformer_head import MaskTransformerHead
