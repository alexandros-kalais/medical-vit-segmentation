_DECODER_REGISTRY = {}

def register(cls):
    name = getattr(cls, "DECODER_NAME", cls.__name__).lower()
    if name in _DECODER_REGISTRY and _DECODER_REGISTRY[name] is not cls:
        raise ValueError(f"Decoder '{name}' already registered")
    _DECODER_REGISTRY[name] = cls
    return cls

def build(name: str, **kwargs):
    key = name.lower()
    if key not in _DECODER_REGISTRY:
        avail = ", ".join(sorted(_DECODER_REGISTRY.keys())) or "<none>"
        raise KeyError(f"Unknown decoder '{name}'. Available: {avail}")
    return _DECODER_REGISTRY[key](**kwargs)

def list_decoders():
    return sorted(_DECODER_REGISTRY.keys())

__all__ = ["register", "build", "list_decoders"]

# trigger built-ins
from .linear_head import LinearHead  # noqa: F401
