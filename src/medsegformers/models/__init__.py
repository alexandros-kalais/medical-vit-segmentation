_REGISTRY = {}

def register(cls):
    """Decorator with no nested function: class must have MODEL_NAME attr."""
    name = getattr(cls, "MODEL_NAME", cls.__name__).lower()
    if name in _REGISTRY and _REGISTRY[name] is not cls:
        raise ValueError(f"Model '{name}' already registered")
    _REGISTRY[name] = cls
    return cls

def build(name: str, **kwargs):
    key = name.lower()
    if key not in _REGISTRY:
        avail = ", ".join(sorted(_REGISTRY.keys())) or "<none>"
        raise KeyError(f"Unknown model '{name}'. Available: {avail}")
    return _REGISTRY[key](**kwargs)


__all__ = ["register", "build"]


from . import unet
from . import vit_linear