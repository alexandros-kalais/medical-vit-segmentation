_REGISTRY = {}

def register(name: str):
    """Decorator to register a model class under a string name."""
    def deco(cls):
        _REGISTRY[name] = cls
        return cls
    return deco

def build(name: str, **kwargs):
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)

__all__ = ["register", "build"]


from . import unet