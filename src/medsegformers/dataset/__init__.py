from typing import Dict, Type, List
from monai.data import Dataset

_DATASET_REGISTRY: Dict[str, Type[Dataset]] = {}

def register_dataset(cls: Type[Dataset]) -> Type[Dataset]:
    name = getattr(cls, "DATASET_NAME", cls.__name__).lower()
    if name in _DATASET_REGISTRY and _DATASET_REGISTRY[name] is not cls:
        raise ValueError(f"Dataset '{name}' already registered with {_DATASET_REGISTRY[name]}")
    _DATASET_REGISTRY[name] = cls
    return cls

def get_dataset_class(name: str) -> Type[Dataset]:
    key = name.lower()
    if key not in _DATASET_REGISTRY:
        available = ", ".join(sorted(_DATASET_REGISTRY.keys())) or "<none>"
        raise KeyError(f"Unknown dataset {name!r}. Available: {available}")
    return _DATASET_REGISTRY[key]

from .endoscopy import EndoscopyDataset
from .transforms import get_transforms

__all__ = [
    "register_dataset",
    "get_dataset_class",
    "EndoscopyDataset",
    "get_transforms"
]
