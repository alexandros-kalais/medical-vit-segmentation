import torch.nn as nn
from .dice import FlexDiceLoss
from .dicece import FlexDiceCELoss

def build_loss(kind: str, num_classes: int, *, reduction: str = "mean", **kwargs) -> nn.Module:
    k = kind.lower()
    if k == "dice":
        return FlexDiceLoss(num_classes=num_classes, reduction=reduction, **kwargs)
    if k == "dicece":
        return FlexDiceCELoss(num_classes=num_classes, reduction=reduction, **kwargs)
    raise ValueError(f"Unknown loss '{kind}'. Use: flexdice, flexdicece")
