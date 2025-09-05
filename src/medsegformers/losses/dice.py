import torch.nn as nn
from monai.losses import DiceLoss

class FlexDiceLoss(nn.Module):
    """
    Pure Dice loss (MONAI) that adapts to binary or multi-class.
    """
    def __init__(self, num_classes: int, reduction: str = "mean"):
        super().__init__()
        #include background for binary, exclude for multi-class
        include_background = (num_classes == 1)

        self.loss = DiceLoss(
            sigmoid=(num_classes == 1),
            softmax=(num_classes > 1),
            to_onehot_y=(num_classes > 1),
            include_background=include_background,
            reduction=reduction,
        )

    def forward(self, logits, targets):

        return self.loss(logits, targets)
