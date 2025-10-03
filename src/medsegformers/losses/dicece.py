import torch.nn as nn
from monai.losses import DiceCELoss

class FlexDiceCELoss(nn.Module):
    """
    Dice + Cross-Entropy that adapts to binary or multi-class.
    """
    def __init__(
        self,
        num_classes: int,
        reduction: str = "mean",
        lambda_dice: float = 1.0,
        lambda_ce: float = 1.0,
    ):
        super().__init__()
        self.loss = DiceCELoss(
            sigmoid=(num_classes == 1),
            softmax=(num_classes > 1),
            to_onehot_y=(num_classes > 1),
            include_background=True,
            reduction=reduction,
            lambda_dice=lambda_dice,
            lambda_ce=lambda_ce,
        )

    def forward(self, logits, targets):
        return self.loss(logits, targets)
