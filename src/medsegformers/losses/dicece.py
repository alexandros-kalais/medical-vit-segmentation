
import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE_INDEX = 255

class FlexDiceCELoss(nn.Module):
    def __init__(self, num_classes: int, lambda_dice: float = 1.0, lambda_ce: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce
        self.ce = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    @staticmethod
    def _one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:

        if labels.dtype != torch.long:
            labels = labels.long()

        b, h, w = labels.shape
        out = labels.new_zeros((b, num_classes, h, w), dtype=torch.float32)

        valid = (labels != IGNORE_INDEX)
        if valid.any():
            safe = labels.clone()
            safe[~valid] = 0
            out.scatter_(1, safe.unsqueeze(1), 1.0)
            out *= valid.unsqueeze(1)

        return out

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

        if targets.ndim == 4 and targets.shape[1] == 1:
            targets = targets[:, 0, :, :]

        targets = targets.long()  

        ce_loss = self.ce(logits, targets)

        valid = (targets != IGNORE_INDEX)

        if not valid.any():
            return self.lambda_ce * ce_loss

        probs = F.softmax(logits, dim=1).float()

        tgt_1h = self._one_hot(targets, self.num_classes)

        probs = probs * valid.unsqueeze(1)
        tgt_1h = tgt_1h * valid.unsqueeze(1)

        dims = (0, 2, 3) 
        intersection = (probs * tgt_1h).sum(dims)
        denom = probs.sum(dims) + tgt_1h.sum(dims) + 1e-6
        dice_per_class = 2.0 * intersection / denom

        dice_loss = 1.0 - dice_per_class.mean()

        return self.lambda_ce * ce_loss + self.lambda_dice * dice_loss
