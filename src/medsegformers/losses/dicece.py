
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
        """
        labels: (B,H,W) in {0..C-1, 255}
        returns: (B,C,H,W) float, ignored pixels become all-zeros
        """
        if labels.dtype != torch.long:
            labels = labels.long()

        b, h, w = labels.shape
        out = labels.new_zeros((b, num_classes, h, w), dtype=torch.float32)

        valid = (labels != IGNORE_INDEX)
        if valid.any():
            # SAFE copy for scatter: route ignored pixels to 0 (in-range), then zero them after
            safe = labels.clone()
            safe[~valid] = 0
            out.scatter_(1, safe.unsqueeze(1), 1.0)
            out *= valid.unsqueeze(1)  # zero-out ignored positions

        return out

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # targets: (B,1,H,W) or (B,H,W) with {0..C-1, 255}
        if targets.ndim == 4 and targets.shape[1] == 1:
            targets = targets[:, 0, :, :]

        targets = targets.long()  # CE expects Long

        # Cross-entropy (ignores 255)
        ce_loss = self.ce(logits, targets)

        # Mask for valid pixels
        valid = (targets != IGNORE_INDEX)

        # If nothing valid in this batch (rare but possible on small crops), skip Dice to avoid NaN
        if not valid.any():
            return self.lambda_ce * ce_loss

        # Softmax probabilities
        probs = F.softmax(logits, dim=1).float()

        # One-hot targets (ignored pixels -> all zeros)
        tgt_1h = self._one_hot(targets, self.num_classes)

        # Apply mask
        probs = probs * valid.unsqueeze(1)      # (B,C,H,W)
        tgt_1h = tgt_1h * valid.unsqueeze(1)    # (B,C,H,W)

        # Dice per class
        dims = (0, 2, 3)  # sum over batch & spatial
        intersection = (probs * tgt_1h).sum(dims)
        denom = probs.sum(dims) + tgt_1h.sum(dims) + 1e-6
        dice_per_class = 2.0 * intersection / denom

        # If a class has zero valid pixels across the batch, its dice contributes 1 - 0 = 1.
        # That’s fine; alternative is to average only over classes with denom>0 if you prefer.
        dice_loss = 1.0 - dice_per_class.mean()

        return self.lambda_ce * ce_loss + self.lambda_dice * dice_loss
