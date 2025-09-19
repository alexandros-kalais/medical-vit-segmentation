from typing import Tuple, Optional
import numpy as np
import torch
from tqdm import tqdm
from monai.data import decollate_batch
from monai.transforms import Compose
from monai.metrics import DiceMetric

def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion,
    device: torch.device,
    epoch: int,
    epochs: int,
    wandb_run=None,
    global_step_start: int = 0,
) -> float:
    model.train()
    running = []

    pbar = tqdm(loader, desc=f"Train {epoch+1}/{epochs}", leave=False)
    for i, batch in enumerate(pbar):
        images, labels = batch["image"].to(device), batch["label"].to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running.append(loss.item())
        if wandb_run:
            step = global_step_start + i
            wandb_run.log(
                {"train_loss": loss.item(), "lr": optimizer.param_groups[0]["lr"], "epoch": epoch+1},
                step=step
            )

    return float(np.mean(running)) if running else float("inf")

def validate_one_epoch(
    model: torch.nn.Module,
    loader,
    criterion,
    device: torch.device,
    num_classes: int,
    post_pred: Compose,
    post_label: Optional[Compose],
    wandb_image_logger=None,
) -> Tuple[float, float]:
    model.eval()
    losses = []
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    with torch.no_grad():
        for i, batch in enumerate(loader):
            images, labels = batch["image"].to(device), batch["label"].to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            losses.append(loss.item())

            
            if wandb_image_logger and i == 1:
                wandb_image_logger(images, labels, outputs)

            y_pred = [post_pred(x) for x in decollate_batch(outputs)]
            if num_classes == 1:
                y_true = decollate_batch(labels)
            else:
                y_true = [post_label(x) for x in decollate_batch(labels)]
            dice_metric(y_pred=y_pred, y=y_true)

    valid_loss = float(np.mean(losses)) if losses else float("inf")
    dice = dice_metric.aggregate().item()
    dice_metric.reset()
    return valid_loss, dice
