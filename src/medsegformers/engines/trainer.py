from pathlib import Path
import os
import torch
from torchvision.utils import make_grid
from monai.transforms import Compose, Activations, AsDiscrete


from medsegformers.engines.loops import train_one_epoch, validate_one_epoch
from medsegformers.utils.paths import ckpt_dir
from medsegformers.utils.vis import colorize_index_map, to_np_uint8
from medsegformers.losses import build_loss

class Trainer:
    def __init__(self, args, model: torch.nn.Module, train_loader, val_loader, *, num_classes: int, device: torch.device, wandb_run=None):
        self.args = args
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_classes = num_classes
        self.wandb = wandb_run

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        self.criterion = build_loss(self.args.loss, self.num_classes)

        # post-proc for metrics
        if num_classes == 1:
            self.post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
            self.post_label = None
        else:
            self.post_pred  = Compose([Activations(softmax=True), AsDiscrete(argmax=True, to_onehot=num_classes)])
            self.post_label = Compose([AsDiscrete(to_onehot=num_classes)])

        self.out_dir = ckpt_dir(args.dataset, args.experiment_id)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.best_valid_loss = float("inf")
        self._current_best_path = None

    def _wandb_image_logger(self, images, labels, outputs, *, step: int):
        if self.wandb is None:
            return
        import wandb
        if self.num_classes == 1:
            preds = (outputs.sigmoid() > 0.5).float()
            images_grid = make_grid(images, nrow=2, normalize=True, scale_each=True)
            preds_grid  = make_grid(preds,  nrow=2, normalize=True, scale_each=True)
            labels_grid = make_grid(labels.float(), nrow=2, normalize=True, scale_each=True)
            self.wandb.log({
                "val_images/original":   wandb.Image(images_grid.permute(1,2,0).cpu().numpy()),
                "val_images/prediction": wandb.Image(preds_grid.permute(1,2,0).cpu().numpy()),
                "val_images/label":      wandb.Image(labels_grid.permute(1,2,0).cpu().numpy()),
            }, step=step, commit=False) 
        else:
            preds_idx = outputs.softmax(1).argmax(1)                   # [B,H,W]
            pred_rgb  = colorize_index_map(preds_idx)                   # uint8
            lab_rgb   = colorize_index_map(labels.squeeze(1).long())    # uint8
            pred_grid_u8 = make_grid(pred_rgb, nrow=2, normalize=False, pad_value=255)
            lab_grid_u8  = make_grid(lab_rgb,  nrow=2, normalize=False, pad_value=255)
            img_grid     = make_grid(images,   nrow=2, normalize=True, scale_each=True, pad_value=1.0)
            self.wandb.log({
                "val_images/original":   wandb.Image(img_grid.permute(1,2,0).cpu().numpy()),
                "val_images/prediction": wandb.Image(to_np_uint8(pred_grid_u8)),
                "val_images/label":      wandb.Image(to_np_uint8(lab_grid_u8)),
            }, step=step, commit=False)

    def fit(self):
        steps_per_epoch = len(self.train_loader)
        global_step = 0

        for epoch in range(self.args.epochs):
            # ---- Train ----
            train_loss = train_one_epoch(
                self.model, self.train_loader, self.optimizer, self.criterion,
                self.device, epoch, self.args.epochs, wandb_run=self.wandb,
                global_step_start=global_step,
            )

            # ---- Validate ----
            val_step = global_step + steps_per_epoch - 1

            global_step += steps_per_epoch

            valid_loss, dice = validate_one_epoch(
                self.model, self.val_loader, self.criterion, self.device, self.num_classes,
                self.post_pred, self.post_label,
                wandb_image_logger=lambda imgs, labs, outs: self._wandb_image_logger(imgs, labs, outs, step=val_step),
            )

            if self.wandb:
                self.wandb.log({"valid_loss": valid_loss, "dice_score": dice, "epoch": epoch+1}, step=val_step)  # commit=True by default

            # ---- Checkpoints ----
            if valid_loss < self.best_valid_loss:
                self.best_valid_loss = valid_loss
                if self._current_best_path and os.path.exists(self._current_best_path):
                    os.remove(self._current_best_path)
                tag = f"{self.args.vit_name}_{self.args.decoder}_{self.args.image_size[0]}x{self.args.image_size[1]}"
                self._current_best_path = str(self.out_dir / f"best-{tag}-epoch={epoch:04d}-val_loss={valid_loss:.4f}.pth")
                torch.save(self.model.state_dict(), self._current_best_path)

        tag = f"{self.args.vit_name}_{self.args.decoder}_{self.args.image_size[0]}x{self.args.image_size[1]}"
        final_path = str(self.out_dir / f"final-{tag}-epoch={self.args.epochs-1:04d}-val_loss={valid_loss:.4f}.pth")
        torch.save(self.model.state_dict(), final_path)
        return final_path, self._current_best_path
