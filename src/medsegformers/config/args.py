from argparse import ArgumentParser

def get_train_args_parser():
    p = ArgumentParser("Training for medical ViT segmentation model")
    p.add_argument("--dataset", type=str, choices=["hyperkvasir", "endoscopy", "endoscopy_eomt"], required=True)
    p.add_argument("--decoder", type=str, default="linear", help="Decoder head key [linear, naive, pup, mla, masktfm]")
    p.add_argument("--vit-name", type=str, default="vit_small_patch14_dinov2.lvd142m", help="timm model name for DINOv2")
    p.add_argument("--freeze-encoder", action="store_true", default=True)
    p.add_argument("--image-size", type=int, nargs=2, default=None)
    p.add_argument("--train-tf-kind", type=str, default="basic", choices=["basic", "aug"])
    p.add_argument("--val-tf-kind", type=str, default="basic", choices=["basic", "aug"])
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--loss", type=str, default="dicece", choices=["dice", "dicece"])
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--experiment-id", type=str, default="vit")
    p.add_argument("--subset", type=int, default=0)
    return p

def get_eval_args_parser():
    p = ArgumentParser("Evaluate a trained segmentation model")
    p.add_argument("--dataset", type=str, choices=["hyperkvasir","endoscopy"], required=True)
    p.add_argument("--decoder", type=str, default="linear", help="Decoder head key [linear, naive, pup, mla, masktfm]")
    p.add_argument("--vit-name", type=str, default="vit_base_patch14_dinov2", help="timm model name for DINOv2 (keep default unless you know what you are doing)")
    p.add_argument("--freeze-encoder", action="store_true", default=True)
    p.add_argument("--image-size", type=int, nargs=2, default=None)
    p.add_argument("--tf-kind", type=str, default="basic", choices=["basic","aug"])
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint", type=str, required=True, help="Path to .pth (state_dict)")
    return p

