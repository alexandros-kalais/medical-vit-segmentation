import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from typing import Tuple, List, Union, Any, Dict, Optional
from medsegformers.models import EncDecModel, EoMT, Mask2FormerModel
from medsegformers.engines import EncoderDecoderSegModule, MaskClassificationSemantic

def encdec_collate(batch):
    return torch.utils.data.dataloader.default_collate(batch)

def eomt_train_collate(batch):
    images, targets = [], []
    for image, target in batch:
        images.append(image)
        targets.append(target)
    images = torch.stack(images)
    return images, targets

def eomt_eval_collate(batch):
    return tuple(zip(*batch))

def _to_tensor(x):
    return x.as_tensor() if hasattr(x, "as_tensor") else torch.as_tensor(x)

def mask2former_collate_cpu(batch):
    images, targets = zip(*batch)
    images = torch.stack([_to_tensor(img) for img in images], dim=0).float()
    clean_targets = []
    for t in targets:
        labels = _to_tensor(t["labels"]).long()
        masks = _to_tensor(t["masks"]).float()
        is_crowd = _to_tensor(t["is_crowd"]).bool()
        clean_targets.append({"labels": labels, "masks": masks, "is_crowd": is_crowd})
    return images, clean_targets

def select_collate(model_type: str):
    if model_type == "enc_dec":
        train_collate_fn = encdec_collate
        val_collate_fn = encdec_collate
    elif model_type == "mask2former":
        train_collate_fn = mask2former_collate_cpu
        val_collate_fn = eomt_eval_collate
    elif model_type == "eomt":
        train_collate_fn = eomt_train_collate
        val_collate_fn = eomt_eval_collate
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")
    return train_collate_fn, val_collate_fn


def infer_patch_size(vit_name: str) -> int:
    vit_lower = vit_name.lower()
    if any(x in vit_lower for x in ["16-", "16_"]):
        return 16
    elif any(x in vit_lower for x in ["14_"]):
        return 14
    else:
        raise ValueError(f"Could not infer patch size from vit_name: {vit_name}")

def freeze_encoder_layers(encoder, freeze_encoder: bool, unfreeze_last_k: int):
    if freeze_encoder:
        n = len(encoder.backbone.blocks) - unfreeze_last_k
        for blk in encoder.backbone.blocks[:n]:
            for p in blk.parameters():
                p.requires_grad_(False)
            for mod in blk.modules():
                if isinstance(mod, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
                    for p in mod.parameters():
                        p.requires_grad_(False)
    else:
        for p in encoder.backbone.parameters():
            p.requires_grad_(True)

def compute_eomt_anneal_schedule(
    steps_per_epoch: int, 
    num_blocks: int
) -> Tuple[List[int], List[int]]:

    anneal_starts = [2 * steps_per_epoch * (i + 1) for i in range(num_blocks)]
    anneal_ends = [2 * steps_per_epoch * (i + 2) for i in range(num_blocks)]
    return anneal_starts, anneal_ends

def generate_experiment_id(args) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    vit_name_lower = args.vit_name.lower()
    
    if "dinov3" in vit_name_lower:
        vit_short = "dinov3"
    elif "dinov2" in vit_name_lower:
        vit_short = "dinov2"
    elif "dino" in vit_name_lower:
        vit_short = "dino"
    else:
        vit_short = "imagenet"
    
    model_type = args.model_type
    h, w = args.image_size
    
    if model_type == "enc_dec":
        model_prefix = args.decoder
    elif model_type == "mask2former":
        model_prefix = "mask2former"
    elif model_type == "eomt":
        model_prefix = "eomt"
    else:
        model_prefix = model_type
    
    exp_id = f"{model_prefix}_{vit_short}_{h}x{w}_lr{args.lr}_bs{args.batch_size}_{timestamp}"
    return exp_id

def build_encdec_model(
    vit: nn.Module,
    num_classes: int,
    config: Union[Any, Dict[str, Any]],
    mode: str = "train",
    device: Optional[torch.device] = None
) -> nn.Module:

    if mode == "train":

        decoder = config.decoder
        decoder_kwargs = getattr(config, "decoder_kwargs", None)
    else:  

        decoder = config["decoder"]
        decoder_kwargs = config.get("decoder_kwargs", None)
    
    model = EncDecModel(
        vit=vit,
        num_classes=num_classes,
        decoder=decoder,
        decoder_kwargs=decoder_kwargs,
    )
    
    if mode == "eval":
        model = model.to(device)
    
    return model

def build_mask2former_model(
    vit: nn.Module,
    num_classes: int,
    config: Union[Any, Dict[str, Any]],
    mode: str = "train",
    image_size: Optional[Tuple[int, int]] = None,
    device: Optional[torch.device] = None
) -> nn.Module:

    if mode == "train":
        nheads = config.nheads
        transformer_in_features = config.transformer_in_features
        dim_feedforward = config.dim_feedforward
        enc_layers = config.enc_layers
        common_stride = config.common_stride
        num_queries = config.num_queries
        dec_layers = config.dec_layers
        masked_attn_enabled = getattr(config, "eomt_masked_attn_enabled", False)
    else:
        nheads = config.get("nheads")
        transformer_in_features = config.get("transformer_in_features")
        dim_feedforward = config.get("dim_feedforward")
        enc_layers = config.get("enc_layers")
        common_stride = config.get("common_stride")
        num_queries = config.get("num_queries")
        dec_layers = config.get("dec_layers")
        masked_attn_enabled = config.get("eomt_masked_attn_enabled", False)
    
    network = Mask2FormerModel(
        encoder=vit,
        num_classes=num_classes,
        adapter_kwargs={
            "deform_num_heads": nheads,
            "n_points": 4,
            "with_cp": False,
            "add_vit_feature": True,
            "deform_ratio": 0.5,
            "drop_path_rate": 0.0,
        },
        pixel_decoder_kwargs={
            "transformer_in_features": transformer_in_features,
            "transformer_nheads": nheads,
            "transformer_dim_feedforward": dim_feedforward,
            "transformer_enc_layers": enc_layers,
            "common_stride": common_stride,
        },
        transformer_kwargs={
            "num_queries": num_queries,
            "nheads": nheads,
            "dim_feedforward": dim_feedforward,
            "dec_layers": dec_layers,
        },
        masked_attn_enabled=masked_attn_enabled,
    )
    
    if mode == "eval":
        model = MaskClassificationSemanticWrapper(
            network, img_size=image_size, num_classes=num_classes
        )
        model = model.to(device)
    else:
        model = network
    
    return model


def build_eomt_model(
    vit: nn.Module,
    num_classes: int,
    config: Union[Any, Dict[str, Any]],
    mode: str = "train",
    image_size: Optional[Tuple[int, int]] = None,
    device: Optional[torch.device] = None
) -> nn.Module:

    if mode == "train":
        num_q = config.eomt_num_q
        num_blocks = config.eomt_num_blocks
        masked_attn_enabled = bool(config.eomt_masked_attn_enabled)
    else:
        num_q = config.get("eomt_num_q")
        num_blocks = config.get("eomt_num_blocks")
        masked_attn_enabled = config.get("eomt_masked_attn_enabled", False)
    
    network = EoMT(
        encoder=vit,
        num_classes=num_classes,
        num_q=num_q,
        num_blocks=num_blocks,
        masked_attn_enabled=masked_attn_enabled,
    )
    
    if mode == "eval":
        model = MaskClassificationSemanticWrapper(
            network, img_size=image_size, num_classes=num_classes
        )
        model = model.to(device)
    else:
        model = network

    return model

def build_model(
    model_type: str,
    vit: nn.Module,
    num_classes: int,
    config: Union[Any, Dict[str, Any]],
    mode: str = "train",
    image_size: Optional[Tuple[int, int]] = None,
    device: Optional[torch.device] = None
    ):

    if model_type == "enc_dec":
        model = build_encdec_model(
            vit=vit,
            num_classes=num_classes, 
            config=config,
            mode=mode, 
            device=device)
    elif model_type == "mask2former":
        model = build_mask2former_model(
            vit=vit,
            num_classes=num_classes, 
            config=config,
            mode=mode, 
            image_size=image_size, 
            device=device)
    elif model_type == "eomt":
        model = build_eomt_model(
            vit=vit,
            num_classes=num_classes, 
            config=config, 
            mode=mode, 
            image_size=image_size, 
            device=device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    return model


def build_encdec_module(args, model, num_classes, warmup_steps):
    module = EncoderDecoderSegModule(
        network=model,
        num_classes=num_classes,
        lr=args.lr,
        weight_decay=args.weight_decay,
        llrd=args.llrd,
        lr_multi=args.lr_multi,
        poly_power=args.poly_power,
        warmup_steps=warmup_steps,
    )
    return module


def build_mask2former_module(args, model, num_classes, warmup_steps):
    module = MaskClassificationSemantic(
        network=model,
        img_size=args.image_size,
        num_classes=num_classes,
        attn_mask_annealing_enabled=bool(getattr(args, "eomt_masked_attn_enabled", False)),
        ignore_idx=255,
        lr=args.lr,
        llrd=args.llrd,
        llrd_l2_enabled=args.llrd_l2_enabled,
        lr_mult=args.lr_multi,
        weight_decay=args.weight_decay,
        num_points=args.num_points,
        oversample_ratio=args.oversample_ratio,
        importance_sample_ratio=args.importance_sample_ratio,
        poly_power=args.poly_power,
        warmup_steps=warmup_steps,
        no_object_coefficient=args.no_object_coefficient,
        mask_coefficient=args.mask_coefficient,
        dice_coefficient=args.dice_coefficient,
        class_coefficient=args.class_coefficient,
        ckpt_path=getattr(args, "ckpt_path", None),
        delta_weights=getattr(args, "delta_weights", False),
        load_ckpt_class_head=getattr(args, "load_ckpt_class_head", True),
    )
    return module


def build_eomt_module(args, model, num_classes, warmup_steps, anneal_starts, anneal_ends):
    module = MaskClassificationSemantic(
        network=model,
        img_size=args.image_size,
        num_classes=num_classes,
        attn_mask_annealing_enabled=bool(args.eomt_masked_attn_enabled),
        lr=args.lr,
        llrd=args.llrd,
        lr_mult=args.lr_multi,
        warmup_steps=warmup_steps,
        attn_mask_annealing_start_steps=anneal_starts,
        attn_mask_annealing_end_steps=anneal_ends,
    )
    return module

def build_module(
    args, 
    model, 
    num_classes, 
    warmup_steps,
    steps_per_epoch,
    eomt_num_blocks = None, 
    ):

    if args.model_type == "enc_dec":
        module = build_encdec_module(args, model, num_classes, warmup_steps)
    elif args.model_type == "mask2former":
        module = build_mask2former_module(args, model, num_classes, warmup_steps)
    elif args.model_type == "eomt":
        anneal_starts, anneal_ends = compute_eomt_anneal_schedule(steps_per_epoch, eomt_num_blocks)
        module = build_eomt_module(args, model, num_classes, warmup_steps, anneal_starts, anneal_ends)
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    return module


class MaskClassificationSemanticWrapper(nn.Module):

    def __init__(self, network: nn.Module, img_size: tuple[int, int], num_classes: int):
        super().__init__()
        self.network = network
        self.img_size = img_size
        self.num_classes = num_classes


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        mask_logits_list, class_logits_list = self.network(x)
        mask_logits = mask_logits_list[-1]
        class_logits = class_logits_list[-1]

        mask_logits = F.interpolate(mask_logits, size=self.img_size, mode="bilinear")

        per_pixel = torch.einsum(
            "bqhw,bqc->bchw",
            mask_logits.sigmoid(),
            class_logits.softmax(dim=-1)[..., :-1],
        )

        if self.num_classes == 2:
            p_fg = per_pixel
            p_bg = (1.0 - p_fg.sum(dim=1, keepdim=True)).clamp(0.0, 1.0)
            return torch.cat([p_bg, p_fg], dim=1)

        return per_pixel
