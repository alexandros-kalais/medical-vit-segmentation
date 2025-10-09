import torch
from monai.data import DataLoader, list_data_collate

from medsegformers.config.args import get_eval_args_parser
from medsegformers.data import get_dataset_class
from medsegformers.transforms import get_transforms
from medsegformers.models import build_segmentation_model
from medsegformers.utils.paths import get_data_root
from medsegformers.engines.evaluator import Evaluator

"""
NEED TO CHANGE THIS!
"""

def main():
    args = get_eval_args_parser().parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DatasetCls = get_dataset_class(args.dataset)
    root = DatasetCls.default_root(get_data_root())
    tf = get_transforms(dataset=args.dataset, kind=args.tf_kind, image_size=args.image_size)
    test_ds = DatasetCls.build_split("test", transform=tf, root=root, seed=args.seed)
    num_classes = getattr(DatasetCls, "NUM_CLASSES", None)

    loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_segmentation_model(
        decoder=args.decoder,
        num_classes=num_classes,
        vit_name=args.vit_name,
        pretrained=True,
        freeze_encoder=args.freeze_encoder
    ).to(device)

    evaluator = Evaluator(model=model, num_classes=num_classes, device=device)
    evaluator.load_checkpoint(args.checkpoint)

    evaluator.run(loader, dataset=args.dataset)

if __name__ == "__main__":
    main()
