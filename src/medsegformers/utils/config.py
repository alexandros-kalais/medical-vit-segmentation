import yaml
from argparse import Namespace
from pathlib import Path

def load_config(cfg_path: str) -> Namespace:

    cfg_file = Path(cfg_path)
    if not cfg_file.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_file}")
    with open(cfg_file, "r") as f:
        data = yaml.safe_load(f)
    return Namespace(**data)
