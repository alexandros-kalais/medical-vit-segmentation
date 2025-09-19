from pathlib import Path

def project_root() -> Path:
    return Path(__file__).resolve().parents[3]

def get_data_root() -> Path:
    return project_root() / "data"

def ckpt_dir(dataset: str, experiment_id: str) -> Path:
    return project_root() / "experiments" / dataset / experiment_id / "checkpoints"
