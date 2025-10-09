
import os
from pathlib import Path

def _env_path(name: str):
    v = os.environ.get(name)
    return Path(v).expanduser().resolve() if v else None

def project_root() -> Path:
    env = _env_path("MEDSEG_PROJECT_ROOT")
    if env:
        return env


    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / ".git").exists() or (p / "pyproject.toml").exists() or (p / "setup.cfg").exists():
            return p
    return here.parents[3] 

def get_data_root() -> Path:
    return _env_path("MEDSEG_DATA_ROOT") or (project_root() / "data")

def get_experiments_root() -> Path:
    return _env_path("MEDSEG_EXPERIMENTS_ROOT") or (project_root() / "experiments")

def ckpt_dir(dataset: str, experiment_id: str) -> Path:
    return get_experiments_root() / dataset / experiment_id / "checkpoints"
