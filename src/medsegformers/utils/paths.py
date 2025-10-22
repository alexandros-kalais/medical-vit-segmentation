import os
from pathlib import Path
from typing import Optional

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
    return get_experiments_root() / dataset / experiment_id


def _read_runs_txt(p: Path) -> list[str]:
    """Read experiment IDs from text file."""
    with p.open("r") as f:
        return [ln.strip() for ln in f.readlines() if ln.strip()]


def _exp_dir_from_id(exp_id: str, dataset: str) -> Path:
    """Get experiment directory from ID and dataset."""
    return get_experiments_root() / dataset / exp_id


def _pick_ckpt(ckpt_dir: Path) -> Optional[Path]:
    """Pick the best checkpoint from directory (best-*.ckpt pattern)."""
    if not ckpt_dir.is_dir():
        return None
    
    best_ckpts = list(ckpt_dir.glob("best-*.ckpt"))
    
    return best_ckpts[-1]

