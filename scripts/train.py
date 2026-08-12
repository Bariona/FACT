import os
from pathlib import Path

import tyro
import wandb
from fact_train import launch_from_config, load_config


def _ensure_wandb_ready(config_path: str) -> None:
    config = load_config(config_path)
    train_cfg = config.get("train", {})
    if train_cfg.get("log_with") != "wandb":
        return

    project_dir = Path(config.project_dir).expanduser().resolve()
    os.environ.setdefault("WANDB_DIR", str(project_dir / "wandb"))
    os.environ.setdefault("WANDB_CACHE_DIR", str(project_dir / "wandb" / "cache"))

    api = wandb.Api(api_key=os.environ.get("WANDB_API_KEY"))
    try:
        api.viewer
    except Exception:
        wandb.login()


def train(config: str):
    """Launch training. `config` is a module path, e.g. world_action_model.configs.robotwin.config"""
    _ensure_wandb_ready(config)
    launch_from_config(config)


if __name__ == '__main__':
    tyro.cli(train)
