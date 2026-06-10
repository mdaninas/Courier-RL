from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap

from src.agents.train_core import train_agent
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("train")


def _algo_checkpoint(save_path: str | None, algo: str) -> str:
    sp = Path(save_path) if save_path else Path(f"data/checkpoints/{algo.lower()}_courier.zip")
    name = sp.name
    for prefix in ("ppo_", "dqn_"):
        if name.lower().startswith(prefix):
            name = f"{algo.lower()}_{name[len(prefix):]}"
            break
    else:
        name = f"{algo.lower()}_{name}"
    return str(sp.with_name(name))


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a courier RL agent.")
    ap.add_argument("--config", default="config/jakarta_menteng.yaml")
    ap.add_argument("--algo", default=None, choices=["PPO", "DQN", "ppo", "dqn"])
    ap.add_argument("--timesteps", type=int, default=None)
    args = ap.parse_args()

    config = load_config(args.config)
    algo = (args.algo or config["training"]["algorithm"]).upper()

    config["training"]["save_path"] = _algo_checkpoint(
        config["training"].get("save_path"), algo
    )

    result = train_agent(config, algo=algo, total_timesteps=args.timesteps)
    logger.info("Training complete. Model at %s", result["model_path"])


if __name__ == "__main__":
    main()
