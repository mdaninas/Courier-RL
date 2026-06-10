from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.agents.env_factory import make_vec_env
from src.map import osm_loader
from src.utils.config import resolve_path, save_config
from src.utils.logger import get_logger, set_global_seed

logger = get_logger(__name__)


class RewardLoggerCallback:
    pass


def _build_callback():
    from stable_baselines3.common.callbacks import BaseCallback

    class _RewardLogger(BaseCallback):
        def __init__(self):
            super().__init__()
            self.episode_rewards: list[float] = []
            self.timesteps: list[int] = []

        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                ep = info.get("episode")
                if ep is not None:
                    self.episode_rewards.append(float(ep["r"]))
                    self.timesteps.append(int(self.num_timesteps))
            return True

    return _RewardLogger()


def train_agent(
    config: dict[str, Any],
    algo: str | None = None,
    total_timesteps: int | None = None,
) -> dict[str, Any]:
    tcfg = config["training"]
    algo = (algo or tcfg.get("algorithm", "PPO")).upper()
    timesteps = int(total_timesteps or tcfg["total_timesteps"])

    seed = int(config.get("project", {}).get("seed", 42))
    set_global_seed(seed)

    log_dir = resolve_path(tcfg.get("log_dir", "data/logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    save_path = resolve_path(tcfg.get("save_path", f"data/checkpoints/{algo.lower()}_courier.zip"))
    save_path.parent.mkdir(parents=True, exist_ok=True)

    graph = osm_loader.load_map(config)
    monitor_file = str(log_dir / f"{algo.lower()}_monitor")
    venv = make_vec_env(config, graph=graph, seed=seed, log_dir=monitor_file)

    model = _make_model(algo, venv, config, seed, log_dir)
    callback = _build_callback()

    logger.info("Training %s for %d timesteps ...", algo, timesteps)
    t0 = time.time()
    model.learn(total_timesteps=timesteps, callback=callback, progress_bar=False)
    elapsed = time.time() - t0
    logger.info("Training finished in %.1fs", elapsed)

    model.save(str(save_path))
    logger.info("Saved model -> %s", save_path)

    rewards = callback.episode_rewards
    timesteps_log = callback.timesteps
    reward_json = save_path.with_suffix(".rewards.json")
    with open(reward_json, "w", encoding="utf-8") as fh:
        json.dump({"timesteps": timesteps_log, "episode_rewards": rewards}, fh, indent=2)

    plot_path = None
    try:
        from src.visualization.plot_metrics import plot_reward_curve

        plot_path = resolve_path(tcfg.get("log_dir", "data/logs")) / f"{algo.lower()}_reward_curve.png"
        plot_reward_curve(rewards, timesteps_log, str(plot_path), title=f"{algo} training reward")
    except Exception as exc:
        logger.warning("Could not plot reward curve: %s", exc)

    save_config(config, save_path.with_suffix(".config.yaml"))

    last_mean = float(np.mean(rewards[-20:])) if rewards else float("nan")
    result = {
        "algo": algo,
        "model_path": str(save_path),
        "reward_log": str(reward_json),
        "reward_curve": str(plot_path) if plot_path else None,
        "episodes": len(rewards),
        "final_mean_reward": round(last_mean, 2),
        "train_seconds": round(elapsed, 1),
    }
    logger.info("Result: %s", result)
    return result


def _make_model(algo: str, venv, config: dict[str, Any], seed: int, log_dir: Path):
    tcfg = config["training"]
    tb = str(log_dir) if tcfg.get("tensorboard") else None
    common = dict(
        policy="MlpPolicy",
        env=venv,
        verbose=1,
        seed=seed,
        gamma=float(tcfg.get("gamma", 0.99)),
        learning_rate=float(tcfg.get("learning_rate", 3e-4)),
        tensorboard_log=tb,
    )
    net = tcfg.get("net_arch")
    if net:
        common["policy_kwargs"] = {"net_arch": list(net)}
    if algo == "PPO":
        from stable_baselines3 import PPO

        return PPO(
            n_steps=int(tcfg.get("n_steps", 2048)),
            batch_size=int(tcfg.get("batch_size", 64)),
            ent_coef=float(tcfg.get("ent_coef", 0.0)),
            **common,
        )
    if algo == "DQN":
        from stable_baselines3 import DQN

        return DQN(
            buffer_size=int(tcfg.get("buffer_size", 50000)),
            batch_size=int(tcfg.get("batch_size", 64)),
            learning_starts=int(tcfg.get("learning_starts", 1000)),
            target_update_interval=int(tcfg.get("target_update_interval", 500)),
            **common,
        )
    raise ValueError(f"Unsupported algorithm '{algo}'. Use PPO or DQN.")
