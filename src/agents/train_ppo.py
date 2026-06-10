from __future__ import annotations

from typing import Any

from src.agents.train_core import train_agent


def train_ppo(config: dict[str, Any], total_timesteps: int | None = None) -> dict[str, Any]:
    return train_agent(config, algo="PPO", total_timesteps=total_timesteps)
