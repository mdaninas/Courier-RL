from __future__ import annotations

from typing import Any

from src.baselines.base_policy import BasePolicy
from src.baselines.earliest_deadline import EarliestDeadlinePolicy
from src.baselines.greedy_score import GreedyScorePolicy
from src.baselines.nearest_neighbor import NearestNeighborPolicy
from src.baselines.random_policy import RandomPolicy

BASELINE_NAMES = ("random", "nearest_neighbor", "earliest_deadline", "greedy_score")


def make_baseline(name: str, config: dict[str, Any] | None = None, seed: int | None = None) -> BasePolicy:
    name = name.lower()
    if name == "random":
        return RandomPolicy(seed=seed)
    if name == "nearest_neighbor":
        return NearestNeighborPolicy()
    if name == "earliest_deadline":
        return EarliestDeadlinePolicy()
    if name == "greedy_score":
        return GreedyScorePolicy(config=config)
    raise ValueError(f"Unknown baseline '{name}'. Choose from {BASELINE_NAMES}.")
