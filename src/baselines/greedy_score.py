from __future__ import annotations

from src.baselines.base_policy import BasePolicy
from src.simulation import package_generator as pkg_gen


class GreedyScorePolicy(BasePolicy):
    name = "greedy_score"

    def __init__(self, config: dict | None = None):
        self.config = config or {}

    def act(self, env, obs) -> int:
        cands = self.candidates(env)
        if not cands:
            return 0
        cfg = env.config if hasattr(env, "config") else self.config

        def score(i: int) -> float:
            c = cands[i]
            weight = pkg_gen.priority_multiplier(c["priority"], cfg)
            urgency = max(0.0, c["deadline_remaining"])
            denom = c["travel_time"] + urgency + 1e-6
            return weight / denom

        return max(range(len(cands)), key=score)
