from __future__ import annotations

import numpy as np

from src.baselines.base_policy import BasePolicy


class RandomPolicy(BasePolicy):
    name = "random"

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def act(self, env, obs) -> int:
        cands = self.candidates(env)
        if not cands:
            return 0
        return int(self.rng.integers(0, len(cands)))
