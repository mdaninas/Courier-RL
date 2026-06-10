from __future__ import annotations

from src.baselines.base_policy import BasePolicy


class EarliestDeadlinePolicy(BasePolicy):
    name = "earliest_deadline"

    def act(self, env, obs) -> int:
        cands = self.candidates(env)
        if not cands:
            return 0
        return min(range(len(cands)), key=lambda i: cands[i]["package"].deadline)
