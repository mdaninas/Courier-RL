from __future__ import annotations

from typing import Any

import numpy as np


class BasePolicy:
    name = "base"

    def reset(self, seed: int | None = None) -> None:
        pass

    def act(self, env, obs: np.ndarray) -> int:
        raise NotImplementedError

    @staticmethod
    def candidates(env) -> list[dict[str, Any]]:
        return env.last_candidates
