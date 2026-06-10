from __future__ import annotations

from collections.abc import Callable
from typing import Any

import networkx as nx

from src.envs.courier_routing_env import CourierRoutingRealMapEnv
from src.map import osm_loader
from src.utils.config import resolve_path


def make_env(
    config: dict[str, Any],
    graph: nx.MultiDiGraph | None = None,
    seed: int | None = None,
    monitor: bool = True,
    log_dir: str | None = None,
) -> Callable[[], Any]:
    if graph is None:
        graph = osm_loader.load_map(config)

    def _init():
        env = CourierRoutingRealMapEnv(config, graph=graph)
        if seed is not None:
            env.reset(seed=seed)
        if monitor:
            from stable_baselines3.common.monitor import Monitor

            path = str(resolve_path(log_dir)) if log_dir else None
            env = Monitor(env, filename=path)
        return env

    return _init


def make_vec_env(config: dict[str, Any], graph=None, seed: int | None = None, log_dir: str | None = None):
    from stable_baselines3.common.vec_env import DummyVecEnv

    return DummyVecEnv([make_env(config, graph=graph, seed=seed, log_dir=log_dir)])
