from __future__ import annotations

from typing import Any

import gymnasium as gym
import networkx as nx
import numpy as np
from gymnasium import spaces

from src.map import graph_utils as gu
from src.map import node_sampler, osm_loader
from src.simulation import package_generator as pkg_gen
from src.simulation.courier_state import CourierState
from src.simulation.package_generator import Package
from src.simulation.traffic_model import TrafficModel
from src.utils.logger import get_logger

logger = get_logger(__name__)

ENV_ID = "CourierRoutingRealMapEnv-v0"


class CourierRoutingRealMapEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        config: dict[str, Any],
        graph: nx.MultiDiGraph | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.config = config
        self.render_mode = render_mode

        sim = config["simulation"]
        self.K = int(sim["candidate_package_count"])
        self.num_packages = int(sim["num_packages"])
        self.max_episode_minutes = float(sim["max_episode_minutes"])
        self.start_hour = int(sim["start_hour"])
        self.max_steps = int(sim.get("max_steps", 200))
        self.shuffle_candidates = bool(sim.get("shuffle_candidates", True))

        self.graph: nx.MultiDiGraph = graph if graph is not None else osm_loader.load_map(config)
        self.bounds = gu.graph_bounds(self.graph)

        self._base_seed = int(config.get("project", {}).get("seed", 42))
        self.rng = np.random.default_rng(self._base_seed)

        self.obs_dim = 9 + 3 * self.K
        self.action_space = spaces.Discrete(self.K)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32
        )

        self.depot_node: int = -1
        self.packages: list[Package] = []
        self.remaining: list[Package] = []
        self.courier: CourierState | None = None
        self.traffic: TrafficModel | None = None
        self.last_candidates: list[dict[str, Any]] = []
        self._steps = 0
        self._invalid_actions = 0
        self._reward_log: list[float] = []

        self._fixed_depot: int | None = None
        self._fixed_packages: list[Package] | None = None

    def set_scenario(self, depot_node: int, packages: list[Package]) -> None:
        self._fixed_depot = depot_node
        self._fixed_packages = packages

    def clear_scenario(self) -> None:
        self._fixed_depot = None
        self._fixed_packages = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._initialize_episode()
        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action: int):
        assert self.courier is not None and self.traffic is not None
        action = int(action)
        self._steps += 1
        reward = 0.0
        terminated = False
        truncated = False
        rcfg = self.config["reward"]
        info_extra: dict[str, Any] = {"invalid_action": False, "delivered_package": None}

        candidates = self.last_candidates
        if action < 0 or action >= len(candidates):
            reward -= float(rcfg["invalid_action_penalty"])
            self._invalid_actions += 1
            info_extra["invalid_action"] = True
            self.traffic.update(self.courier.clock_minutes)
        else:
            reward += self._deliver_package(candidates[action], info_extra)
            self.traffic.update(self.courier.clock_minutes)

        if not self.remaining:
            reward += float(rcfg["completion_bonus"])
            terminated = True

        if self.courier.current_time >= self.max_episode_minutes:
            truncated = True
        if self._steps >= self.max_steps:
            truncated = True

        self._reward_log.append(reward)
        obs = self._get_observation()
        info = self._get_info()
        info.update(info_extra)
        if terminated or truncated:
            info["episode_metrics"] = self.episode_metrics()

        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.courier is None:
            return ""
        s = self.courier.summary()
        line = (
            f"t={s['elapsed_minutes']:.0f}min hour={self.courier.clock_hour:.1f} "
            f"node={s['current_node']} delivered={s['delivered_count']}/{self.num_packages} "
            f"late={s['late_deliveries']} dist={s['total_distance_km']:.2f}km"
        )
        if self.render_mode == "human":
            print(line)
        return line

    def close(self):
        pass

    def _initialize_episode(self) -> None:
        if self._fixed_packages is not None and self._fixed_depot is not None:
            import copy as _copy

            self.depot_node = self._fixed_depot
            self.packages = _copy.deepcopy(self._fixed_packages)
            self.num_packages = len(self.packages)
        else:
            self.depot_node = node_sampler.sample_depot(self.graph, self.rng)
            self.packages = pkg_gen.generate_packages(
                self.graph, self.depot_node, self.config, self.rng
            )
        self.remaining = list(self.packages)
        self.courier = CourierState(current_node=self.depot_node, start_hour=self.start_hour)
        self.traffic = TrafficModel(self.graph, self.config, self.rng)
        self.traffic.reset(self.courier.clock_minutes)
        self._steps = 0
        self._invalid_actions = 0
        self._reward_log = []
        self.last_candidates = []

    def _deliver_package(self, candidate: dict[str, Any], info_extra: dict[str, Any]) -> float:
        rcfg = self.config["reward"]
        package: Package = candidate["package"]
        route: list[int] = candidate["route"]
        travel_min: float = candidate["travel_time"]
        distance_m = gu.route_length_m(self.graph, route)

        self.courier.advance(route, travel_min, distance_m)

        package.delivered = True
        package.delivery_time = self.courier.current_time
        self.courier.delivered_packages.append(package.package_id)
        self.remaining = [p for p in self.remaining if not p.delivered]

        prio_mult = pkg_gen.priority_multiplier(package.priority, self.config)
        delivery_reward = float(rcfg["base_delivery_reward"]) * prio_mult
        travel_pen = float(rcfg["travel_time_penalty_weight"]) * travel_min
        dist_pen = float(rcfg.get("distance_penalty_weight", 0.0)) * (distance_m / 1000.0)
        lateness = package.lateness()
        lp = rcfg.get("lateness_priority", {"normal": 1.0, "express": 1.8, "urgent": 3.0})
        late_pen = float(rcfg["lateness_penalty_weight"]) * float(lp.get(package.priority, 1.0)) * lateness
        if package.is_late():
            self.courier.late_deliveries += 1

        info_extra["delivered_package"] = package.package_id
        info_extra["reward_breakdown"] = {
            "delivery_reward": delivery_reward,
            "travel_penalty": -travel_pen,
            "distance_penalty": -dist_pen,
            "lateness_penalty": -late_pen,
            "lateness": lateness,
        }
        return delivery_reward - travel_pen - dist_pen - late_pen

    def _compute_candidates(self) -> list[dict[str, Any]]:
        if not self.remaining or self.courier is None:
            return []
        distances, paths = gu.single_source_travel_times(self.graph, self.courier.current_node)
        scored = []
        for p in self.remaining:
            t = distances.get(p.destination_node)
            if t is None:
                continue
            scored.append((t, paths[p.destination_node], p))
        scored.sort(key=lambda x: x[0])
        candidates = []
        for t, route, p in scored[: self.K]:
            candidates.append(
                {
                    "package": p,
                    "travel_time": float(t),
                    "route": route,
                    "deadline_remaining": p.deadline - self.courier.current_time,
                    "priority": p.priority,
                }
            )
        # Shuffle so the action index carries no "nearest = 0" shortcut; the agent
        # must read per-candidate features (distance/deadline/priority) to choose.
        if self.shuffle_candidates and len(candidates) > 1:
            self.rng.shuffle(candidates)
        return candidates

    def _get_observation(self) -> np.ndarray:
        self.last_candidates = self._compute_candidates()
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        if self.courier is None:
            return obs

        c = self.courier
        window = self.max_episode_minutes
        max_mult = max(1e-6, float(self.config["traffic"]["max_multiplier"]) - 1.0)

        obs[0] = self._norm_lon(self.graph.nodes[c.current_node]["x"])
        obs[1] = self._norm_lat(self.graph.nodes[c.current_node]["y"])
        obs[2] = _clip01(c.current_time / window)
        obs[3] = _clip01(len(self.remaining) / max(1, self.num_packages))

        nearest_t = min((c["travel_time"] for c in self.last_candidates), default=window)
        obs[4] = _clip01(nearest_t / window)
        obs[5] = _clip01((self.traffic.average_multiplier() - 1.0) / max_mult)

        urgent = sum(1 for p in self.remaining if p.priority == "urgent")
        obs[6] = _clip01(urgent / max(1, self.num_packages))

        if self.remaining:
            min_dl_remaining = min(p.deadline - c.current_time for p in self.remaining)
            obs[7] = _clip_pm1(min_dl_remaining / window)
        else:
            obs[7] = 1.0

        depot_t = self._depot_travel_time()
        obs[8] = _clip01(depot_t / window)

        for k in range(self.K):
            base = 9 + 3 * k
            if k < len(self.last_candidates):
                cand = self.last_candidates[k]
                obs[base + 0] = _clip01(cand["travel_time"] / window)
                obs[base + 1] = _clip_pm1(cand["deadline_remaining"] / window)
                obs[base + 2] = pkg_gen.priority_index(cand["priority"])
            else:
                obs[base + 0] = -1.0
                obs[base + 1] = -1.0
                obs[base + 2] = -1.0
        return np.clip(obs, -1.0, 1.0).astype(np.float32)

    def _depot_travel_time(self) -> float:
        if self.courier is None or self.courier.current_node == self.depot_node:
            return 0.0
        try:
            return nx.shortest_path_length(
                self.graph, self.courier.current_node, self.depot_node, weight="current_travel_time"
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return self.max_episode_minutes

    def _get_info(self) -> dict[str, Any]:
        return {
            "elapsed_minutes": self.courier.current_time if self.courier else 0.0,
            "delivered": len(self.courier.delivered_packages) if self.courier else 0,
            "remaining": len(self.remaining),
            "num_candidates": len(self.last_candidates),
            "steps": self._steps,
        }

    def episode_metrics(self) -> dict[str, Any]:
        delivered = [p for p in self.packages if p.delivered]
        late = [p for p in delivered if p.is_late()]
        latenesses = [p.lateness() for p in delivered]
        total_reward = float(np.sum(self._reward_log))
        return {
            "delivered": len(delivered),
            "total_packages": self.num_packages,
            "completion_rate": len(delivered) / max(1, self.num_packages),
            "total_travel_minutes": round(self.courier.total_travel_time, 2) if self.courier else 0.0,
            "total_distance_km": round(self.courier.total_distance / 1000.0, 3) if self.courier else 0.0,
            "late_packages": len(late),
            "avg_lateness": round(float(np.mean(latenesses)), 2) if latenesses else 0.0,
            "avg_time_per_package": (
                round(self.courier.total_travel_time / len(delivered), 2)
                if self.courier and delivered else 0.0
            ),
            "invalid_actions": self._invalid_actions,
            "total_reward": round(total_reward, 2),
        }

    def _norm_lon(self, lon: float) -> float:
        span = self.bounds["max_lon"] - self.bounds["min_lon"] or 1.0
        return _clip01((lon - self.bounds["min_lon"]) / span)

    def _norm_lat(self, lat: float) -> float:
        span = self.bounds["max_lat"] - self.bounds["min_lat"] or 1.0
        return _clip01((lat - self.bounds["min_lat"]) / span)

    def route_history(self) -> list[int]:
        return list(self.courier.route_nodes) if self.courier else []


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def _clip_pm1(x: float) -> float:
    return float(min(1.0, max(-1.0, x)))
