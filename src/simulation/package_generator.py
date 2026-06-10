from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np

from src.map import node_sampler

PRIORITIES = ("normal", "express", "urgent")
PRIORITY_WEIGHTS = (0.55, 0.30, 0.15)


@dataclass
class Package:
    package_id: int
    destination_node: int
    deadline: float
    priority: str = "normal"
    delivered: bool = False
    delivery_time: float | None = None

    def is_late(self) -> bool:
        return (
            self.delivered
            and self.delivery_time is not None
            and self.delivery_time > self.deadline
        )

    def lateness(self) -> float:
        if not self.delivered or self.delivery_time is None:
            return 0.0
        return max(0.0, self.delivery_time - self.deadline)

    def as_dict(self) -> dict[str, Any]:
        return {
            "package_id": self.package_id,
            "destination_node": self.destination_node,
            "deadline": self.deadline,
            "priority": self.priority,
            "delivered": self.delivered,
            "delivery_time": self.delivery_time,
        }


PRIORITY_DEADLINE_FACTOR = {"urgent": 0.55, "express": 0.78, "normal": 1.0}


def generate_packages(
    graph: nx.MultiDiGraph,
    depot_node: int,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> list[Package]:
    sim_cfg = config["simulation"]
    n = int(sim_cfg["num_packages"])
    window = float(sim_cfg["max_episode_minutes"])
    tour_factor = float(sim_cfg.get("deadline_tour_factor", 0.65))
    slack_min = float(sim_cfg.get("deadline_slack_min", 0.35))
    slack_max = float(sim_cfg.get("deadline_slack_max", 1.25))

    nodes = node_sampler.sample_package_nodes(
        graph, n, rng, exclude=[depot_node], reachable_from=depot_node
    )

    depot_times, _ = nx.single_source_dijkstra(graph, depot_node, weight="base_travel_time")
    reach = [depot_times.get(nd, window) for nd in nodes]
    mean_leg = float(np.mean(reach)) if reach else window / n
    tour_est = max(mean_leg, mean_leg * n * tour_factor)

    packages: list[Package] = []
    for i, node in enumerate(nodes):
        priority = str(rng.choice(PRIORITIES, p=PRIORITY_WEIGHTS))
        slot = rng.uniform(slack_min, slack_max)
        raw = tour_est * slot * PRIORITY_DEADLINE_FACTOR[priority]
        deadline = float(np.clip(raw, reach[i] * 1.15, window))
        packages.append(
            Package(
                package_id=i,
                destination_node=int(node),
                deadline=round(deadline, 1),
                priority=priority,
            )
        )
    return packages


def priority_multiplier(priority: str, config: dict[str, Any]) -> float:
    rcfg = config["reward"]
    return {
        "normal": float(rcfg.get("normal_multiplier", 1.0)),
        "express": float(rcfg.get("express_multiplier", 1.5)),
        "urgent": float(rcfg.get("urgent_multiplier", 2.0)),
    }.get(priority, 1.0)


def priority_index(priority: str) -> float:
    return {"normal": 0.0, "express": 0.5, "urgent": 1.0}.get(priority, 0.0)
