from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np

from src.map import graph_utils as gu
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TrafficModel:
    def __init__(self, graph: nx.MultiDiGraph, config: dict[str, Any], rng: np.random.Generator):
        self.graph = graph
        self.cfg = config["traffic"]
        self.rng = rng
        self.enabled = bool(self.cfg.get("enable_dynamic_traffic", True))
        self.base = float(self.cfg.get("base_multiplier", 1.0))
        self.max_mult = float(self.cfg.get("max_multiplier", 3.5))
        self.incident_p = float(self.cfg.get("random_incident_probability", 0.05))
        self.update_every = float(self.cfg.get("update_every_minutes", 15))

        self._edge_mids = gu.edge_midpoints(graph)
        self.hotspots: list[dict[str, float]] = self._resolve_hotspots(config)
        self._last_update_clock: float | None = None

    def _resolve_hotspots(self, config: dict[str, Any]) -> list[dict[str, float]]:
        hotspots = self.cfg.get("hotspots")
        if hotspots:
            return [dict(h) for h in hotspots]
        bounds = gu.graph_bounds(self.graph)
        k = int(self.cfg.get("num_auto_hotspots", 3))
        out = []
        for _ in range(k):
            lat = self.rng.uniform(bounds["min_lat"], bounds["max_lat"])
            lon = self.rng.uniform(bounds["min_lon"], bounds["max_lon"])
            out.append(
                {
                    "lat": float(lat),
                    "lon": float(lon),
                    "radius_m": float(self.rng.uniform(400, 900)),
                    "multiplier": float(self.rng.uniform(1.6, 2.4)),
                }
            )
        return out

    @staticmethod
    def get_time_based_multiplier(current_hour: float, cfg: dict[str, Any]) -> float:
        m_start, m_end = cfg.get("rush_hour_morning", [7, 9])
        e_start, e_end = cfg.get("rush_hour_evening", [17, 19])
        mid = cfg.get("midday_window", [12, 13])
        if m_start <= current_hour <= m_end:
            return 2.0
        if e_start <= current_hour <= e_end:
            return 2.5
        if mid[0] <= current_hour <= mid[1]:
            return 1.5
        return 1.0

    def reset(self, clock_minutes: float) -> None:
        self._last_update_clock = None
        self.update(clock_minutes, force=True)

    def update(self, clock_minutes: float, force: bool = False) -> None:
        if not self.enabled:
            if force:
                for _, _, data in self.graph.edges(data=True):
                    data["traffic_multiplier"] = 1.0
                    data["current_travel_time"] = data["base_travel_time"]
            return

        if (
            not force
            and self._last_update_clock is not None
            and (clock_minutes - self._last_update_clock) < self.update_every
        ):
            return
        self._last_update_clock = clock_minutes

        hour = (clock_minutes / 60.0) % 24
        time_mult = self.get_time_based_multiplier(hour, self.cfg)

        for (u, v, key), (mlat, mlon) in self._edge_mids.items():
            spatial = 1.0
            for hs in self.hotspots:
                d = gu.haversine_m(mlat, mlon, hs["lat"], hs["lon"])
                if d <= hs["radius_m"]:
                    strength = 1.0 - (d / hs["radius_m"])
                    spatial = max(spatial, 1.0 + (hs["multiplier"] - 1.0) * strength)

            incident = 1.0
            if self.rng.random() < self.incident_p:
                incident = float(self.rng.uniform(1.3, 2.0))

            mult = self.base * time_mult * spatial * incident
            mult = float(np.clip(mult, 1.0, self.max_mult))

            data = self.graph.edges[u, v, key]
            data["traffic_multiplier"] = mult
            data["current_travel_time"] = data["base_travel_time"] * mult

    def average_multiplier(self) -> float:
        vals = [d.get("traffic_multiplier", 1.0) for _, _, d in self.graph.edges(data=True)]
        return float(np.mean(vals)) if vals else 1.0
