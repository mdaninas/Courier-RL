from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CourierState:
    current_node: int
    current_time: float = 0.0
    start_hour: int = 8
    total_distance: float = 0.0
    total_travel_time: float = 0.0
    late_deliveries: int = 0
    delivered_packages: list[int] = field(default_factory=list)
    route_nodes: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.route_nodes:
            self.route_nodes = [self.current_node]

    @property
    def clock_minutes(self) -> float:
        return self.start_hour * 60.0 + self.current_time

    @property
    def clock_hour(self) -> float:
        return (self.clock_minutes / 60.0) % 24

    def advance(self, route: list[int], travel_minutes: float, distance_m: float) -> None:
        self.current_time += travel_minutes
        self.total_travel_time += travel_minutes
        self.total_distance += distance_m
        if route:
            self.route_nodes.extend(route[1:] if route[0] == self.current_node else route)
            self.current_node = route[-1]

    def summary(self) -> dict[str, Any]:
        return {
            "current_node": self.current_node,
            "elapsed_minutes": round(self.current_time, 2),
            "total_distance_km": round(self.total_distance / 1000.0, 3),
            "total_travel_minutes": round(self.total_travel_time, 2),
            "late_deliveries": self.late_deliveries,
            "delivered_count": len(self.delivered_packages),
        }
