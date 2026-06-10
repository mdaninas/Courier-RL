from __future__ import annotations

import networkx as nx
import numpy as np

from src.map import graph_utils as gu

try:
    import osmnx as ox

    _HAS_OSMNX = True
except Exception:
    ox = None
    _HAS_OSMNX = False


def nearest_node(graph: nx.MultiDiGraph, lat: float, lon: float) -> int:
    if _HAS_OSMNX:
        try:
            return int(ox.distance.nearest_nodes(graph, X=lon, Y=lat))
        except Exception:
            pass
    best, best_d = None, float("inf")
    for node, data in graph.nodes(data=True):
        d = gu.haversine_m(lat, lon, data["y"], data["x"])
        if d < best_d:
            best, best_d = node, d
    return int(best)


def sample_depot(graph: nx.MultiDiGraph, rng: np.random.Generator) -> int:
    bounds = gu.graph_bounds(graph)
    return nearest_node(graph, bounds["center_lat"], bounds["center_lon"])


def sample_package_nodes(
    graph: nx.MultiDiGraph,
    n: int,
    rng: np.random.Generator,
    exclude: list[int] | None = None,
    reachable_from: int | None = None,
) -> list[int]:
    exclude = set(exclude or [])
    candidates = [nd for nd in graph.nodes if nd not in exclude]

    if reachable_from is not None:
        reachable = set(nx.descendants(graph, reachable_from)) | {reachable_from}
        candidates = [nd for nd in candidates if nd in reachable]

    if len(candidates) < n:
        idx = rng.choice(len(candidates), size=n, replace=True)
    else:
        idx = rng.choice(len(candidates), size=n, replace=False)
    return [int(candidates[i]) for i in idx]
