from __future__ import annotations

import math

import networkx as nx

EARTH_RADIUS_M = 6_371_000.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


def node_coords(graph: nx.Graph) -> dict[int, tuple[float, float]]:
    return {n: (data["x"], data["y"]) for n, data in graph.nodes(data=True)}


def node_latlon(graph: nx.Graph, node: int) -> tuple[float, float]:
    data = graph.nodes[node]
    return float(data["y"]), float(data["x"])


def graph_bounds(graph: nx.Graph) -> dict[str, float]:
    xs = [d["x"] for _, d in graph.nodes(data=True)]
    ys = [d["y"] for _, d in graph.nodes(data=True)]
    return {
        "min_lon": min(xs),
        "max_lon": max(xs),
        "min_lat": min(ys),
        "max_lat": max(ys),
        "center_lat": (min(ys) + max(ys)) / 2,
        "center_lon": (min(xs) + max(xs)) / 2,
    }


def largest_strongly_connected(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    if graph.is_directed():
        components = list(nx.strongly_connected_components(graph))
    else:
        components = list(nx.connected_components(graph))
    if not components:
        return graph
    largest = max(components, key=len)
    return graph.subgraph(largest).copy()


def enrich_edge_attributes(graph: nx.MultiDiGraph, default_speed_kph: float = 30.0) -> nx.MultiDiGraph:
    for u, v, _key, data in graph.edges(keys=True, data=True):
        length = float(data.get("length", 0.0) or 0.0)
        if length <= 0:
            lat1, lon1 = node_latlon(graph, u)
            lat2, lon2 = node_latlon(graph, v)
            length = max(1.0, haversine_m(lat1, lon1, lat2, lon2))
            data["length"] = length

        speed = data.get("speed_kph", data.get("base_speed_kph"))
        if speed is None:
            maxspeed = data.get("maxspeed")
            speed = _parse_maxspeed(maxspeed, default_speed_kph)
        speed = float(speed)
        data["base_speed_kph"] = speed

        if "travel_time" in data and data["travel_time"]:
            base_minutes = float(data["travel_time"]) / 60.0
        else:
            base_minutes = (length / 1000.0) / max(speed, 1e-6) * 60.0
        base_minutes = max(base_minutes, 1e-3)
        data["base_travel_time"] = base_minutes

        try:
            mult = float(data.get("traffic_multiplier", 1.0))
        except (TypeError, ValueError):
            mult = 1.0
        data["traffic_multiplier"] = mult
        data["current_travel_time"] = base_minutes * mult
    return graph


def _parse_maxspeed(maxspeed, default: float) -> float:
    if maxspeed is None:
        return default
    if isinstance(maxspeed, (list, tuple)):
        vals = [_parse_maxspeed(m, default) for m in maxspeed]
        return sum(vals) / len(vals) if vals else default
    try:
        return float(str(maxspeed).split()[0])
    except (ValueError, IndexError):
        return default


def edge_midpoints(graph: nx.MultiDiGraph) -> dict[tuple[int, int, int], tuple[float, float]]:
    mids: dict[tuple[int, int, int], tuple[float, float]] = {}
    for u, v, key in graph.edges(keys=True):
        lat1, lon1 = node_latlon(graph, u)
        lat2, lon2 = node_latlon(graph, v)
        mids[(u, v, key)] = ((lat1 + lat2) / 2.0, (lon1 + lon2) / 2.0)
    return mids


def single_source_travel_times(
    graph: nx.MultiDiGraph, source: int, weight: str = "current_travel_time"
) -> tuple[dict[int, float], dict[int, list[int]]]:
    distances, paths = nx.single_source_dijkstra(graph, source, weight=weight)
    return distances, paths


def shortest_route(
    graph: nx.MultiDiGraph, source: int, target: int, weight: str = "current_travel_time"
) -> list[int]:
    return nx.shortest_path(graph, source=source, target=target, weight=weight)


def route_length_m(graph: nx.MultiDiGraph, route: list[int]) -> float:
    total = 0.0
    for u, v in zip(route[:-1], route[1:]):
        edge = _min_edge(graph, u, v, "length")
        total += float(edge.get("length", 0.0))
    return total


def route_travel_minutes(
    graph: nx.MultiDiGraph, route: list[int], weight: str = "current_travel_time"
) -> float:
    total = 0.0
    for u, v in zip(route[:-1], route[1:]):
        edge = _min_edge(graph, u, v, weight)
        total += float(edge.get(weight, 0.0))
    return total


def _min_edge(graph: nx.MultiDiGraph, u: int, v: int, weight: str) -> dict:
    datadict = graph.get_edge_data(u, v)
    if datadict is None:
        return {}
    return min(datadict.values(), key=lambda d: d.get(weight, float("inf")))
