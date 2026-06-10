from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx

from src.map import graph_utils as gu
from src.utils.config import resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import osmnx as ox

    _HAS_OSMNX = True
except Exception:
    ox = None
    _HAS_OSMNX = False


def load_map(config: dict[str, Any], force_download: bool = False) -> nx.MultiDiGraph:
    map_cfg = config["map"]
    cache_path = resolve_path(map_cfg["cache_path"])

    if map_cfg.get("force_synthetic"):
        logger.info("force_synthetic=True -> building synthetic grid.")
        return _finalise(build_synthetic_grid(config))

    if cache_path.exists() and not force_download:
        logger.info("Loading cached map from %s", cache_path)
        try:
            graph = _load_graphml(cache_path)
            return _finalise(graph)
        except Exception as exc:
            logger.warning("Failed to read cache (%s); will try to rebuild.", exc)

    if _HAS_OSMNX:
        try:
            graph = download_osm_graph(
                place_name=map_cfg.get("place_name"),
                bbox=map_cfg.get("bbox"),
                network_type=map_cfg.get("network_type", "drive"),
            )
            graph = _finalise(graph)
            save_graph(graph, cache_path)
            return graph
        except Exception as exc:
            logger.warning("OSM download failed (%s).", exc)
            if not map_cfg.get("allow_synthetic_fallback", True):
                raise

    if not map_cfg.get("allow_synthetic_fallback", True):
        raise RuntimeError("Map unavailable and synthetic fallback disabled.")
    logger.warning("Using SYNTHETIC street-grid fallback (not real OSM data).")
    graph = build_synthetic_grid(config)
    graph = _finalise(graph)
    save_graph(graph, cache_path)
    return graph


def download_osm_graph(
    place_name: str | None = None,
    bbox: list | None = None,
    network_type: str = "drive",
) -> nx.MultiDiGraph:
    if not _HAS_OSMNX:
        raise RuntimeError("osmnx is not installed.")

    if place_name:
        logger.info("Downloading OSM graph for place: %s", place_name)
        graph = ox.graph_from_place(place_name, network_type=network_type)
    elif bbox:
        north, south, east, west = bbox
        logger.info("Downloading OSM graph for bbox N%.4f S%.4f E%.4f W%.4f", north, south, east, west)
        graph = ox.graph_from_bbox((west, south, east, north), network_type=network_type)
    else:
        raise ValueError("Provide either place_name or bbox in map config.")

    graph = _add_speeds_and_times(graph)
    return graph


def save_graph(graph: nx.MultiDiGraph, path: Path | str) -> Path:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if _HAS_OSMNX:
        try:
            ox.save_graphml(graph, filepath=str(path))
            return path
        except Exception as exc:
            logger.warning("osmnx save failed (%s); using networkx writer.", exc)
    _write_graphml_networkx(graph, path)
    return path


def _add_speeds_and_times(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    add_speeds = getattr(getattr(ox, "routing", ox), "add_edge_speeds", None)
    add_times = getattr(getattr(ox, "routing", ox), "add_edge_travel_times", None)
    try:
        if add_speeds:
            graph = add_speeds(graph)
        if add_times:
            graph = add_times(graph)
    except Exception as exc:
        logger.warning("Could not add osmnx speeds/times (%s); deriving instead.", exc)
    return graph


def _finalise(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    graph = gu.largest_strongly_connected(graph)
    graph = gu.enrich_edge_attributes(graph)
    logger.info("Map ready: %d nodes, %d edges", graph.number_of_nodes(), graph.number_of_edges())
    return graph


def _load_graphml(path: Path) -> nx.MultiDiGraph:
    if _HAS_OSMNX:
        try:
            return ox.load_graphml(filepath=str(path))
        except Exception:
            pass
    graph = nx.read_graphml(str(path))
    return _coerce_numeric(nx.MultiDiGraph(graph))


def _write_graphml_networkx(graph: nx.MultiDiGraph, path: Path) -> None:
    g = graph.copy()
    for _, _, data in g.edges(data=True):
        for k, v in list(data.items()):
            if isinstance(v, (list, tuple)):
                data[k] = ",".join(map(str, v))
            elif v is None:
                data[k] = ""
    for _, data in g.nodes(data=True):
        for k, v in list(data.items()):
            if isinstance(v, (list, tuple)):
                data[k] = ",".join(map(str, v))
            elif v is None:
                data[k] = ""
    nx.write_graphml(g, str(path))


def _coerce_numeric(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    # GraphML stores node ids as strings; osmnx keeps them as ints. Normalise to
    # int so downstream int(node) lookups match (matters when osmnx is absent).
    try:
        graph = nx.relabel_nodes(graph, {n: int(n) for n in graph.nodes})
    except (TypeError, ValueError):
        pass
    for _, data in graph.nodes(data=True):
        for key in ("x", "y"):
            if key in data:
                data[key] = float(data[key])
    for _, _, data in graph.edges(data=True):
        for key in ("length", "base_speed_kph", "base_travel_time", "traffic_multiplier", "current_travel_time", "travel_time", "speed_kph"):
            if key in data and data[key] not in (None, ""):
                try:
                    data[key] = float(data[key])
                except (TypeError, ValueError):
                    pass
    return graph


def build_synthetic_grid(config: dict[str, Any]) -> nx.MultiDiGraph:
    n = int(config["map"].get("synthetic_grid_size", 12))
    lat0, lat1 = -6.215, -6.185
    lon0, lon1 = 106.815, 106.845

    graph = nx.MultiDiGraph()
    graph.graph["crs"] = "epsg:4326"
    graph.graph["synthetic"] = True

    def node_id(r: int, c: int) -> int:
        return r * n + c

    for r in range(n):
        for c in range(n):
            lat = lat0 + (lat1 - lat0) * (r / (n - 1))
            lon = lon0 + (lon1 - lon0) * (c / (n - 1))
            graph.add_node(node_id(r, c), x=lon, y=lat)

    speed = 30.0
    for r in range(n):
        for c in range(n):
            here = node_id(r, c)
            neighbours = []
            if c + 1 < n:
                neighbours.append(node_id(r, c + 1))
            if r + 1 < n:
                neighbours.append(node_id(r + 1, c))
            for other in neighbours:
                lat1n, lon1n = graph.nodes[here]["y"], graph.nodes[here]["x"]
                lat2n, lon2n = graph.nodes[other]["y"], graph.nodes[other]["x"]
                length = gu.haversine_m(lat1n, lon1n, lat2n, lon2n)
                for a, b in ((here, other), (other, here)):
                    graph.add_edge(
                        a, b, 0,
                        length=length,
                        base_speed_kph=speed,
                        speed_kph=speed,
                    )
    return graph
