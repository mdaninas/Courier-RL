from __future__ import annotations

import copy
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import networkx as nx
import numpy as np

from src.agents.evaluate_agent import RLPolicy, load_rl_model, run_episode
from src.baselines.registry import BASELINE_NAMES, make_baseline
from src.envs.courier_routing_env import CourierRoutingRealMapEnv
from src.map import graph_utils as gu
from src.map import node_sampler, osm_loader
from src.simulation.package_generator import Package
from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger("interactive_app")

STATIC_DIR = Path(__file__).resolve().parent / "static"

AREAS = [
    {"key": "menteng", "label": "Menteng, Jakarta Pusat",
     "place": "Menteng, Jakarta, Indonesia", "cache": "data/maps/menteng_routing.graphml"},
    {"key": "kebayoran_baru", "label": "Kebayoran Baru, Jakarta Selatan",
     "place": "Kebayoran Baru, Jakarta, Indonesia", "cache": "data/maps/kebayoran_baru_routing.graphml"},
    {"key": "tebet", "label": "Tebet, Jakarta Selatan",
     "place": "Tebet, Jakarta, Indonesia", "cache": "data/maps/tebet_routing.graphml"},
]


class DemoState:

    def __init__(self, config: dict[str, Any], model=None, model_algo=None, label: str = ""):
        self.config = config
        self.label = label
        self.graph = osm_loader.load_map(config)
        self.bounds = gu.graph_bounds(self.graph)
        self.seed = int(config.get("project", {}).get("seed", 42))
        self.obs_dim = 9 + 3 * int(config["simulation"]["candidate_package_count"])
        if model is not None and tuple(model.observation_space.shape) == (self.obs_dim,):
            self.model, self.model_algo = model, model_algo
        else:
            self.model, self.model_algo = None, None
        self.boundary = self._load_boundary()
        self._compute_inside()

    def _compute_inside(self) -> None:
        # Nodes strictly inside the admin boundary are the only valid depot/package
        # locations.  Routing still uses the full (buffered) graph, so paths may pass
        # through roads outside the boundary -- "lewat boleh, taruh tidak".
        self._view_bounds = self.bounds
        self._inside = list(self.graph.nodes)
        if not self.boundary:
            return
        try:
            from shapely.geometry import Point, shape

            poly = shape(self.boundary)
            inside = [n for n, d in self.graph.nodes(data=True) if poly.contains(Point(d["x"], d["y"]))]
            if len(inside) >= 20:
                self._inside = inside
                minx, miny, maxx, maxy = poly.bounds
                self._view_bounds = {
                    "min_lat": miny, "max_lat": maxy, "min_lon": minx, "max_lon": maxx,
                    "center_lat": (miny + maxy) / 2, "center_lon": (minx + maxx) / 2,
                }
        except Exception as exc:
            logger.warning("inside-node compute failed: %s", exc)

    def _depot_inside(self) -> int:
        clat, clon = self._view_bounds["center_lat"], self._view_bounds["center_lon"]
        pool = self._inside or list(self.graph.nodes)
        return int(min(pool, key=lambda n: gu.haversine_m(
            clat, clon, self.graph.nodes[n]["y"], self.graph.nodes[n]["x"])))

    def _sample_inside(self, n: int, rng, exclude=()) -> list[int]:
        ex = set(exclude)
        pool = [x for x in self._inside if x not in ex] or list(self.graph.nodes)
        idx = rng.choice(len(pool), size=min(n, len(pool)), replace=len(pool) < n)
        return [int(pool[i]) for i in idx]

    def _load_boundary(self) -> dict[str, Any] | None:
        place = self.config["map"].get("place_name")
        if not place or self.graph.graph.get("synthetic"):
            return None
        cache = resolve_path(self.config["map"]["cache_path"]).with_suffix(".boundary.geojson")
        if cache.exists():
            try:
                return json.loads(cache.read_text(encoding="utf-8"))
            except Exception:
                pass
        try:
            import osmnx as ox
            from shapely.geometry import mapping

            geocode = getattr(ox, "geocode_to_gdf", None) or ox.geocoder.geocode_to_gdf
            geom = mapping(geocode(place).iloc[0].geometry)
            cache.write_text(json.dumps(geom), encoding="utf-8")
            return geom
        except Exception as exc:
            logger.warning("Boundary unavailable for %s: %s", place, exc)
            return None

    def snap(self, lat: float, lon: float) -> int:
        return node_sampler.nearest_node(self.graph, float(lat), float(lon))

    def calibrate_deadlines(self, depot_node: int, dest_nodes: list[int], priorities: list[str]) -> list[float]:
        from src.simulation.traffic_model import TrafficModel

        window = float(self.config["simulation"]["max_episode_minutes"])
        tour_factor = float(self.config["simulation"].get("deadline_tour_factor", 0.65))
        prio_factor = {"urgent": 0.6, "express": 0.8, "normal": 1.0}
        try:
            depot_times, _ = nx.single_source_dijkstra(self.graph, depot_node, weight="base_travel_time")
        except Exception:
            depot_times = {}
        reach = [float(depot_times.get(nd, window)) for nd in dest_nodes]
        n = max(1, len(dest_nodes))
        mean_leg = float(np.mean(reach)) if reach else window / n

        start_hour = int(self.config["simulation"].get("start_hour", 8))
        congestion = TrafficModel.get_time_based_multiplier(start_hour, self.config["traffic"])
        buffer = congestion * 1.25
        eff_leg = mean_leg * buffer
        tour_est = max(eff_leg, eff_leg * n * tour_factor)
        slot = 1.15
        out = []
        for r, prio in zip(reach, priorities):
            raw = tour_est * slot * prio_factor.get(prio, 1.0)
            out.append(round(float(np.clip(raw, r * buffer * 1.2 + 1.0, window)), 1))
        return out

    def policies(self) -> list[str]:
        names = list(BASELINE_NAMES)
        if self.model is not None:
            names.append("ppo_agent")
        return names

    def make_policy(self, name: str):
        if name == "ppo_agent" and self.model is not None:
            return RLPolicy(self.model, name=f"{self.model_algo}_agent")
        return make_baseline(name, config=self.config, seed=self.seed)

    def init_payload(self) -> dict[str, Any]:
        rng = np.random.default_rng(self.seed)
        vb = self._view_bounds
        depot = self._depot_inside()
        dlat, dlon = gu.node_latlon(self.graph, depot)
        pkg_nodes = self._sample_inside(6, rng, exclude=[depot])
        prios = ["normal", "express", "urgent", "normal", "express", "normal"]
        packages = []
        for i, nd in enumerate(pkg_nodes):
            lat, lon = gu.node_latlon(self.graph, nd)
            packages.append({"lat": lat, "lon": lon, "priority": prios[i % len(prios)]})
        return {
            "center": [vb["center_lat"], vb["center_lon"]],
            "bounds": [
                [vb["min_lat"], vb["min_lon"]],
                [vb["max_lat"], vb["max_lon"]],
            ],
            "depot": {"lat": dlat, "lon": dlon},
            "packages": packages,
            "policies": self.policies(),
            "synthetic": bool(self.graph.graph.get("synthetic", False)),
            "area": self.label or self.config["map"].get("place_name") or "Synthetic grid",
            "boundary": self.boundary,
            "start_hour": int(self.config["simulation"].get("start_hour", 8)),
            "has_model": self.model is not None,
        }

    def _build_scenario(self, payload: dict[str, Any]):
        depot_in = payload["depot"]
        pkgs_in = payload.get("packages", [])
        depot_node = self.snap(depot_in["lat"], depot_in["lon"])
        dest_nodes = [self.snap(p["lat"], p["lon"]) for p in pkgs_in]
        priorities = [p.get("priority", "normal") for p in pkgs_in]
        deadlines = self.calibrate_deadlines(depot_node, dest_nodes, priorities) if dest_nodes else []
        packages: list[Package] = [
            Package(package_id=i, destination_node=dest_nodes[i], deadline=deadlines[i], priority=priorities[i])
            for i in range(len(pkgs_in))
        ]
        return depot_node, packages

    def compare_policies(self, payload: dict[str, Any]) -> dict[str, Any]:
        depot_node, packages = self._build_scenario(payload)
        if not packages:
            return {"results": []}
        results = []
        for name in self.policies():
            env = CourierRoutingRealMapEnv(self.config, graph=self.graph)
            env.set_scenario(depot_node, packages)
            m = run_episode(env, self.make_policy(name), self.seed)
            tot = max(1, m["total_packages"])
            results.append({
                "policy": name,
                "reward": round(m["total_reward"], 1),
                "late": m["late_packages"],
                "delivered": f"{m['delivered']}/{m['total_packages']}",
                "time": round(m["total_travel_minutes"], 1),
                "ontime": round((m["delivered"] - m["late_packages"]) / tot * 100),
            })
        results.sort(key=lambda r: r["reward"], reverse=True)
        return {"results": results}

    def compute_route(self, payload: dict[str, Any]) -> dict[str, Any]:
        policy_name = payload.get("policy", "greedy_score")
        depot_node, packages = self._build_scenario(payload)
        snapped_depot = list(gu.node_latlon(self.graph, depot_node))
        if not packages:
            return {
                "route": [snapped_depot],
                "order": [],
                "depot": snapped_depot,
                "packages": [],
                "metrics": {"delivered": 0, "total_packages": 0, "late_packages": 0,
                            "total_travel_minutes": 0.0, "total_distance_km": 0.0,
                            "total_reward": 0.0, "avg_lateness": 0.0},
                "policy": policy_name,
            }

        env = CourierRoutingRealMapEnv(self.config, graph=self.graph)
        env.set_scenario(depot_node, packages)
        policy = self.make_policy(policy_name)
        metrics = run_episode(env, policy, self.seed)

        route_nodes = env.route_history()
        route_coords = [list(gu.node_latlon(self.graph, n)) for n in route_nodes]
        route_times = self._route_cumulative_times(route_nodes, metrics.get("total_travel_minutes", 0.0))
        stops = self._route_stops(route_nodes, env)

        snapped_pkgs = []
        seq_of = {pid: i + 1 for i, pid in enumerate(env.courier.delivered_packages)}
        for p in env.packages:
            lat, lon = gu.node_latlon(self.graph, p.destination_node)
            snapped_pkgs.append({
                "id": p.package_id, "lat": lat, "lon": lon, "priority": p.priority,
                "deadline": p.deadline, "delivered": p.delivered,
                "delivery_time": p.delivery_time, "late": p.is_late(),
                "seq": seq_of.get(p.package_id),
            })
        return {
            "route": route_coords,
            "route_times": route_times,
            "stops": stops,
            "order": env.courier.delivered_packages,
            "depot": snapped_depot,
            "packages": snapped_pkgs,
            "metrics": metrics,
            "policy": policy_name,
            "start_hour": int(self.config["simulation"].get("start_hour", 8)),
        }

    def _route_cumulative_times(self, route_nodes: list[int], total_minutes: float) -> list[float]:
        times = [0.0]
        for u, v in zip(route_nodes[:-1], route_nodes[1:]):
            data = self.graph.get_edge_data(u, v)
            t = min(d.get("current_travel_time", d.get("base_travel_time", 0.5)) for d in data.values()) if data else 0.5
            times.append(times[-1] + float(t))
        if times and times[-1] > 0 and total_minutes > 0:
            scale = total_minutes / times[-1]
            times = [round(t * scale, 3) for t in times]
        return times

    def _route_stops(self, route_nodes: list[int], env) -> list[dict[str, Any]]:
        node_of = {p.package_id: p.destination_node for p in env.packages}
        stops, search_from = [], 0
        for pid in env.courier.delivered_packages:
            target = node_of.get(pid)
            for idx in range(search_from, len(route_nodes)):
                if route_nodes[idx] == target:
                    stops.append({"i": idx, "id": pid})
                    search_from = idx
                    break
        return stops


class Demo:

    def __init__(self, base_config_path: str):
        self.base = load_config(base_config_path)
        self.model, self.model_algo = self._load_model()
        self.states: dict[str, DemoState] = {}

    def _load_model(self):
        obs_dim = 9 + 3 * int(self.base["simulation"]["candidate_package_count"])
        for name in ("ppo_courier.zip", "dqn_courier.zip"):
            p = resolve_path(f"data/checkpoints/{name}")
            if p.exists():
                try:
                    model, algo = load_rl_model(str(p))
                    if tuple(model.observation_space.shape) == (obs_dim,):
                        logger.info("Loaded compatible model %s (%s)", name, algo)
                        return model, algo
                except Exception as exc:
                    logger.warning("Could not load %s: %s", name, exc)
        return None, None

    def _area_config(self, area: dict[str, Any]) -> dict[str, Any]:
        cfg = copy.deepcopy(self.base)
        cfg["map"]["place_name"] = area["place"]
        cfg["map"]["cache_path"] = area["cache"]
        cfg["map"]["bbox"] = None
        cfg["map"]["force_synthetic"] = False
        return cfg

    def default_key(self) -> str:
        return AREAS[0]["key"]

    def areas(self) -> list[dict[str, str]]:
        return [{"key": a["key"], "label": a["label"]} for a in AREAS]

    def get(self, key: str | None) -> DemoState:
        area = next((a for a in AREAS if a["key"] == key), AREAS[0])
        if area["key"] not in self.states:
            self.states[area["key"]] = DemoState(
                self._area_config(area), self.model, self.model_algo, label=area["label"]
            )
        return self.states[area["key"]]

    def init_payload(self, key: str | None) -> dict[str, Any]:
        state = self.get(key)
        payload = state.init_payload()
        payload["areas"] = self.areas()
        payload["area_key"] = next((a["key"] for a in AREAS if a["key"] == key), self.default_key())
        return payload


def make_handler(demo: Demo):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send_json(self, obj: Any, code: int = 200) -> None:
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, path: Path, ctype: str) -> None:
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path in ("/", "/index.html"):
                self._send_file(STATIC_DIR / "interactive.html", "text/html; charset=utf-8")
            elif parsed.path == "/api/areas":
                self._send_json(demo.areas())
            elif parsed.path == "/api/init":
                key = parse_qs(parsed.query).get("area", [demo.default_key()])[0]
                try:
                    self._send_json(demo.init_payload(key))
                except Exception as exc:
                    logger.exception("init failed")
                    self._send_json({"error": str(exc)}, code=500)
            else:
                self.send_error(404)

        def do_POST(self):
            path = urlparse(self.path).path
            if path not in ("/api/route", "/api/compare"):
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            try:
                state = demo.get(payload.get("area"))
                result = state.compare_policies(payload) if path == "/api/compare" else state.compute_route(payload)
                self._send_json(result)
            except Exception as exc:
                logger.exception("request failed")
                self._send_json({"error": str(exc)}, code=500)

    return Handler


def serve(config_path: str, host: str = "127.0.0.1", port: int = 8000, open_browser: bool = True) -> None:
    demo = Demo(config_path)
    demo.get(demo.default_key())
    handler = make_handler(demo)
    httpd = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{port}"
    logger.info("Interactive demo serving at %s  (%d areas)", url, len(AREAS))
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()
