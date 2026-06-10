from __future__ import annotations

from src.envs.courier_routing_env import CourierRoutingRealMapEnv
from src.simulation.package_generator import Package


def test_set_scenario_uses_fixed_packages(config, graph):
    nodes = list(graph.nodes)
    depot = nodes[0]
    pkgs = [
        Package(package_id=0, destination_node=nodes[5], deadline=120.0, priority="urgent"),
        Package(package_id=1, destination_node=nodes[10], deadline=200.0, priority="normal"),
        Package(package_id=2, destination_node=nodes[20], deadline=300.0, priority="express"),
    ]
    env = CourierRoutingRealMapEnv(config, graph=graph)
    env.set_scenario(depot, pkgs)
    env.reset(seed=1)

    assert env.depot_node == depot
    assert env.num_packages == 3
    assert sorted(p.destination_node for p in env.packages) == sorted([nodes[5], nodes[10], nodes[20]])
    assert all(not p.delivered for p in env.packages)
    assert pkgs[0].delivered is False


def test_demo_state_compute_route():
    from src.utils.config import load_config
    from src.visualization.interactive_app import DemoState

    state = DemoState(load_config("config/synthetic_smoke.yaml"))
    init = state.init_payload()
    assert "greedy_score" in init["policies"]
    assert len(init["packages"]) > 0

    depot = init["depot"]
    payload = {"depot": depot, "packages": init["packages"], "policy": "nearest_neighbor"}
    res = state.compute_route(payload)

    assert len(res["route"]) > 1
    assert res["metrics"]["delivered"] == len(init["packages"])
    assert len(res["packages"]) == len(init["packages"])
    assert all(p["seq"] is not None for p in res["packages"])


def test_demo_state_empty_packages():
    from src.utils.config import load_config
    from src.visualization.interactive_app import DemoState

    state = DemoState(load_config("config/synthetic_smoke.yaml"))
    depot = state.init_payload()["depot"]
    res = state.compute_route({"depot": depot, "packages": [], "policy": "greedy_score"})
    assert res["metrics"]["delivered"] == 0
    assert len(res["route"]) == 1
