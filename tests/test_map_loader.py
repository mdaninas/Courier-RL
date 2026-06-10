from __future__ import annotations

import networkx as nx

from src.map import graph_utils as gu
from src.map import node_sampler


def test_synthetic_graph_built(graph):
    assert isinstance(graph, nx.MultiDiGraph)
    assert graph.number_of_nodes() == 8 * 8
    assert graph.number_of_edges() > 0
    assert graph.graph.get("synthetic") is True


def test_edges_have_required_attributes(graph):
    required = {"length", "base_speed_kph", "base_travel_time", "traffic_multiplier", "current_travel_time"}
    for _, _, data in graph.edges(data=True):
        assert required.issubset(data.keys())
        assert data["length"] > 0
        assert data["base_travel_time"] > 0
        assert data["current_travel_time"] >= data["base_travel_time"] - 1e-9


def test_strongly_connected(graph):
    assert nx.is_strongly_connected(graph)


def test_nearest_node_and_sampling(graph):
    bounds = gu.graph_bounds(graph)
    node = node_sampler.nearest_node(graph, bounds["center_lat"], bounds["center_lon"])
    assert node in graph.nodes

    import numpy as np

    rng = np.random.default_rng(0)
    nodes = node_sampler.sample_package_nodes(graph, 5, rng, exclude=[node])
    assert len(nodes) == 5
    assert node not in nodes
    assert all(n in graph.nodes for n in nodes)


def test_shortest_route(graph):
    nodes = list(graph.nodes)
    route = gu.shortest_route(graph, nodes[0], nodes[-1])
    assert route[0] == nodes[0] and route[-1] == nodes[-1]
    assert gu.route_travel_minutes(graph, route) > 0
