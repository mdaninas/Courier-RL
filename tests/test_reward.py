from __future__ import annotations

from src.envs.courier_routing_env import CourierRoutingRealMapEnv
from src.simulation import package_generator as pkg_gen


def test_priority_multipliers(config):
    assert pkg_gen.priority_multiplier("normal", config) == 1.0
    assert pkg_gen.priority_multiplier("express", config) == config["reward"]["express_multiplier"]
    assert pkg_gen.priority_multiplier("urgent", config) == config["reward"]["urgent_multiplier"]


def test_delivery_reward_breakdown(config, graph):
    env = CourierRoutingRealMapEnv(config, graph=graph)
    env.reset(seed=7)
    _, reward, _, _, info = env.step(0)
    bd = info["reward_breakdown"]
    base = config["reward"]["base_delivery_reward"]
    assert bd["delivery_reward"] >= base
    assert bd["travel_penalty"] <= 0
    assert bd["lateness_penalty"] <= 0
    recomputed = (
        bd["delivery_reward"]
        + bd["travel_penalty"]
        + bd["distance_penalty"]
        + bd["lateness_penalty"]
    )
    assert abs(recomputed - reward) < 1e-6


def test_completion_bonus_applied(config, graph):
    env = CourierRoutingRealMapEnv(config, graph=graph)
    env.reset(seed=8)
    total = 0.0
    done = False
    guard = 0
    got_bonus_step = False
    while not done and guard < 500:
        _, reward, terminated, truncated, _ = env.step(0)
        total += reward
        if terminated:
            got_bonus_step = True
        done = terminated or truncated
        guard += 1
    if got_bonus_step:
        assert total > config["reward"]["completion_bonus"] - 1e6


def test_lateness_penalty_scales(config, graph):
    env = CourierRoutingRealMapEnv(config, graph=graph)
    env.reset(seed=11)
    for p in env.remaining:
        p.deadline = 0.0
    _, reward, _, _, info = env.step(0)
    bd = info["reward_breakdown"]
    assert bd["lateness"] > 0
    assert bd["lateness_penalty"] < 0


def test_lateness_priority_weighting(config, graph):
    config["simulation"]["shuffle_candidates"] = False
    env = CourierRoutingRealMapEnv(config, graph=graph)
    env.reset(seed=11)
    for p in env.remaining:
        p.deadline = 0.0
        p.priority = "urgent"
    _, _, _, _, info = env.step(0)
    bd = info["reward_breakdown"]
    w = config["reward"]["lateness_penalty_weight"]
    factor = config["reward"]["lateness_priority"]["urgent"]
    assert abs(bd["lateness_penalty"] - (-w * factor * bd["lateness"])) < 1e-6
