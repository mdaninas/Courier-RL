from __future__ import annotations

import numpy as np

from src.baselines.registry import BASELINE_NAMES, make_baseline
from src.envs.courier_routing_env import CourierRoutingRealMapEnv


def test_spaces_and_reset(config, graph):
    env = CourierRoutingRealMapEnv(config, graph=graph)
    K = config["simulation"]["candidate_package_count"]
    assert env.action_space.n == K
    assert env.observation_space.shape == (9 + 3 * K,)

    obs, info = env.reset(seed=1)
    assert obs.shape == (9 + 3 * K,)
    assert env.observation_space.contains(obs)
    assert info["remaining"] == config["simulation"]["num_packages"]


def test_sb3_env_checker(config, graph):
    import pytest

    pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.env_checker import check_env

    env = CourierRoutingRealMapEnv(config, graph=graph)
    check_env(env, warn=True, skip_render_check=True)


def test_step_delivers_and_observation_valid(config, graph):
    env = CourierRoutingRealMapEnv(config, graph=graph)
    env.reset(seed=2)
    obs, reward, terminated, truncated, info = env.step(0)
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert info["delivered"] == 1
    assert not (terminated and truncated)


def test_invalid_action_penalised(config, graph):
    env = CourierRoutingRealMapEnv(config, graph=graph)
    env.reset(seed=3)
    big_action = env.action_space.n - 1
    env.last_candidates = []
    obs, reward, terminated, truncated, info = env.step(big_action)
    assert info["invalid_action"] is True
    assert reward <= -config["reward"]["invalid_action_penalty"]


def test_full_episode_terminates(config, graph):
    env = CourierRoutingRealMapEnv(config, graph=graph)
    env.reset(seed=4)
    done = False
    steps = 0
    while not done and steps < env.max_steps + 5:
        action = 0 if env.last_candidates else 0
        _, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
    assert done
    assert "episode_metrics" in info
    m = info["episode_metrics"]
    assert 0 <= m["delivered"] <= config["simulation"]["num_packages"]


def test_baselines_run_full_episode(config, graph):
    for name in BASELINE_NAMES:
        env = CourierRoutingRealMapEnv(config, graph=graph)
        obs, _ = env.reset(seed=5)
        policy = make_baseline(name, config=config, seed=5)
        done = False
        guard = 0
        metrics = {}
        while not done and guard < 500:
            action = policy.act(env, obs)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            metrics = info.get("episode_metrics", metrics)
            guard += 1
        assert done, f"{name} did not finish"
        assert metrics["delivered"] >= 1, f"{name} delivered nothing"


def test_determinism_same_seed(config, graph):
    env = CourierRoutingRealMapEnv(config, graph=graph)
    obs_a, _ = env.reset(seed=99)
    pkgs_a = [p.destination_node for p in env.packages]
    obs_b, _ = env.reset(seed=99)
    pkgs_b = [p.destination_node for p in env.packages]
    assert pkgs_a == pkgs_b
    assert np.allclose(obs_a, obs_b)
