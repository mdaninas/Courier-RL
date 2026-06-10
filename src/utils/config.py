from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_TRAINING_FILE = PROJECT_ROOT / "config" / "default_training.yaml"

DEFAULTS: dict[str, Any] = {
    "project": {"name": "adaptive-courier-routing-rl", "seed": 42},
    "map": {
        "place_name": "Menteng, Jakarta, Indonesia",
        "bbox": None,
        "network_type": "drive",
        "cache_path": "data/maps/menteng_drive.graphml",
        "allow_synthetic_fallback": True,
        "synthetic_grid_size": 12,
    },
    "simulation": {
        "num_packages": 20,
        "max_episode_minutes": 480,
        "start_hour": 8,
        "candidate_package_count": 10,
        "max_steps": 200,
        "return_to_depot": False,
        "deadline_tour_factor": 0.65,
        "deadline_slack_min": 0.35,
        "deadline_slack_max": 1.25,
        "shuffle_candidates": False,
    },
    "traffic": {
        "enable_dynamic_traffic": True,
        "rush_hour_morning": [7, 9],
        "rush_hour_evening": [17, 19],
        "midday_window": [12, 13],
        "base_multiplier": 1.0,
        "max_multiplier": 3.5,
        "random_incident_probability": 0.05,
        "update_every_minutes": 15,
        "hotspots": None,
        "num_auto_hotspots": 3,
    },
    "reward": {
        "base_delivery_reward": 100,
        "normal_multiplier": 1.0,
        "express_multiplier": 1.5,
        "urgent_multiplier": 2.0,
        "travel_time_penalty_weight": 0.2,
        "distance_penalty_weight": 0.0,
        "lateness_penalty_weight": 4.0,
        "lateness_priority": {"normal": 1.0, "express": 1.3, "urgent": 1.6},
        "completion_bonus": 500,
        "invalid_action_penalty": 100,
    },
    "training": {
        "algorithm": "PPO",
        "total_timesteps": 200000,
        "learning_rate": 0.0003,
        "gamma": 0.99,
        "n_steps": 2048,
        "batch_size": 64,
        "ent_coef": 0.01,
        "net_arch": [128, 128],
        "save_path": "data/checkpoints/ppo_courier.zip",
        "log_dir": "data/logs",
        "tensorboard": False,
    },
    "evaluation": {
        "episodes": 50,
        "results_dir": "data/results",
        "compare_with": ["random", "nearest_neighbor", "earliest_deadline", "greedy_score"],
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _read_yaml(path: os.PathLike | str) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_config(config_path: os.PathLike | str | None = None) -> dict[str, Any]:
    cfg = deep_merge(DEFAULTS, _read_yaml(DEFAULT_TRAINING_FILE))
    if config_path is not None:
        cfg = deep_merge(cfg, _read_yaml(config_path))

    cfg["_meta"] = {
        "config_path": str(config_path) if config_path else None,
        "project_root": str(PROJECT_ROOT),
    }
    return cfg


def resolve_path(path: str | os.PathLike) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def save_config(cfg: dict[str, Any], path: os.PathLike | str) -> None:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dump = {k: v for k, v in cfg.items() if not k.startswith("_")}
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(dump, fh, sort_keys=False, allow_unicode=True)
