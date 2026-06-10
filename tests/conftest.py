from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.map import osm_loader
from src.utils.config import load_config


@pytest.fixture()
def config(tmp_path):
    cfg = load_config()
    cfg["map"]["force_synthetic"] = True
    cfg["map"]["synthetic_grid_size"] = 8
    cfg["map"]["cache_path"] = str(tmp_path / "synthetic.graphml")
    cfg["simulation"]["num_packages"] = 8
    cfg["simulation"]["candidate_package_count"] = 5
    cfg["simulation"]["max_episode_minutes"] = 480
    cfg["project"]["seed"] = 123
    return cfg


@pytest.fixture()
def graph(config):
    return osm_loader.load_map(config)
