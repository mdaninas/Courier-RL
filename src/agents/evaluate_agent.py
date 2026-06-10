from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.baselines.registry import make_baseline
from src.envs.courier_routing_env import CourierRoutingRealMapEnv
from src.map import osm_loader
from src.utils.config import resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)

METRIC_KEYS = [
    "delivered",
    "completion_rate",
    "total_travel_minutes",
    "total_distance_km",
    "late_packages",
    "avg_lateness",
    "avg_time_per_package",
    "total_reward",
]


class RLPolicy:

    name = "rl_agent"

    def __init__(self, model, name: str = "rl_agent"):
        self.model = model
        self.name = name

    def reset(self, seed=None):
        pass

    def act(self, env, obs) -> int:
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)


def load_rl_model(model_path: str, algo: str | None = None):
    model_path = str(resolve_path(model_path))
    algos = [algo.upper()] if algo else []
    algos += [a for a in ("PPO", "DQN") if a not in algos]
    last_exc = None
    for a in algos:
        try:
            if a == "PPO":
                from stable_baselines3 import PPO

                return PPO.load(model_path), "PPO"
            if a == "DQN":
                from stable_baselines3 import DQN

                return DQN.load(model_path), "DQN"
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(f"Could not load model {model_path}: {last_exc}")


def run_episode(env: CourierRoutingRealMapEnv, policy, seed: int) -> dict[str, Any]:
    obs, _ = env.reset(seed=seed)
    policy.reset(seed=seed)
    done = False
    metrics: dict[str, Any] = {}
    while not done:
        action = policy.act(env, obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if "episode_metrics" in info:
            metrics = info["episode_metrics"]
    return metrics


def evaluate(
    config: dict[str, Any],
    model_path: str | None = None,
    algo: str | None = None,
    episodes: int | None = None,
    methods: list[str] | None = None,
    results_dir: str | None = None,
) -> dict[str, Any]:
    eval_cfg = config["evaluation"]
    n_episodes = int(episodes or eval_cfg.get("episodes", 50))
    baseline_names = methods if methods is not None else list(eval_cfg.get("compare_with", []))
    seed0 = int(config.get("project", {}).get("seed", 42))

    out_dir = resolve_path(results_dir or eval_cfg.get("results_dir", "data/results"))
    out_dir.mkdir(parents=True, exist_ok=True)

    graph = osm_loader.load_map(config)
    env = CourierRoutingRealMapEnv(config, graph=graph)

    policies: dict[str, Any] = {}
    for name in baseline_names:
        policies[name] = make_baseline(name, config=config, seed=seed0)
    rl_label = None
    if model_path:
        model, detected_algo = load_rl_model(model_path, algo)
        rl_label = f"{detected_algo}_agent"
        policies[rl_label] = RLPolicy(model, name=rl_label)

    seeds = [seed0 + i for i in range(n_episodes)]

    per_episode: list[dict[str, Any]] = []
    summary: dict[str, dict[str, float]] = {}
    for method, policy in policies.items():
        logger.info("Evaluating '%s' over %d episodes ...", method, n_episodes)
        rows = []
        for ep, sd in enumerate(seeds):
            m = run_episode(env, policy, sd)
            m_row = {"method": method, "episode": ep, "seed": sd, **m}
            rows.append(m_row)
            per_episode.append(m_row)
        summary[method] = _aggregate(rows)

    csv_path = out_dir / "evaluation_per_episode.csv"
    _write_csv(per_episode, csv_path)
    json_path = out_dir / "evaluation_summary.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    table = comparison_table(summary, rl_label=rl_label)
    table_path = out_dir / "comparison_table.md"
    table_path.write_text(table, encoding="utf-8")

    logger.info("Saved per-episode CSV -> %s", csv_path)
    logger.info("Saved summary JSON   -> %s", json_path)
    logger.info("Comparison table:\n%s", table)

    return {
        "summary": summary,
        "csv": str(csv_path),
        "json": str(json_path),
        "table_md": str(table_path),
        "table": table,
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    out = {}
    for key in METRIC_KEYS:
        vals = [float(r.get(key, 0.0)) for r in rows if r.get(key) is not None]
        out[key] = round(float(np.mean(vals)), 3) if vals else 0.0
        if key == "total_reward" and len(vals) > 1:
            sd = float(np.std(vals, ddof=1))
            out["total_reward_std"] = round(sd, 2)
            out["total_reward_ci95"] = round(1.96 * sd / np.sqrt(len(vals)), 2)
    tot = [float(r.get("total_packages", 0.0)) for r in rows if r.get("total_packages")]
    if tot:
        on_time = out["delivered"] - out["late_packages"]
        out["ontime_rate"] = round(max(0.0, on_time) / (sum(tot) / len(tot)), 3)
    out["episodes"] = len(rows)
    return out


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fieldnames = list({k for r in rows for k in r.keys()})
    preferred = ["method", "episode", "seed"] + METRIC_KEYS + ["invalid_actions", "total_packages"]
    fieldnames = [f for f in preferred if f in fieldnames] + [f for f in fieldnames if f not in preferred]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def comparison_table(summary: dict[str, dict[str, float]], rl_label: str | None = None) -> str:
    header = (
        "| Method | Delivered | On-time | Total Time (min) | Late Pkgs | Avg Lateness | Reward (mean ± 95% CI) |\n"
        "|--------|-----------|---------|------------------|-----------|--------------|------------------------|\n"
    )
    order = [m for m in summary if m != rl_label]
    if rl_label:
        order.append(rl_label)
    best = max(summary, key=lambda m: summary[m]["total_reward"]) if summary else None
    lines = []
    for method in order:
        s = summary[method]
        ci = s.get("total_reward_ci95", 0.0)
        ot = s.get("ontime_rate", 0.0) * 100
        star = " ★" if method == best else ""
        lines.append(
            f"| {method}{star} | {s['delivered']:.1f} | {ot:.0f}% | {s['total_travel_minutes']:.1f} | "
            f"{s['late_packages']:.1f} | {s['avg_lateness']:.1f} | {s['total_reward']:.0f} ± {ci:.0f} |"
        )
    return header + "\n".join(lines) + "\n"
