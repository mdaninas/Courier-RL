from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.agents.evaluate_agent import RLPolicy, load_rl_model, run_episode
from src.baselines.registry import BASELINE_NAMES, make_baseline
from src.envs.courier_routing_env import CourierRoutingRealMapEnv
from src.map import osm_loader
from src.utils.config import load_config, resolve_path
from src.visualization.folium_map import render_episode_map

AREAS = {
    "Menteng, Jakarta (real OSM)": "config/jakarta_menteng.yaml",
    "Synthetic grid (offline)": "config/synthetic_smoke.yaml",
}

POLICY_LABELS = {
    "random": "Random",
    "nearest_neighbor": "Nearest Neighbor",
    "earliest_deadline": "Earliest Deadline",
    "greedy_score": "Greedy Score",
    "ppo_agent": "PPO Agent (trained)",
}


@st.cache_resource(show_spinner="Loading road network ...")
def get_graph(config_path: str):
    cfg = load_config(config_path)
    return osm_loader.load_map(cfg)


@st.cache_resource(show_spinner="Loading trained model ...")
def get_model(model_path: str, algo: str | None):
    model, detected = load_rl_model(model_path, algo)
    return model, detected


def find_checkpoint() -> str | None:
    for name in ("ppo_courier.zip", "dqn_courier.zip", "ppo_smoke.zip"):
        p = resolve_path(f"data/checkpoints/{name}")
        if p.exists():
            return str(p)
    return None


def build_config(base_path: str, overrides: dict[str, Any]) -> dict[str, Any]:
    cfg = load_config(base_path)
    cfg["simulation"]["num_packages"] = overrides["num_packages"]
    cfg["simulation"]["start_hour"] = overrides["start_hour"]
    cfg["simulation"]["deadline_tour_factor"] = overrides["tour_factor"]
    cfg["traffic"]["enable_dynamic_traffic"] = overrides["traffic"]
    cfg["project"]["seed"] = overrides["seed"]
    return cfg


def make_policy(name: str, cfg: dict[str, Any], seed: int, model_path: str | None):
    if name == "ppo_agent":
        model, detected = get_model(model_path, None)
        return RLPolicy(model, name=f"{detected}_agent")
    return make_baseline(name, config=cfg, seed=seed)


class IncompatibleModel(Exception):
    pass


def run_one(cfg: dict[str, Any], graph, policy_name: str, seed: int, model_path: str | None):
    env = CourierRoutingRealMapEnv(cfg, graph=graph)
    policy = make_policy(policy_name, cfg, seed, model_path)
    if isinstance(policy, RLPolicy):
        want = env.observation_space.shape
        got = tuple(policy.model.observation_space.shape)
        if tuple(want) != got:
            raise IncompatibleModel(
                f"Trained model expects observation {got} but this scenario needs {want}. "
                f"The model was trained with a different candidate count (K). "
                f"Train a model for this area (scripts/train.py --config …) or pick a baseline."
            )
    metrics = run_episode(env, policy, seed)
    return env, metrics


def render_map_html(env, metrics, label: str) -> str:
    tmp = Path(tempfile.gettempdir()) / "courier_dashboard_map.html"
    render_episode_map(env, env.route_history(), save_path=str(tmp), title=label, metrics=metrics)
    return tmp.read_text(encoding="utf-8")


def main() -> None:
    st.set_page_config(page_title="Adaptive Courier RL", layout="wide", page_icon="🛵")
    st.title("🛵 Adaptive Courier Routing — Interactive Demo")
    st.caption("Change the controls on the left — the route and metrics update live.")

    sb = st.sidebar
    sb.header("⚙️ Scenario")
    area_label = sb.selectbox("Area / map", list(AREAS.keys()))
    config_path = AREAS[area_label]

    ckpt = find_checkpoint()
    policy_options = list(BASELINE_NAMES) + (["ppo_agent"] if ckpt else [])

    mode = sb.radio("Mode", ["Single policy", "Compare all policies"], horizontal=False)

    if mode == "Single policy":
        policy_name = sb.selectbox(
            "Policy", policy_options, format_func=lambda x: POLICY_LABELS.get(x, x),
            index=policy_options.index("greedy_score") if "greedy_score" in policy_options else 0,
        )
    else:
        policy_name = None

    seed = sb.slider("Random seed", 0, 200, 42, help="Different seed = different package layout")
    num_packages = sb.slider("Number of packages", 5, 40, 20)
    start_hour = sb.slider("Start hour (clock)", 0, 23, 8, help="Affects rush-hour traffic")
    tour_factor = sb.slider("Deadline tightness", 0.30, 1.50, 0.65, 0.05,
                            help="Lower = tighter deadlines = more pressure")
    traffic = sb.checkbox("Dynamic traffic", value=True)

    if not ckpt:
        sb.info("No trained model found yet. Train one with `scripts/train.py` to enable the PPO Agent.")
    else:
        sb.success(f"Model: {Path(ckpt).name}")

    overrides = dict(
        seed=seed, num_packages=num_packages, start_hour=start_hour,
        tour_factor=tour_factor, traffic=traffic,
    )
    cfg = build_config(config_path, overrides)
    graph = get_graph(config_path)

    if mode == "Single policy":
        try:
            with st.spinner(f"Running {POLICY_LABELS.get(policy_name, policy_name)} ..."):
                env, m = run_one(cfg, graph, policy_name, seed, ckpt)
        except IncompatibleModel as exc:
            st.error(str(exc))
            st.stop()
        _metric_row(m)
        html = render_map_html(env, m, POLICY_LABELS.get(policy_name, policy_name))
        st.components.v1.html(html, height=620, scrolling=False)
        st.caption("▶ Press play (bottom of map) to animate the courier along the real streets.")

    else:
        rows: list[dict[str, Any]] = []
        envs: dict[str, Any] = {}
        prog = st.progress(0.0, text="Running all policies on the same scenario ...")
        for i, name in enumerate(policy_options):
            try:
                env, m = run_one(cfg, graph, name, seed, ckpt)
            except IncompatibleModel:
                prog.progress((i + 1) / len(policy_options))
                continue
            envs[name] = (env, m)
            rows.append({"Policy": POLICY_LABELS.get(name, name), **_short_metrics(m)})
            prog.progress((i + 1) / len(policy_options))
        prog.empty()
        if "ppo_agent" in policy_options and "ppo_agent" not in envs:
            st.warning("PPO Agent skipped: trained model is incompatible with this area's K. "
                       "Train one for this area to include it.")

        st.subheader("📊 Comparison (same packages, same traffic)")
        st.dataframe(rows, use_container_width=True, hide_index=True)
        _comparison_bars(rows)

        viewable = list(envs.keys())
        view = st.selectbox("View route for", viewable,
                            format_func=lambda x: POLICY_LABELS.get(x, x))
        env, m = envs[view]
        _metric_row(m)
        html = render_map_html(env, m, POLICY_LABELS.get(view, view))
        st.components.v1.html(html, height=620, scrolling=False)


def _metric_row(m: dict[str, Any]) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Delivered", f"{m['delivered']}/{m['total_packages']}")
    c2.metric("Late packages", m["late_packages"])
    c3.metric("Total time", f"{m['total_travel_minutes']:.0f} min")
    c4.metric("Distance", f"{m['total_distance_km']:.1f} km")
    c5.metric("Reward", f"{m['total_reward']:.0f}")


def _short_metrics(m: dict[str, Any]) -> dict[str, Any]:
    return {
        "Delivered": m["delivered"],
        "Late": m["late_packages"],
        "Time (min)": round(m["total_travel_minutes"], 1),
        "Dist (km)": round(m["total_distance_km"], 1),
        "Avg lateness": round(m["avg_lateness"], 1),
        "Reward": round(m["total_reward"], 1),
    }


def _comparison_bars(rows: list[dict[str, Any]]) -> None:
    import pandas as pd

    df = pd.DataFrame(rows).set_index("Policy")
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Reward (higher = better)")
        st.bar_chart(df[["Reward"]])
    with col2:
        st.caption("Late packages (lower = better)")
        st.bar_chart(df[["Late"]])


if __name__ == "__main__":
    main()
