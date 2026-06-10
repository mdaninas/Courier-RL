from __future__ import annotations

import argparse

import _bootstrap

from src.agents.evaluate_agent import RLPolicy, load_rl_model, run_episode
from src.baselines.registry import BASELINE_NAMES, make_baseline
from src.envs.courier_routing_env import CourierRoutingRealMapEnv
from src.map import osm_loader
from src.utils.config import load_config
from src.utils.logger import get_logger
from src.visualization.folium_map import render_episode_map
from src.visualization.plot_metrics import plot_route_static

logger = get_logger("run_demo")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run one episode and render the route.")
    ap.add_argument("--config", default="config/jakarta_menteng.yaml")
    ap.add_argument("--model", default=None, help="trained model .zip (overrides --policy)")
    ap.add_argument("--algo", default=None, choices=["PPO", "DQN", "ppo", "dqn"])
    ap.add_argument("--policy", default="greedy_score", choices=list(BASELINE_NAMES))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--output", default="data/results/route_map.html")
    args = ap.parse_args()

    config = load_config(args.config)
    seed = args.seed if args.seed is not None else config.get("project", {}).get("seed", 42)

    graph = osm_loader.load_map(config)
    env = CourierRoutingRealMapEnv(config, graph=graph)

    if args.model:
        model, detected = load_rl_model(args.model, args.algo)
        policy = RLPolicy(model, name=f"{detected}_agent")
        label = policy.name
    else:
        policy = make_baseline(args.policy, config=config, seed=seed)
        label = args.policy

    logger.info("Running demo episode with policy '%s' (seed=%s)", label, seed)
    metrics = run_episode(env, policy, int(seed))

    print(f"\n=== Episode metrics ({label}) ===")
    for k, v in metrics.items():
        print(f"  {k:24s}: {v}")

    html = render_episode_map(
        env, env.route_history(), save_path=args.output, title=f"{label}", metrics=metrics
    )
    png = plot_route_static(env, env.route_history(),
                            save_path=args.output.replace(".html", "_static.png"))
    print(f"\nInteractive map : {html}")
    print(f"Static route    : {png}")


if __name__ == "__main__":
    main()
