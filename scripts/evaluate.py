from __future__ import annotations

import argparse

import _bootstrap

from src.agents.evaluate_agent import evaluate
from src.utils.config import load_config
from src.utils.logger import get_logger
from src.visualization.plot_metrics import plot_comparison

logger = get_logger("evaluate")


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate RL vs baselines.")
    ap.add_argument("--config", default="config/jakarta_menteng.yaml")
    ap.add_argument("--model", default=None, help="path to trained model (.zip)")
    ap.add_argument("--algo", default=None, choices=["PPO", "DQN", "ppo", "dqn"])
    ap.add_argument("--episodes", type=int, default=None)
    args = ap.parse_args()

    config = load_config(args.config)
    result = evaluate(
        config,
        model_path=args.model,
        algo=args.algo,
        episodes=args.episodes,
    )
    try:
        plot_comparison(result["summary"])
    except Exception as exc:
        logger.warning("Comparison plot failed: %s", exc)

    print("\n=== Comparison ===\n")
    print(result["table"])


if __name__ == "__main__":
    main()
