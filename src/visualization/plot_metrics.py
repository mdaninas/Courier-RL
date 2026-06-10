from __future__ import annotations

from src.utils.config import resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _moving_average(x: list[float], window: int = 20) -> list[float]:
    if len(x) < window or window <= 1:
        return x
    import numpy as np

    return list(np.convolve(x, np.ones(window) / window, mode="valid"))


def plot_reward_curve(
    rewards: list[float],
    timesteps: list[int] | None = None,
    save_path: str = "data/logs/reward_curve.png",
    title: str = "Training reward",
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = resolve_path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if rewards:
        x = timesteps if timesteps and len(timesteps) == len(rewards) else list(range(len(rewards)))
        ax.plot(x, rewards, alpha=0.3, label="episode reward", color="#0275d8")
        ma = _moving_average(rewards, 20)
        if len(ma) != len(rewards):
            ax.plot(x[-len(ma):], ma, color="#d9534f", label="moving avg (20)")
    ax.set_xlabel("timestep" if timesteps else "episode")
    ax.set_ylabel("reward")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info("Saved reward curve -> %s", out)
    return str(out)


def plot_comparison(summary: dict[str, dict[str, float]], save_path: str = "data/results/comparison.png") -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = resolve_path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    methods = list(summary.keys())
    rewards = [summary[m]["total_reward"] for m in methods]
    err = [summary[m].get("total_reward_ci95", 0.0) for m in methods]
    ontime = [summary[m].get("ontime_rate", 0.0) * 100 for m in methods]
    best = max(range(len(methods)), key=lambda i: rewards[i]) if methods else 0
    bar_colors = ["#16a34a" if i == best else "#4f46e5" for i in range(len(methods))]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(methods, rewards, yerr=err, capsize=4, color=bar_colors)
    axes[0].set_title("Mean reward (±95% CI) — green = best")
    axes[0].axhline(0, color="#94a3b8", linewidth=0.8)
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(methods, ontime, color=bar_colors)
    axes[1].set_title("On-time delivery rate (%)")
    axes[1].set_ylim(0, 100)
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info("Saved comparison chart -> %s", out)
    return str(out)


def plot_route_static(env, route_nodes: list[int] | None = None, save_path: str = "data/results/route_static.png") -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = resolve_path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    graph = env.graph

    fig, ax = plt.subplots(figsize=(8, 8))
    for u, v in graph.edges():
        x = [graph.nodes[u]["x"], graph.nodes[v]["x"]]
        y = [graph.nodes[u]["y"], graph.nodes[v]["y"]]
        ax.plot(x, y, color="#dddddd", linewidth=0.5, zorder=1)

    for p in env.packages:
        lon, lat = graph.nodes[p.destination_node]["x"], graph.nodes[p.destination_node]["y"]
        color = "#5cb85c" if (p.delivered and not p.is_late()) else ("#f0ad4e" if p.is_late() else "#999999")
        ax.scatter(lon, lat, c=color, s=40, zorder=3, edgecolors="k", linewidths=0.4)

    ax.scatter(graph.nodes[env.depot_node]["x"], graph.nodes[env.depot_node]["y"],
               c="black", marker="s", s=90, zorder=4, label="depot")

    route_nodes = route_nodes if route_nodes is not None else env.route_history()
    if route_nodes and len(route_nodes) > 1:
        rx = [graph.nodes[n]["x"] for n in route_nodes]
        ry = [graph.nodes[n]["y"] for n in route_nodes]
        ax.plot(rx, ry, color="#0275d8", linewidth=2, zorder=2, label="route")

    ax.set_title("Courier route (static)")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.legend()
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info("Saved static route -> %s", out)
    return str(out)
