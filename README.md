---
title: Courier RL
emoji: 🛵
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Adaptive Courier Routing with Reinforcement Learning on Real-World Maps

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Hugging%20Face%20Spaces-ffcc4d?logo=huggingface&logoColor=black)](https://huggingface.co/spaces/mdaninas1999/Courier-RL)
![CI](https://img.shields.io/badge/CI-pytest-success)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![RL](https://img.shields.io/badge/RL-PPO%20%2F%20DQN-7c3aed)
![map](https://img.shields.io/badge/maps-OpenStreetMap-06b6d4)

> **▶ Live demo:** <https://huggingface.co/spaces/mdaninas1999/Courier-RL> — drag the
> courier & packages on a real Jakarta map, switch routing policies (incl. the
> trained PPO agent), and watch the route animate.

An AI courier that **learns which package to deliver next** on a *real* road
network (OpenStreetMap), under **dynamic traffic** and **delivery deadlines**.
A PPO/DQN agent is trained with [Gymnasium](https://gymnasium.farama.org/) +
[Stable-Baselines3](https://stable-baselines3.readthedocs.io/) and benchmarked
against four classical routing baselines, with route visualisation on the real
map.

> Implements the PRD *“Adaptive Courier Routing with Reinforcement Learning on
> Real-World Maps”* (v1.0). See [`PRD_Adaptive_Courier_Routing_RL_Real_World_Map.md`](PRD_Adaptive_Courier_Routing_RL_Real_World_Map.md).

<p align="center">
  <img src="assets/demo-route.png" width="560" alt="Courier route on the real Menteng street network"><br>
  <sub>A delivery route on the real Menteng (Jakarta) road network — depot (■), packages (● green = on-time, orange = late), route following actual streets.</sub>
</p>

---

## ✨ What it does

- Downloads a drivable road graph from **OpenStreetMap** via `osmnx` (cached locally).
- Places a **depot** and configurable **packages** (with deadlines + priorities) on real nodes.
- Simulates **dynamic traffic**: rush-hour patterns, spatial congestion hotspots, random incidents.
- Exposes a **Gymnasium environment** (`CourierRoutingRealMapEnv-v0`) where the agent
  picks one of the *K nearest* packages; travel uses the dynamic shortest path.
- Trains **PPO / DQN** agents and compares them to **Random, Nearest-Neighbor,
  Earliest-Deadline-First, Greedy-Score** baselines.
- Exports a **comparison table** with **95% confidence intervals** + on-time rate
  (CSV/JSON/Markdown), a **reward curve**, and a **Folium route map** + static plot.
- Ships a **drag-and-drop web demo** (`scripts/serve_demo.py`): pick among 3 Jakarta
  districts (real **administrative boundary** polygons), drag the courier & packages,
  add/remove on click, watch the courier animate along real streets, switch policies
  (incl. the trained PPO agent) — all constrained to the service area while routes may
  cut through adjacent roads.

### Offline / no-internet?
Every script falls back to a **synthetic street grid** if OSM is unavailable, so
the whole pipeline (train → evaluate → demo) runs without internet. Use
`config/synthetic_smoke.yaml` or set `map.force_synthetic: true`.

---

## 📦 Installation

Requires **Python 3.10+** (tested on 3.14).

```bash
pip install -r requirements.txt
```

The heavy dependencies are `torch`, `stable-baselines3`, `osmnx`, `geopandas`,
`shapely`, `folium`. If a geospatial wheel fails to build on your platform, you
can still run everything on the synthetic map (osmnx is only needed for real OSM
downloads).

---

## 🚀 Quickstart

```bash
# 1. Download a real-world map (cached to data/maps/)
python scripts/download_map.py --place "Menteng, Jakarta, Indonesia"

# 2. Train a PPO agent
python scripts/train.py --config config/jakarta_menteng.yaml --algo PPO

# 3. Evaluate RL vs baselines (writes CSV/JSON + comparison table)
python scripts/evaluate.py --config config/jakarta_menteng.yaml \
        --model data/checkpoints/ppo_courier.zip

# 4. Visual demo: run one episode and render the route on the real map
python scripts/run_demo.py --model data/checkpoints/ppo_courier.zip \
        --config config/jakarta_menteng.yaml
```

### Fast offline smoke test (no internet, ~1 min)

```bash
python scripts/train.py    --config config/synthetic_smoke.yaml --algo PPO
python scripts/evaluate.py --config config/synthetic_smoke.yaml \
        --model data/checkpoints/ppo_smoke.zip --algo PPO
python scripts/run_demo.py --config config/synthetic_smoke.yaml --policy greedy_score
```

### 🎮 Interactive drag-and-drop demo (recommended)

A Leaflet web app where you **drag the courier and package pins**, **click the map
to add packages**, **click a pin to remove it**, and pick a routing policy — the
backend snaps every point to the nearest real road node and recomputes the route
live (following actual streets). The trained **PPO Agent** appears automatically
once a compatible `data/checkpoints/ppo_courier.zip` exists.

```bash
python scripts/serve_demo.py --config config/jakarta_menteng.yaml
# open http://localhost:8000
```

Backend uses only the Python standard library (`http.server`) — no Flask needed.

### Parameter dashboard (optional)

A Streamlit app to sweep scenario parameters (seed, package count, start hour,
traffic, deadline tightness) and compare all policies side by side:

```bash
streamlit run src/visualization/dashboard.py -- --config config/jakarta_menteng.yaml
```

---

## 🗺️ How it works

### Map → graph
`osmnx` downloads the drivable network; the largest strongly-connected component
is kept so every node is reachable. Each edge is normalised to:

| attribute | meaning |
|-----------|---------|
| `length` | metres |
| `base_speed_kph` | free-flow speed |
| `base_travel_time` | free-flow travel time (minutes) |
| `traffic_multiplier` | current congestion factor (≥ 1.0) |
| `current_travel_time` | `base_travel_time * traffic_multiplier` |

### RL formulation
- **Action** — `Discrete(K)`: choose one of the `K` nearest remaining packages.
  Travel to it uses `networkx` shortest path weighted by `current_travel_time`.
- **Observation** — length `9 + 3·K` vector, scaled to `[-1, 1]`:
  - *global (9):* courier x/y, elapsed time, remaining count, nearest-package
    time, average traffic, urgent count, min-deadline-remaining, depot distance.
  - *per candidate (3):* travel time, deadline remaining, priority.
- **Reward**

  ```
  reward = base_delivery_reward * priority_multiplier
         - travel_time_penalty
         - distance_penalty
         - lateness_penalty
         + completion_bonus          (when all packages delivered)
         - invalid_action_penalty    (when the chosen slot has no package)
  ```

### Traffic
`traffic_multiplier` combines a **time-of-day** term (morning/evening rush,
midday), **spatial hotspots** (lat/lon + radius with linear falloff), and
**random incidents**, recomputed every `traffic.update_every_minutes`.

### Deadlines (why the comparison is meaningful)
Deadlines are **calibrated to the estimated delivery workload**
(`mean depot leg × num_packages × tour_factor`), not the raw shift length —
otherwise no package is ever late and RL has nothing to beat the baselines on.
Tune via `simulation.deadline_tour_factor / deadline_slack_min / deadline_slack_max`.

---

## 📊 Example results

`scripts/evaluate.py` runs every method over N episodes with **identical seeds**
(paired comparison) and reports means with **95% confidence intervals**, the
**on-time rate**, and marks the best method (★). Illustrative baseline run on the
real **Menteng** map (20 packages, 20 episodes):

| Method            | Delivered | On-time | Total Time | Late Pkgs | Reward (±95% CI) |
|-------------------|-----------|---------|------------|-----------|------------------|
| Random            | 20/20     | 19%     | 137 min    | 16.2      | −1687 ± 270      |
| Earliest Deadline | 20/20     | 12%     | 133 min    | 17.6      | −1085 ± 279      |
| Greedy Score      | 20/20     | 15%     | 95 min     | 17.0      | 835 ± 155        |
| Nearest Neighbor  | 20/20     | 41%     | 85 min     | 11.8      | **1349 ± 158** ★ |
| PPO Agent         | 20/20     | 41%     | 85 min     | 11.8      | **1349 ± 158**   |

<p align="center">
  <img src="assets/comparison.png" width="660" alt="Mean reward (95% CI) and on-time rate per method"><br>
  <sub>Mean reward (±95% CI) and on-time delivery rate over 20 episodes — best method in green. Generated by <code>scripts/evaluate.py</code>.</sub>
</p>

**Reward shaping for a meaningful comparison:** lateness is penalised
(`lateness_penalty_weight: 4.0`, priority-weighted ×1.0/1.3/1.6), so methods that
ignore deadlines and zig-zag across the map (random, earliest-deadline) pile up
lateness and score *negative*. **Nearest-neighbor is near-optimal here** —
minimising travel time also minimises lateness across the board — and PPO (trained
with `ent_coef: 0.01`) *learns to match it* (delivers 20/20, 41% on-time). Beating
it would require trading a little extra travel to save a tight deadline — richer
signals (action masking, deadline-margin reward, longer training), left as future
work. The honest result: **PPO ≈ best baseline**, far above the deadline-blind
heuristics. Train and re-run with `--model`:

```bash
python scripts/train.py    --config config/jakarta_menteng.yaml --algo PPO
python scripts/evaluate.py --config config/jakarta_menteng.yaml --model data/checkpoints/ppo_courier.zip
```

(Exact numbers vary with seed, training budget and area.)

---

## ⚙️ Configuration

Config is layered: built-in defaults → `config/default_training.yaml` →
your area file (e.g. `config/jakarta_menteng.yaml`). Only override what differs.
Key knobs:

```yaml
map:        { place_name, network_type, cache_path, force_synthetic }
simulation: { num_packages, candidate_package_count (K), max_episode_minutes,
              start_hour, deadline_tour_factor }
traffic:    { enable_dynamic_traffic, rush_hour_*, hotspots, max_multiplier }
reward:     { base_delivery_reward, *_multiplier, *_penalty_weight, completion_bonus }
training:   { algorithm, total_timesteps, learning_rate, gamma, save_path }
evaluation: { episodes, compare_with }
```

Use a different city by changing `map.place_name` (e.g.
`"Gubeng, Surabaya, Indonesia"`, `"Bandung Wetan, Bandung, Indonesia"`).

---

## 🗂️ Project structure

```
adaptive-courier-rl/
├── config/                 # YAML configs (default + per-area + synthetic smoke)
├── data/                   # maps cache, checkpoints, logs, results (generated)
├── scripts/                # CLI: download_map / train / evaluate / run_demo / serve_demo
├── src/
│   ├── map/                # osm_loader, graph_utils, node_sampler
│   ├── simulation/         # package_generator, traffic_model, courier_state
│   ├── envs/               # courier_routing_env  (Gymnasium env; supports custom scenarios)
│   ├── agents/             # train_core/ppo/dqn, env_factory, evaluate_agent
│   ├── baselines/          # random, nearest_neighbor, earliest_deadline, greedy_score
│   ├── visualization/      # folium_map, plot_metrics, dashboard,
│   │                       #   interactive_app + static/  (drag-and-drop demo)
│   └── utils/              # config, logger
└── tests/                  # pytest: env contract, map loader, reward, interactive backend
```

---

## 🧪 Tests

```bash
pytest -q
```

Tests run fully offline on the synthetic grid and cover the Gymnasium contract
(incl. SB3 `check_env`), map normalisation, reward decomposition, full episodes
for every baseline, and the interactive demo backend.

---

## 🛠️ Development

```bash
pip install -r requirements-dev.txt   # runtime + pytest + ruff
pytest                                # config in pyproject.toml (pythonpath, addopts)
ruff check src scripts tests          # lint
pip install -e .                      # optional: install as a package
```

- **`pyproject.toml`** — project metadata, pytest config, ruff lint rules.
- **CI** — `.github/workflows/ci.yml` runs the offline test-suite + lint on Python
  3.11/3.12. Heavy deps (torch/sb3/osmnx) are skipped via `pytest.importorskip`,
  so CI is fast and network-free.

---

## ✅ Definition of Done (PRD §23)

1. ✅ Load a real map from OpenStreetMap
2. ✅ Generate depot + packages on real road nodes
3. ✅ Run a full delivery episode
4. ✅ Train a PPO / DQN agent (checkpoint + reward log saved)
5. ✅ Compare RL against ≥ 3 baselines
6. ✅ Visualise the route on the real map (Folium + static)
7. ✅ Export evaluation results to CSV / JSON
8. ✅ README with install / train / evaluate / demo instructions

---

## 🔭 Possible extensions (PRD §22)

Multi-courier, dynamic new orders mid-episode, vehicle capacity + depot reloads,
weather effects, and SUMO/TraCI for microscopic traffic — all left as future work.

---

## 📝 Notes & references

- OpenStreetMap data via **OSMnx**; shortest paths via **NetworkX**.
- Gymnasium-compatible env; PPO/DQN via **Stable-Baselines3** (PyTorch).
- The synthetic-grid fallback makes the project reproducible offline / in CI.
