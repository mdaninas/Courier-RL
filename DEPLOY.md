# Deploy — Hugging Face Spaces (Docker)

How the pieces fit:

```
   GitHub (source code)  --[GitHub Action]-->  HF Space (Docker)  -->  public live UI
   you push here              auto-mirror           rebuilds image        visitors open this URL
```

- **Code** lives on **GitHub** (source of truth).
- The **running demo (the Leaflet UI)** is served by the **HF Space** at
  `https://huggingface.co/spaces/<user>/<space>` — that is the link you share.
- A **GitHub Action** mirrors GitHub → HF on every push to `main`, so the Space
  rebuilds automatically. (Optional — you can also push to HF by hand.)

The image bakes in the 3 road graphs, boundary polygons and the trained PPO model
(~12 MB total, tracked in git), so the Space runs fully offline — no OSM download
at startup.

---

## Option A — automatic (GitHub → HF via Action)  ← recommended

1. **Create the Space** at <https://huggingface.co/new-space>
   - SDK: **Docker** · template: **Blank** · name e.g. `courier-rl`.

2. **Create a write token** at <https://huggingface.co/settings/tokens> (role: *Write*).

3. **In the GitHub repo → Settings → Secrets and variables → Actions:**
   - Secret  `HF_TOKEN`    = the write token
   - Variable `HF_USERNAME` = your HF username
   - Variable `HF_SPACE`    = the Space name (e.g. `courier-rl`)

4. **Push to GitHub:**
   ```bash
   git init && git add -A && git commit -m "Courier RL demo"
   git branch -M main
   git remote add origin https://github.com/<you>/<repo>.git
   git push -u origin main
   ```
   The `Deploy to Hugging Face Spaces` workflow runs → mirrors to the Space →
   HF builds the Docker image → demo goes live in a few minutes.

## Option B — manual (push straight to HF)

```bash
git init && git add -A && git commit -m "Courier RL demo"
git remote add space https://huggingface.co/spaces/<user>/<space>
git push --force space main
```
(Enter your HF username + a write token when prompted for credentials.)

---

## Local sanity check (optional)

```bash
docker build -t courier-rl .
docker run --rm -p 7860:7860 courier-rl
# open http://localhost:7860
```

## Notes
- HF Spaces free tier: ~16 GB RAM, 2 vCPU — comfortably fits torch (CPU) + the graphs.
- The container listens on **7860** (set via `app_port: 7860` in the README front-matter).
- If the PPO model ever fails to load (e.g. a torch/sb3 version mismatch), the demo
  degrades gracefully to the baseline policies — it never crashes.
- First request after a cold start loads the model + graphs (a few seconds).
