# Hugging Face Spaces — Docker SDK. Serves the interactive courier demo on :7860.
FROM python:3.11-slim

# Non-root user (Hugging Face Spaces best practice)
RUN useradd -m -u 1000 appuser

WORKDIR /app

# CPU-only PyTorch first (small wheel ~200 MB instead of the CUDA build ~2 GB)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Lean runtime dependencies
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# App code + baked assets (road graphs, boundary polygons, trained PPO model)
COPY --chown=appuser:appuser . .

USER appuser
ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/tmp/hf \
    MPLCONFIGDIR=/tmp/mpl

EXPOSE 7860

CMD ["python", "scripts/serve_demo.py", \
     "--config", "config/jakarta_menteng.yaml", \
     "--host", "0.0.0.0", "--port", "7860", "--no-browser"]
