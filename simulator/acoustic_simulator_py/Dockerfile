# syntax=docker/dockerfile:1.6
#
# Q-WiSE acoustic simulator — HuggingFace Space-compatible image.
# Single-stage build is fine: the dependency footprint is small and Space
# rebuilds are infrequent. The image exposes port 7860 (the default
# HF Space port) and runs as UID 1000 to match the HF Space sandbox.

FROM python:3.10-slim

# libsndfile is needed by `soundfile`, libgomp by `onnxruntime`,
# ffmpeg lets soundfile fall back through libav for mp3/m4a uploads.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libsndfile1 \
      libgomp1 \
      ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# HuggingFace Spaces run containers as a non-root user. Match that
# locally so the /data mount stays writable across both environments.
ARG USERNAME=qwise
ARG USER_UID=1000
RUN useradd -m -u ${USER_UID} ${USERNAME}
WORKDIR /app

# Install Python deps first so app-only edits don't bust the layer cache.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r /app/requirements.txt

# App code. .dockerignore filters out recordings/, tests/, .git/, etc.
COPY --chown=${USERNAME}:${USERNAME} backend     /app/backend
COPY --chown=${USERNAME}:${USERNAME} frontend    /app/frontend
COPY --chown=${USERNAME}:${USERNAME} models      /app/models
COPY --chown=${USERNAME}:${USERNAME} wavs        /app/wavs
COPY --chown=${USERNAME}:${USERNAME} pyproject.toml /app/pyproject.toml

# Writable run-time directory for session recordings + uploaded WAVs.
# HF Spaces auto-provides /data as ephemeral; the cleanup task drops
# stale folders every 5 minutes so the volume stays bounded.
RUN mkdir -p /data/recordings /data/uploads \
 && chown -R ${USERNAME}:${USERNAME} /data
ENV QWISE_DATA_DIR=/data

USER ${USERNAME}

EXPOSE 7860
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=7860

# Liveness probe — Python-only so we don't add curl/wget to the image.
# HF Spaces polls the app_port itself, but this keeps `docker run` and
# `docker-compose` honest in local + CI runs.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; \
sys.exit(0) if urllib.request.urlopen('http://127.0.0.1:7860/healthz', \
timeout=3).status == 200 else sys.exit(1)" || exit 1

CMD ["uvicorn", "backend.api.app:app", "--host", "0.0.0.0", "--port", "7860"]
