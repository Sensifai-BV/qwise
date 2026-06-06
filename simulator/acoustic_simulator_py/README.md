---
title: Q-WiSE Acoustic Simulator
emoji: 🎙️
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Neural-guided Multi-channel Wiener Filter demo (Q-WiSE)
---

# Q-WiSE Acoustic Simulator

Python port of the MATLAB Q-WiSE simulator (`/Users/javad/Projects/qwise/simulator/acoustic_simulator`), exposed as a FastAPI + WebSocket web demo and packaged as a HuggingFace Docker Space.

The pipeline matches the MATLAB reference exactly:

1. **SourceMixer** — N microphones, every mic receives speech + drone + env with fractional-sample TDOA and 1/r spreading gain clamped at `cfg.distance_ref` (`backend.core.source_mixer`).
2. **VAD** — either an energy + spectral-flatness baseline or the Q-WiSE ONNX classifier (`backend.vad`).
3. **MWF** — neural-guided Multi-channel Wiener Filter in the STFT domain (GEV / SDW-MWF / MVDR), with optional single-channel Wiener post-filter (`backend.mwf`).

The frontend is a single HTML page with a collapsible right-hand sidebar bound to every field in `cfg`, plus a center column of live waveforms, spectrograms and the 3-D acoustic scene.

## Run locally (no Docker)

```bash
pip install -r requirements.txt
uvicorn backend.api.app:app --reload --port 7860
```

Then open <http://localhost:7860/>. Note: browser `getUserMedia` requires HTTPS or `localhost` — load the page via `http://localhost:7860` (not the LAN IP) or terminate TLS in front.

## Run via Docker (mirrors the HF Space build)

```bash
docker build -t qwise-sim .
docker run --rm -p 7860:7860 -v "$PWD/data:/data" qwise-sim
```

The mounted `./data` directory holds recording sessions + speech-WAV uploads. Drop it and the next container starts with a clean slate; HF Spaces does the same automatically on every restart.

## Deploy as a HuggingFace Docker Space

The YAML front-matter at the top of this file is the entire HF Space config. To publish:

1. Create an empty Space at <https://huggingface.co/spaces/new> with **SDK = Docker** and **app port = 7860**.
2. Either use the HF web UI to upload this directory, or push via Git:

   ```bash
   git remote add hf https://huggingface.co/spaces/<your-user>/qwise-acoustic-simulator
   git push hf main
   ```
3. HF Spaces reads the YAML at the top of this file, builds the `Dockerfile`, and exposes port 7860 over HTTPS. The image takes a couple of minutes to build (slim Python + `onnxruntime`).

### Runtime knobs

The container honours three environment variables (`Settings → Variables and secrets` in the HF UI):

| Variable | Default | Effect |
|---|---|---|
| `QWISE_RATE_LIMIT_MAX`        | `10`      | Max recording-start events per IP per window. |
| `QWISE_RATE_LIMIT_WINDOW_SEC` | `10800`   | Window in seconds (3 h). Also the recording auto-purge TTL. |
| `QWISE_CLEANUP_INTERVAL_SEC`  | `300`     | How often the cleanup task scans for stale folders. |
| `QWISE_DATA_DIR`              | `/data`   | Where recordings + uploads land. Don't override on HF. |
| `PORT`                        | `7860`    | Listen port (HF expects 7860). |

### What HF Spaces actually does

* Builds the `Dockerfile` (no buildx tricks, no multi-stage shenanigans).
* Starts the container as **UID 1000**, matching the `qwise` user defined in the Dockerfile.
* Proxies HTTPS → port 7860, including the WebSocket on `/ws/stream`.
* Provides an ephemeral `/data` volume that is **wiped on every restart**; the cleanup task keeps it bounded between restarts.

### Verifying the image before pushing

The repo ships a Python-only build-config validator that mimics what HF Spaces checks at upload time (Dockerfile sanity, README front-matter, bundled assets present):

```bash
python scripts/verify_hf_config.py
```

It exits non-zero if any required field is missing.

## Layout

```
backend/
  config/        # default.py — single source of truth for runtime parameters
  core/          # geometry, source mixer, audio loops (ported from MATLAB core/)
  vad/           # energy + Q-WiSE ONNX dispatcher (ported from MATLAB vad/)
  mwf/           # streaming + batch Wiener filter (ported from MATLAB mwf/)
  audio/         # AudioIO equivalent — looped sources + session recording
  api/           # FastAPI app + WebSocket stream + REST endpoints
frontend/        # static HTML, JS and CSS for the demo UI
models/          # qwise_vad.onnx (copied from the MATLAB project)
wavs/            # drone_fan.wav and env_ambient.wav loops
tests/           # pytest suite, mirrors tests/ in the MATLAB project
scripts/         # build-config validator + small ops helpers
```
