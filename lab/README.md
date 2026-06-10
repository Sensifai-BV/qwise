# Q-WiSE Acoustic Simulator

A MATLAB simulator for the [Q-WiSE](https://sensifai.com/en/portfolio/qwise)
ultra-low-power speech-enhancement ONNX. It places a talker and a noise-emitting
drone in a physical acoustic scene, mixes clean speech with drone-fan and
environment noise across a virtual microphone array, and denoises the resulting
noisy array with `onnx/qwise.onnx` — so you can listen to, watch, and record the
before/after entirely from a desktop, without any hardware.

The enhancement model itself is documented in the [project README](../README.md);
this folder is only the simulation and visualization front-end around it.

## What it does

1. Loads a clean-speech sample and a scene preset (talker height, drone distance
   and position, environment-noise distance).
2. Builds an *N*-microphone array centered on the drone and mixes speech + drone
   fan + environment noise into each channel with per-microphone time-of-arrival
   delays and 1/r distance gains (`SourceMixer`).
3. Plays the noisy mix with live per-mic waveforms and a live spectrogram.
4. Feeds the noisy array to `qwise.onnx` (`mic[M, L] -> clean[L]`), plays the
   enhanced speech, and shows the noisy-vs-clean waveform overlay and a live
   clean spectrogram.
5. Auto-records each run to `recordings/` as `mic01.wav … micNN.wav` + `clean.wav`,
   playable from the *Latest recording* panel.

## Requirements

- MATLAB (R2022a or newer recommended; uses `clim`, `uibuttongroup`, `audioplayer`).
- Python 3 with `onnxruntime`, `numpy`, `soundfile`. The setup script creates a
  local virtual environment at `lab/.pyenv` so nothing is installed system-wide.

MATLAB runs the ONNX through `onnxruntime` via its Python bridge: the
`qwise_ort_helper.py` module owns the ORT session and all numpy handling, and
`run_simulation.m` points MATLAB's `pyenv` at `lab/.pyenv` automatically.

## Setup and run

From this folder:

```bash
./run_qwise.sh
```

This checks for `python3`, creates the `lab/.pyenv` venv, installs the Python
dependencies, then launches `run_simulation.m` via `matlab -batch`.

Options:

```bash
# Different MATLAB install
MATLAB_BIN=/Applications/MATLAB_R2025a.app/bin/matlab ./run_qwise.sh

# Skip venv/deps after the first setup, just launch MATLAB
SKIP_SETUP=1 ./run_qwise.sh

# Launch a different script instead of run_simulation
./run_qwise.sh some_other_script
```

To run from inside MATLAB instead, make sure `pyenv` points at
`lab/.pyenv/bin/python3` in a fresh session, then call `run_simulation`. (Once
Python is loaded in a session, `pyenv` can no longer be switched — restart
MATLAB.)

## Using the UI

| Control | What it does |
|---|---|
| **Speech sample** | Pick a `.wav` from `samples/speech/`; **Listen** plays the raw clean sample. |
| **Scene preset** | Radio buttons: drone 1 m center / 1 m left / 1 m right / 3 m center (talker 170 cm, env 8 m). Changing it moves the drone in the 3-D scene and re-mixes. |
| **Microphone array** | Number of mics (2–5) and inter-mic spacing (10 / 20 / 30 cm). Rebuilds the array, the per-mic waveform rows, and the 3-D scene. The ONNX accepts any mic count (`mic[M, L]`). |
| **Noise mix** | Drone-fan and environment gain sliders. The defaults sit at a recoverable SNR; the cap keeps you in a range the model can still clean. |
| **Play Mixed** | Plays the noisy mix with live per-mic waveforms + noisy spectrogram. |
| **Clean (ONNX)** | Runs `qwise.onnx`, plays the enhanced speech, draws the noisy-vs-clean overlay + live clean spectrogram, and records the run. Press again to restart playback. |
| **Latest recording** | Plays back the captured `mic1 … micN` and `clean` WAVs. |

## Configuration

Everything is driven from [`config/default.m`](config/default.m): sample rate,
microphone count and spacing, scene geometry, noise mix levels, the slider cap,
the preset table, and the ONNX model path. Edit that file to reshape the
simulator; the presets are a plain struct array you can extend.

## Layout

```
lab/
├── run_qwise.sh          # venv setup + MATLAB launcher
├── run_simulation.m      # entry point (paths, pyenv, wiring, UI)
├── qwise_ort_helper.py   # onnxruntime bridge (session registry + enhance)
├── config/
│   └── default.m         # all runtime parameters + scene presets
├── core/
│   ├── OnnxEnhancer.m    # MATLAB wrapper around qwise_ort_helper
│   ├── SourceMixer.m     # physical multi-source -> N-mic array mixer
│   ├── build_geometry.m  # scene geometry, per-source TDOA + 1/r gains
│   ├── AudioIO.m         # sample/noise loading + recording
│   ├── load_wav_loop.m   # looped noise buffer loader
│   ├── loop_chunk.m      # circular-buffer reader
│   └── print_geometry.m  # geometry console dump
├── visualization/
│   ├── SimulatorUI.m     # control panel, waveforms, spectrograms, playback
│   ├── draw_scene.m      # 3-D acoustic scene (centered on talker + drone)
│   └── draw_drone.m      # quad-rotor body + mic markers
├── samples/speech/       # clean-speech samples (dropdown source)
├── wavs/                 # drone_fan.wav, env_ambient.wav noise loops
└── recordings/           # auto-saved mic + clean captures
```
