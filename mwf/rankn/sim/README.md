# Drone acoustic simulation + live energy test (`sim_drone_test.py`)

End-to-end proof for **`rankn_vad_16k.onnx`**: simulate the drone/human scene,
run only that ONNX, show its power use in **milliwatts in real time**, and save
the clean speech when you stop.

## Scene

- 3-mic linear array under the drone, **10 cm** between mics
- human **2.5 m** from the drone, mouth height 0.88 × 1.70 m
- your speech (laptop mic, or a `--speech` WAV) is propagated to each mic with
  free-field fractional delay + 1/r gain
- two noises mixed the same way: **drone fan** (co-located with the array) and
  **environment** (distant) — from the simulator's `wavs/drone_fan.wav` and
  `wavs/env_ambient.wav`
- noise is scaled to a chosen input SNR at the reference mic (`--snr`)

Only `rankn_vad_16k.onnx` is used for enhancement (Silero-VAD + rank-N MWF in
one graph); the host adds OMLSA + VAD gating + iSTFT.

## Run

```bash
pip install sounddevice            # for the live laptop mic

# live: speak into the laptop mic, Ctrl+C to stop and save
python sim_drone_test.py --save clean.wav

# no mic? feed a speech file (real-time paced; --fast to skip pacing)
python sim_drone_test.py --speech some_speech.wav --save clean.wav --fast --save-noisy
```

While running it prints, per window:

```
t=  14.0s | onnx  5.1ms | RTF 0.003 | MODEL  15.2 mW | speech [#########  ] 56%
```

On Ctrl+C it writes the clean speech and prints a summary (avg/peak mW, total
energy, PASS/FAIL vs a 50 mW budget).

## Options

| flag | meaning | default |
|------|---------|---------|
| `--snr` | input speech-to-noise at the ref mic (dB) | 3 |
| `--fan` / `--env` | relative weights of the two noises | 1.0 / 0.4 |
| `--window` | processing window (s) | 2.0 |
| `--watts` | assumed active power for the mW estimate (no PMIC/RAPL) | 6.0 |
| `--speech` | use a WAV instead of the live mic | — |
| `--fast` | (file mode) don't pace to real-time | off |
| `--save-noisy` | also save the simulated mic-1 for A/B | off |

## What the milliwatt number means

`MODEL mW` is the ONNX's **average power** = (energy of the inference) ÷ (audio
duration). Energy is **measured** where possible (Linux RAPL, or run on a Pi 5
where the PMIC is read) and otherwise **estimated** as `--watts × RTF`. Because
the model is busy only ~0.3 % of the time (RTF ≈ 0.003 here), this lands around
**15 mW — well under 50 mW**. The instantaneous draw during the ~5 ms compute
burst is watts; the mW figure is the amortized real-time average.

## Verified denoising

On the bundled speech at 3 dB input SNR: SI-SDR (vs speech as it reaches mic 1)
**−2.7 → +4.8 dB**, and noise-only regions drop **−14 dB**. `demo_clean.wav`
(output) vs `sim_noisy_mic1.wav` (one simulated array channel) show the before/
after.

## Notes

- The 3 mics are derived from one source per signal, so the speech is perfectly
  coherent across mics (ideal for the rank-1 MWF). Real arrays with independent
  mic self-noise behave a little differently.
- VAD runs on the noisy reference mic; at very low SNR (≪ 0 dB) detection
  degrades — raise `--snr` or speak closer.
- For a *measured* (not estimated) mW figure, run on a Raspberry Pi 5 (PMIC) or
  a Linux box with RAPL; the script picks those up automatically.
