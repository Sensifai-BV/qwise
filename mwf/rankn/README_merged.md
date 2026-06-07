# Silero-VAD + rank-N MWF — single merged ONNX

`signal → ONNX[ Silero-VAD → rank-N MWF ] → clean STFT → wav`

The **whole flow runs inside one ONNX file**. Silero is embedded by rebuilding
its network from the published weights (`silero_vad_16k.safetensors`) as plain
ONNX ops with an explicit LSTM state, wrapped in a `Scan`, and fused with the
rank-N MWF graph. (The *shipped* `silero_vad.onnx` cannot be embedded — it is
nested `If` subgraphs that crash onnxruntime when scanned — so we rebuild the
clean, `If`-free version from the safetensors. It matches the official model:
correlation 0.995, identical speech detection.)

## Build + run

```bash
pip install numpy scipy soundfile onnx onnxruntime spox
python build_merged.py                 # writes rankn_vad_16k.onnx (+ the masks-input models)
python run_merged.py noise_speech.wav --single --save clean.wav
python run_merged.py noise_speech.wav --single --play

# multiple mics (one mono file per mic) -> the spatial MWF actually beamforms:
python run_merged.py mic01.wav mic02.wav mic03.wav --single --save clean.wav
```

### Multi-mic and VAD gating

- **Pass all mics.** rank-N MWF is a *multichannel* beamformer. A single
  `mic01.wav` gives no spatial gain (only VAD gating + OMLSA apply) — feed every
  mic (`mic01.wav mic02.wav ...`, or one multichannel WAV) to get beamforming.
- **VAD output gating** (on by default) silences non-speech regions, so the
  output contains speech only where Silero detects it. The gate is applied
  *after* OMLSA, built from the graph's `vprob` output (threshold + hangover +
  smoothing). Controls:

  ```bash
  python run_merged.py mic01.wav --single --no-gate          # keep continuous output
  python run_merged.py mic01.wav --single --gate-floor 0.05  # -26 dB floor instead of silence
  python run_merged.py mic01.wav --single --gate-hangover 0.3
  ```

Verified end-to-end on a real-speech 3-mic mixture (test.wav + drone-rotor comb):
**SI-SDR 0.9 → 3.7 dB (+2.7)** at 16 kHz; the single file is ~1.25 MB.

## The single merged ONNX: `rankn_vad_16k.onnx`

```
inputs : vad_in [N, 576]   reference-mic chunks (host-framed: 64 ctx + 512)
         state0 [2,1,128]   Silero LSTM init state (zeros)
         k_Xr, k_Xi [M,F,T] multichannel STFT (real / imag)
outputs: k_Yr, k_Yi [F,T]   clean beamformed STFT  ("clean signal matrix")
         k_vprob [T]         per-frame VAD speech probability (drives the gate)
         final_state [2,1,128]   (ignore)
```

Inside the graph: `Scan(Silero per-chunk, carry LSTM state) → speech probs →
speech/noise masks → masked covariances → rank-N MWF (power-iter + CG) →
beamform`. `M`, `F`, `T`, `N` are dynamic.

The only things on the host are the STFT (to make `vad_in` + `Xr/Xi`), the
OMLSA post-filter, and the iSTFT — because ONNX has an `STFT` op but **no
inverse-STFT op**, so the graph naturally ends at the clean STFT matrix. The
host glue is ~10 lines (see below / `merged_ref.enhance_single`).

### Minimal host glue

```python
import numpy as np, soundfile as sf, onnxruntime as ort
import rankn as R, merged_ref as MR

x, fs = sf.read("noise_speech.wav", always_2d=True)        # [L, M], 16 kHz
cfg = R.config_for(fs)
X = R.stft_multi(x, cfg.n_fft, cfg.hop)                     # [M,F,T]
vad_in, _ = MR._vad_chunks(x[:, cfg.ref_mic], fs)          # [N, 576]
sess = ort.InferenceSession("rankn_vad_16k.onnx")
Yr, Yi = sess.run(["k_Yr", "k_Yi"], {
    "vad_in": vad_in.astype("float32"),
    "state0": np.zeros((2,1,128), "float32"),
    "k_Xr": np.real(X).astype("float32"), "k_Xi": np.imag(X).astype("float32")})
Y = R.omlsa(Yr + 1j*Yi)                                     # host OMLSA
y = R.istft(Y, cfg.n_fft, cfg.hop, length=x.shape[0])       # host iSTFT
sf.write("clean.wav", (y/(abs(y).max()+1e-9)*0.9).astype("float32"), fs)
```

## Files

| file | role |
|------|------|
| `rankn_vad_16k.onnx` | **the single merged file** (Silero + rank-N MWF) |
| `silero_onnx.py` | rebuilds Silero from safetensors as ONNX ops + Scan |
| `build_merged.py` | `export_single_file()` fuses Silero + rankn; also exports the masks-input models |
| `rankn_vadmask_{16k,8k}.onnx` | masks-input MWF (for the host-looped-Silero path) |
| `merged_ref.py` | NumPy reference + `enhance_single()` / `enhance_onnx()` |
| `run_merged.py` | CLI: `--single` (one file), `--onnx`, or pure NumPy |
| `rankn.py` | shared DSP + rank-N math + OMLSA |

## 8 kHz

The single-file embedding needs the matching Silero weights. Only
`silero_vad_16k.safetensors` ships in the repo, so `rankn_vad_16k.onnx` is the
16 kHz model. For 8 kHz, either supply `silero_vad_8k.safetensors` (then
`export_single_file(path, 8000, that_file)`), or use the host-looped path:

```bash
python run_merged.py noise_8k.wav --onnx --save clean.wav   # rankn_vadmask_8k.onnx + host silero
```

## Notes

- The embedded VAD matches the official Silero to corr 0.995 / mean prob diff
  0.011 (a few transition frames differ by fp32 LSTM accumulation); speech/noise
  masks are functionally identical.
- Pipeline is **utterance-level** (covariances use the whole signal), not
  streaming.
- Mono input runs (VAD + OMLSA only; no spatial gain).
