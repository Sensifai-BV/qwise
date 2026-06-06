# rank-N MWF — NumPy reference + ONNX export + demo

Python/NumPy port of the Q-WiSE **rank-N multichannel Wiener filter** (the
`rankn` method from `acoustic_simulator/mwf`), an ONNX export for **8 kHz and
16 kHz**, and a runnable demo.

The spatial stage is **eig-free and inverse-free** so it exports cleanly to
ONNX:

```
(phi_s, d)  = dominant eigenpair of Phi_ss   via power iteration (ref-column init)
x           = Phi_nn^{-1} d                   via complex conjugate gradient
w(f)        = phi_s * conj(d_ref) * x / (mu + phi_s * d^H x)
```

`mu = 1`, `power_iters = 0`, N ∈ {2,3} reproduces the original `mwf_2mic` /
`mwf_3mic` exactly. Any number of microphones N ≥ 2 is supported (CG is exact
in ≤ N steps; the exported `cg_iters = 8` is exact up to 8 mics — raise it for
larger arrays and re-export).

## Files

| file | what |
|------|------|
| `rankn.py` | NumPy reference: STFT, energy VAD, covariances, rank-N weights, beamform, OMLSA, iSTFT |
| `build_onnx.py` | builds + exports `rankn_8k.onnx` and `rankn_16k.onnx` (spox) |
| `rankn_8k.onnx`, `rankn_16k.onnx` | the exported spatial-MWF graphs (fp32) |
| `demo.py` | CLI: read a noisy WAV, run the ONNX model, save/play the result |

## Install

```bash
pip install numpy scipy soundfile onnx onnxruntime spox
pip install sounddevice          # only needed for --play
```

## Re-export the ONNX models

```bash
python build_onnx.py             # writes rankn_8k.onnx and rankn_16k.onnx
```

## Demo

```bash
# real recording (multichannel WAV, one channel per mic), 8 k or 16 k
python demo.py noise_speech.wav --save enhanced.wav
python demo.py noise_speech.wav --play

# no file handy? synthesize a 3-mic mixture, then enhance
python demo.py --synth --rate 16000 --save enhanced.wav
python demo.py --synth --rate 8000  --play

# compare the ONNX result against the pure-NumPy reference
python demo.py noise_speech.wav --save out.wav --ref

# pure-NumPy path (no onnxruntime)
python demo.py noise_speech.wav --no-onnx --save out.wav
```

The matching ONNX file is selected automatically from the WAV sample rate.
A mono file works too, but only the OMLSA single-channel stage applies
(spatial filtering needs ≥ 2 mics).

## What runs where

The ONNX graph runs the **rank-N spatial MWF** — the genuinely new math:

```
inputs : Xr, Xi  [M, F, T]   multichannel STFT (real / imag)
         ms, mn  [T]         speech / noise frame masks
outputs: Yr, Yi  [F, T]      beamformed STFT (real / imag),  Y = w^H X
```

`M`, `F`, `T` are dynamic axes. STFT, the energy VAD, the OMLSA post-suppressor
and the iSTFT stay on the host (standard DSP, in `rankn.py`). The per-rate STFT
parameters (`n_fft`, `hop`) are stored in each model's `metadata_props`.

## Notes

- **fp32 vs fp64.** The ONNX path matches the NumPy reference to ~1e-6 on
  well-conditioned data; on highly tonal / rank-deficient covariances the gap
  widens (~1e-2) but SI-SDR is unchanged. Diagonal loading `eps_reg = 1e-5`
  keeps the fp32 CG well-conditioned.
- **OMLSA in ONNX.** The post-suppressor is recursive over time; it is kept on
  the host here. It can be folded into the graph later as a Scan, or as a
  parallel minimum-statistics variant. The only non-ONNX-native op needed is
  the exponential integral `E1` (replace with a polynomial approximation).
- **Exactness.** `python -c "import rankn, numpy as np; ..."` — the weight
  kernel equals the analytic `mwf_2mic`/`mwf_3mic` for N=2,3 and the direct
  SDW-MWF for larger N (verified to ~1e-12 in fp64).
