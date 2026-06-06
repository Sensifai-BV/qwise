#!/usr/bin/env python3
"""
demo.py — run the rank-N MWF ONNX enhancer on a noisy speech WAV.

The ONNX file (rankn_8k.onnx / rankn_16k.onnx) runs the rank-N spatial MWF
(covariance -> power-iteration rank-1 -> conjugate-gradient -> beamform). The
host (this script) does STFT, a simple energy VAD, the OMLSA post-suppressor
and the iSTFT. The matching file is selected automatically from the WAV's
sample rate.

The input WAV should be MULTICHANNEL (one channel per microphone) for spatial
denoising. A mono file still works but only the OMLSA single-channel stage
applies (no spatial gain).

Examples
--------
    # process a real recording and save the result
    python demo.py noise_speech.wav --save enhanced.wav

    # process and play it
    python demo.py noise_speech.wav --play

    # no file? synthesize a 16 kHz (or 8 kHz) 3-mic mixture, then process
    python demo.py --synth --rate 16000 --save enhanced.wav
    python demo.py --synth --rate 8000  --play

    # compare ONNX vs the pure-NumPy reference
    python demo.py noise_speech.wav --save out.wav --ref
"""

import argparse
import os
import sys
import numpy as np
import soundfile as sf

import rankn as R


def _onnx_session(onnx_dir, fs):
    import onnxruntime as ort
    path = os.path.join(onnx_dir, f"rankn_{fs // 1000}k.onnx")
    if not os.path.exists(path):
        sys.exit(f"ONNX model not found: {path}\nRun:  python build_onnx.py")
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"]), path


def synth_mixture(fs, seconds=4.0, n_mic=3, snr_db=0.0, seed=0):
    """Multichannel speech + drone-rotor-comb noise. Returns (mix [L,M], clean_ref [L])."""
    rng = np.random.default_rng(seed)
    L = int(seconds * fs)
    t = np.arange(L) / fs
    # voiced speech: harmonic stack (f0 = 140 Hz) gated into a few utterances
    env = np.zeros(L)
    for a, b in [(0.4, 1.2), (1.7, 2.5), (3.0, 3.7)]:
        env[int(a * fs):int(b * fs)] = 1.0
    k = max(1, int(0.025 * fs))
    env = np.convolve(env, np.ones(k) / k, mode="same")
    speech = sum((1.0 / h) * np.sin(2 * np.pi * 140 * h * t)
                 for h in range(1, min(40, fs // (2 * 140))))
    speech *= env
    speech /= np.std(speech) + 1e-9
    # rotor comb (133 Hz fundamental + harmonics) + a little broadband
    rotor = sum(np.sin(2 * np.pi * 133 * h * t + rng.uniform(0, 6)) for h in range(1, 8))
    rotor += 0.3 * rng.standard_normal(L)
    rotor /= np.std(rotor) + 1e-9
    # place sources with per-mic fractional delays (linear array)
    ds = np.linspace(0.0, 2.0, n_mic)       # speech delays (samples)
    dn = np.linspace(0.0, 1.0, n_mic) + 0.6  # noise delays (samples)
    g = 10 ** (-snr_db / 20)                 # noise gain for target SNR
    mix = np.zeros((L, n_mic))
    for m in range(n_mic):
        s = np.interp(t - ds[m] / fs, t, speech)
        n = np.interp(t - dn[m] / fs, t, rotor)
        mix[:, m] = s + g * n
    ref = np.interp(t - ds[0] / fs, t, speech)
    mix /= np.max(np.abs(mix)) + 1e-9
    return mix.astype(np.float64), ref.astype(np.float64)


def si_sdr(est, ref):
    m = min(len(est), len(ref))
    est = est[:m] - est[:m].mean()
    ref = ref[:m] - ref[:m].mean()
    a = (est @ ref) / (ref @ ref + 1e-12)
    tgt = a * ref
    return 10 * np.log10((tgt @ tgt) / (((est - tgt) @ (est - tgt)) + 1e-12) + 1e-12)


def play(y, fs):
    try:
        import sounddevice as sd
    except Exception:
        print("[demo] 'sounddevice' not installed — cannot --play.\n"
              "       pip install sounddevice   (or use --save out.wav)")
        return
    sd.play(y, fs)
    sd.wait()


def main():
    ap = argparse.ArgumentParser(description="rank-N MWF ONNX speech enhancer demo")
    ap.add_argument("input", nargs="?", help="noisy multichannel WAV (8 k or 16 k)")
    ap.add_argument("--save", metavar="OUT.wav", help="write enhanced WAV")
    ap.add_argument("--play", action="store_true", help="play enhanced audio")
    ap.add_argument("--synth", action="store_true", help="synthesize a test mixture")
    ap.add_argument("--rate", type=int, default=16000, choices=[8000, 16000],
                    help="sample rate for --synth (default 16000)")
    ap.add_argument("--mics", type=int, default=3, help="mics for --synth (default 3)")
    ap.add_argument("--snr", type=float, default=0.0, help="input SNR dB for --synth")
    ap.add_argument("--floor-db", type=float, default=-30.0, help="OMLSA spectral floor dB")
    ap.add_argument("--no-omlsa", action="store_true", help="disable OMLSA post-suppressor")
    ap.add_argument("--no-onnx", action="store_true", help="use the pure-NumPy reference")
    ap.add_argument("--ref", action="store_true", help="also run NumPy reference and compare")
    ap.add_argument("--onnx-dir", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    clean = None
    if args.synth:
        fs = args.rate
        x, clean = synth_mixture(fs, n_mic=args.mics, snr_db=args.snr)
        if args.input is None:
            args.input = f"synth_{fs // 1000}k_{args.mics}mic.wav"
        sf.write(args.input, x.astype(np.float32), fs)
        print(f"[demo] synthesized {args.input}  ({x.shape[1]} ch, {fs} Hz, "
              f"{x.shape[0] / fs:.1f}s, SNR {args.snr:.0f} dB)")
    else:
        if not args.input:
            ap.error("provide an input WAV, or use --synth")
        x, fs = sf.read(args.input, always_2d=True)
        x = x.astype(np.float64)
        print(f"[demo] read {args.input}  ({x.shape[1]} ch, {fs} Hz, {x.shape[0] / fs:.1f}s)")

    if fs not in (8000, 16000):
        sys.exit(f"Unsupported sample rate {fs}. Resample the file to 8000 or 16000 Hz.")

    cfg = R.config_for(fs)
    cfg.post_omlsa = not args.no_omlsa
    cfg.omlsa_floor_db = args.floor_db

    if args.no_onnx:
        print("[demo] enhancing with the pure-NumPy reference")
        y = R.enhance(x, cfg)
    else:
        sess, path = _onnx_session(args.onnx_dir, fs)
        print(f"[demo] enhancing with ONNX: {os.path.basename(path)} "
              f"(OMLSA {'on' if cfg.post_omlsa else 'off'})")
        y = R.enhance_with_onnx(x, cfg, sess)

    if args.ref and not args.no_onnx:
        yref = R.enhance(x, cfg)
        m = min(len(y), len(yref))
        rel = np.max(np.abs(y[:m] - yref[:m])) / (np.max(np.abs(yref[:m])) + 1e-9)
        print(f"[demo] ONNX vs NumPy reference: rel max diff = {rel:.2e}")

    if clean is not None:
        print(f"[demo] SI-SDR  input(mic0)={si_sdr(x[:, 0], clean):5.1f} dB   "
              f"enhanced={si_sdr(y, clean):5.1f} dB   "
              f"gain={si_sdr(y, clean) - si_sdr(x[:, 0], clean):+.1f} dB")

    if args.save:
        sf.write(args.save, y.astype(np.float32), fs)
        print(f"[demo] wrote {args.save}")
    if args.play:
        play(y.astype(np.float32), fs)
    if not args.save and not args.play:
        out = "enhanced.wav"
        sf.write(out, y.astype(np.float32), fs)
        print(f"[demo] (no --save/--play given) wrote {out}")


if __name__ == "__main__":
    main()
