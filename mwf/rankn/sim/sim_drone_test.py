#!/usr/bin/env python3
"""
sim_drone_test.py — live drone/human acoustic simulation + rankn_vad_16k.onnx test.

Scenario
--------
A drone hovers ~2.5 m from a person (mouth height 0.88 x 1.70 m). Under the drone
is a 3-microphone linear array, 10 cm between mics. Your speech (from the laptop
mic, or a --speech WAV) is the source; it is propagated to each of the 3 mics with
free-field fractional delay + 1/r gain, and two noises are mixed in the same way:

  * drone fan      (co-located with the array — loud, near-equal on all mics)
  * environment    (a distant point source)

The 3-channel noisy array is fed to rankn_vad_16k.onnx (Silero-VAD + rank-N MWF in
one graph); the host adds OMLSA + VAD gating + iSTFT. While running it prints the
model's energy use in milliwatts in real time. Press Ctrl+C to stop — it then
writes the clean speech (everything captured between start and stop) to a WAV.

Usage
-----
    # live laptop mic (needs: pip install sounddevice)
    python sim_drone_test.py --save clean.wav
    # no mic? feed a speech file (real-time paced; add --fast to skip pacing)
    python sim_drone_test.py --speech /path/speech.wav --save clean.wav --fast

Only rankn_vad_16k.onnx is used for enhancement.
"""

import argparse
import glob
import os
import sys
import time
import threading
import queue
import numpy as np
import soundfile as sf
import scipy.signal as ss

HERE = os.path.dirname(os.path.abspath(__file__))
RANKN = os.path.dirname(HERE)                      # .../mwf/rankn
sys.path.insert(0, RANKN)
import rankn as R                                  # noqa: E402
import merged_ref as MR                            # noqa: E402

C_SOUND = 343.0
FS = 16000
SIM_DIR = "/Users/javad/Projects/qwise/simulator/acoustic_simulator"


# ---------------------------------------------------------------------------
#  Geometry: 3-mic array under the drone; human 2.5 m away; two noise sources
# ---------------------------------------------------------------------------
def build_geometry(spacing=0.10, human_dist=2.5, human_h=1.70, drone_h=2.6):
    """Return mic positions [3,3] and source dict {name: (pos, level)}."""
    # mics along x, centered under the drone (array origin)
    mics = np.array([[-spacing, 0, 0], [0.0, 0, 0], [spacing, 0, 0]], float)
    mouth_h = 0.88 * human_h
    # place the mouth `human_dist` (slant) from the array center, in the x-z plane
    dz = mouth_h - drone_h                         # mouth below the array (negative)
    dx = np.sqrt(max(human_dist**2 - dz**2, 0.01))
    speech_pos = np.array([dx, 0.0, dz])
    fan_pos = np.array([0.0, 0.0, 0.12])           # drone fan just above the array
    env_pos = np.array([-6.0, 4.0, -0.5])          # distant ambient source (~7.2 m)
    return mics, {"speech": speech_pos, "fan": fan_pos, "env": env_pos}


def per_mic_delay_gain(mics, pos, dref=1.0):
    """Free-field fractional delay (samples) and 1/r gain for a source at each mic."""
    d = np.linalg.norm(mics - pos[None, :], axis=1)        # [3] distances
    delay = (d - d.min()) / C_SOUND * FS                   # relative TDOA, samples
    gain = dref / np.maximum(d, dref)                      # clamped 1/r
    return delay, gain


def frac_read(buf, idx, loop):
    """Linear-interpolated read of buf at float positions idx. loop=True wraps;
    loop=False returns 0 outside [0, len)."""
    i0 = np.floor(idx).astype(np.int64)
    frac = idx - i0
    L = len(buf)
    if loop:
        a = buf[i0 % L]; b = buf[(i0 + 1) % L]
        return (1 - frac) * a + frac * b
    valid = (i0 >= 0) & (i0 + 1 < L)
    i0c = np.clip(i0, 0, max(L - 2, 0))
    out = (1 - frac) * buf[i0c] + frac * buf[i0c + 1]
    out[~valid] = 0.0
    return out


# ---------------------------------------------------------------------------
#  Noise loading
# ---------------------------------------------------------------------------
def load_noise(path):
    x, fs = sf.read(path)
    if x.ndim > 1:
        x = x.mean(1)
    if fs != FS:
        x = ss.resample(x, int(len(x) * FS / fs))
    x = x.astype(np.float64)
    x /= (np.std(x) + 1e-9)
    return x


# ---------------------------------------------------------------------------
#  Energy measurement (real via RAPL on Linux; estimate otherwise)
# ---------------------------------------------------------------------------
def rapl_uj():
    tot, ok = 0, False
    for f in glob.glob("/sys/class/powercap/intel-rapl:*/energy_uj"):
        try:
            tot += int(open(f).read()); ok = True
        except Exception:
            pass
    return tot if ok else None


# ---------------------------------------------------------------------------
#  Per-window enhancement with rankn_vad_16k.onnx (session reused)
# ---------------------------------------------------------------------------
def enhance_window(x, sess, cfg):
    """x [L,3] -> clean y [L]; returns (y, onnx_seconds, speech_fraction)."""
    X = R.stft_multi(x, cfg.n_fft, cfg.hop)
    vad_in, _ = MR._vad_chunks(x[:, cfg.ref_mic], FS)
    feeds = {"vad_in": vad_in.astype(np.float32),
             "state0": np.zeros((2, 1, 128), np.float32),
             "k_Xr": np.real(X).astype(np.float32),
             "k_Xi": np.imag(X).astype(np.float32)}
    e0 = rapl_uj(); t0 = time.perf_counter()
    Yr, Yi, vprob = sess.run(["k_Yr", "k_Yi", "k_vprob"], feeds)
    t_inf = time.perf_counter() - t0
    e1 = rapl_uj()
    Y = Yr.astype(np.float64) + 1j * Yi.astype(np.float64)
    if cfg.post_omlsa:
        Y = R.omlsa(Y, cfg.omlsa_floor_db, cfg.omlsa_alpha_dd,
                    cfg.omlsa_alpha_s, cfg.omlsa_alpha_d, cfg.omlsa_win_min)
    if cfg.vad_gate:
        g = MR.vad_gate(vprob, cfg.hop, FS, thr=cfg.gate_thr,
                        hangover_s=cfg.gate_hangover_s, floor=cfg.gate_floor)
        Y = Y * g[None, :len(Y[0])]
    y = R.istft(Y, cfg.n_fft, cfg.hop, length=x.shape[0])
    rapl_j = (e1 - e0) / 1e6 if (e0 is not None and e1 is not None and e1 >= e0) else None
    return y, t_inf, float(np.mean(vprob > 0.5)), rapl_j


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Live drone sim + rankn_vad_16k.onnx energy test")
    ap.add_argument("--save", default="clean.wav", help="output clean speech WAV")
    ap.add_argument("--onnx", default=os.path.join(RANKN, "rankn_vad_16k.onnx"))
    ap.add_argument("--speech", help="speech WAV to use instead of the live mic")
    ap.add_argument("--fast", action="store_true", help="(file mode) don't pace to real-time")
    ap.add_argument("--window", type=float, default=2.0, help="processing window (s)")
    ap.add_argument("--snr", type=float, default=3.0,
                    help="input speech-to-noise ratio at the reference mic (dB)")
    ap.add_argument("--fan", type=float, default=1.0, help="drone-fan weight (relative)")
    ap.add_argument("--env", type=float, default=0.4, help="environment weight (relative)")
    ap.add_argument("--speech-gain", type=float, default=1.0)
    ap.add_argument("--speech-rms", type=float, default=0.08,
                    help="assumed speech level for SNR scaling in live-mic mode")
    ap.add_argument("--watts", type=float, default=6.0,
                    help="assumed active power (W) for the mW estimate when RAPL/PMIC absent")
    ap.add_argument("--fan-wav", default=os.path.join(SIM_DIR, "wavs", "drone_fan.wav"))
    ap.add_argument("--env-wav", default=os.path.join(SIM_DIR, "wavs", "env_ambient.wav"))
    ap.add_argument("--save-noisy", action="store_true", help="also save the simulated mic1")
    args = ap.parse_args()

    import onnxruntime as ort
    if not os.path.exists(args.onnx):
        sys.exit(f"model not found: {args.onnx}\nbuild it: python ../build_merged.py")
    so = ort.SessionOptions(); so.log_severity_level = 3
    sess = ort.InferenceSession(args.onnx, so, providers=["CPUExecutionProvider"])

    cfg = R.config_for(FS)
    mics, srcs = build_geometry()
    dg = {k: per_mic_delay_gain(mics, p) for k, p in srcs.items()}
    fan = load_noise(args.fan_wav) * args.fan
    env = load_noise(args.env_wav) * args.env
    WIN = int(args.window * FS)
    maxdelay = int(np.ceil(max(d.max() for d, _ in dg.values()))) + 2
    ref = cfg.ref_mic
    g_sp_ref = dg["speech"][1][ref] * args.speech_gain

    # ---- pre-load speech (file mode) so we can set the SNR ----
    speech_src = None
    if args.speech:
        s, fs0 = sf.read(args.speech)
        if s.ndim > 1:
            s = s.mean(1)
        if fs0 != FS:
            s = ss.resample(s, int(len(s) * FS / fs0))
        speech_src = (s / (np.max(np.abs(s)) + 1e-9)).astype(np.float64)
        act = speech_src[np.abs(speech_src) > 0.05]            # active samples
        sp_rms = np.sqrt(np.mean(act ** 2)) if len(act) else np.std(speech_src)
    else:
        sp_rms = args.speech_rms                               # live-mic assumption

    # ---- scale noise to the requested input SNR at the reference mic ----
    probe = min(len(fan), len(env), FS * 5)
    n_ref = (fan[:probe] * dg["fan"][1][ref] + env[:probe] * dg["env"][1][ref])
    n_rms = np.sqrt(np.mean(n_ref ** 2)) + 1e-9
    target = sp_rms * g_sp_ref / (10 ** (args.snr / 20))      # desired noise ref RMS
    noise_scale = target / n_rms
    fan *= noise_scale
    env *= noise_scale

    print("=" * 64)
    print(" Drone acoustic sim  ·  3-mic array (10 cm)  ·  human @ 2.5 m")
    print(f" model: {os.path.basename(args.onnx)}   window: {args.window:.1f}s")
    print(f" input SNR: {args.snr:+.0f} dB  (fan:env = {args.fan}:{args.env})   power ref: "
          f"{'RAPL (measured)' if rapl_uj() else f'est @ {args.watts} W'}")
    print(" speak now — Ctrl+C to stop and save clean speech")
    print("=" * 64)

    cap = []                      # captured mono source samples
    clean_out = []               # produced clean speech (concatenated windows)
    noisy_mic1 = []              # simulated mic-1 (for optional comparison)
    npos = 0                     # global sample cursor (processed source samples)
    stats = {"wins": 0, "rtf": [], "mw": [], "energy_j": 0.0}
    stop = threading.Event()

    def build_array(start, n):
        """Simulated 3-mic noisy block for source samples [start, start+n)."""
        idx = np.arange(start, start + n, dtype=np.float64)
        capbuf = np.asarray(cap, dtype=np.float64)
        out = np.zeros((n, 3))
        for m in range(3):
            s = frac_read(capbuf, idx - dg["speech"][0][m], loop=False) * dg["speech"][1][m] * args.speech_gain
            f = frac_read(fan, idx - dg["fan"][0][m], loop=True) * dg["fan"][1][m]
            e = frac_read(env, idx - dg["env"][0][m], loop=True) * dg["env"][1][m]
            out[:, m] = s + f + e
        return out

    def process_ready():
        nonlocal npos
        while len(cap) - npos >= WIN:
            block = build_array(npos, WIN)
            y, t_inf, sp_frac, rapl_j = enhance_window(block, sess, cfg)
            npos += WIN
            energy_j = rapl_j if rapl_j is not None else args.watts * t_inf
            mw = energy_j / args.window * 1000.0          # avg power per s of audio
            rtf = t_inf / args.window
            clean_out.append(y)
            if args.save_noisy:
                noisy_mic1.append(block[:, 0])
            stats["wins"] += 1; stats["rtf"].append(rtf); stats["mw"].append(mw)
            stats["energy_j"] += energy_j
            bar = "#" * int(min(sp_frac, 1) * 20)
            sys.stdout.write(
                f"\r  t={npos/FS:6.1f}s | onnx {t_inf*1000:5.1f}ms | RTF {rtf:5.3f} "
                f"| MODEL {mw:6.1f} mW | speech [{bar:<20}] {sp_frac*100:3.0f}%   ")
            sys.stdout.flush()

    # ---- source: live mic OR speech file ----
    if args.speech:
        s = speech_src
        blk = WIN
        try:
            for i in range(0, len(s), blk):
                if stop.is_set():
                    break
                cap.extend(s[i:i + blk].tolist())
                process_ready()
                if not args.fast:
                    time.sleep(blk / FS)
        except KeyboardInterrupt:
            pass
    else:
        try:
            import sounddevice as sd
        except Exception:
            sys.exit("sounddevice not installed (pip install sounddevice), "
                     "or use --speech <file.wav>")
        q = queue.Queue()

        def cb(indata, frames, t, status):
            q.put(indata[:, 0].copy())
        try:
            with sd.InputStream(samplerate=FS, channels=1, blocksize=WIN // 2, callback=cb):
                while True:
                    try:
                        cap.extend(q.get(timeout=0.5).tolist())
                        process_ready()
                    except queue.Empty:
                        pass
        except KeyboardInterrupt:
            pass

    # ---- save on stop ----
    print("\n" + "=" * 64)
    if clean_out:
        y = np.concatenate(clean_out)
        pk = np.max(np.abs(y))
        if pk > 0:
            y = y / pk * 0.9
        sf.write(args.save, y.astype(np.float32), FS)
        print(f" saved clean speech : {args.save}  ({len(y)/FS:.1f}s)")
        if args.save_noisy and noisy_mic1:
            nm = np.concatenate(noisy_mic1)
            sf.write("sim_noisy_mic1.wav", (nm / (np.max(np.abs(nm)) + 1e-9) * 0.9).astype(np.float32), FS)
            print(" saved simulated mic1: sim_noisy_mic1.wav")
    else:
        print(" no audio processed.")
    if stats["wins"]:
        print(f" windows processed  : {stats['wins']}")
        print(f" avg RTF            : {np.mean(stats['rtf']):.4f}  "
              f"({1/np.mean(stats['rtf']):.0f}x real-time)")
        print(f" avg MODEL power    : {np.mean(stats['mw']):.1f} mW   "
              f"(peak {np.max(stats['mw']):.1f} mW)")
        print(f" total onnx energy  : {stats['energy_j']*1000:.1f} mJ over the session")
        tag = "measured (RAPL)" if rapl_uj() else f"estimated @ {args.watts} W active"
        print(f" energy basis       : {tag}")
        verdict = "PASS" if np.mean(stats["mw"]) < 50 else "over"
        print(f" <50 mW budget      : {verdict} (avg {np.mean(stats['mw']):.1f} mW)")
    print("=" * 64)


if __name__ == "__main__":
    main()
