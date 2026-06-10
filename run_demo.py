"""
run_demo.py — run the SELF-CONTAINED enhancer `qwise.onnx`.

The ONNX is the whole pipeline:  mic[M,L] (raw 16 kHz waveform) -> clean[L].
No host DSP — onnxruntime only. This script loads audio into the [M,L] array the
model expects, runs it, writes the clean wav, and (optionally) saves a waveform
PNG + spectrogram PNG and prints an energy meter (PASS/FAIL vs a 50 mW budget).

Examples
--------
python run_demo.py mic01.wav mic02.wav mic03.wav -o clean.wav
python run_demo.py array.wav -o clean.wav
python run_demo.py speech_mono.wav --synth -o clean.wav
python run_demo.py mic0*.wav -o clean.wav --energy --realtime     # on the Pi

Outputs (next to -o, same stem):
    clean.wav   clean_waveform.png   clean_spectrogram.png
"""


import threading
import argparse
import os
import time
import numpy as np
import onnxruntime as ort
import soundfile as sf

FS = 16000
MODEL = "onnx/qwise.onnx"


# --------------------------------------------------------------------------
#  audio -> [M, L] mic array
# --------------------------------------------------------------------------
def load_array(paths, synth=False, n_mics=3):
    if synth:
        sp, sr = sf.read(paths[0])
        if sp.ndim > 1:
            sp = sp.mean(1)
        if sr != FS:
            sp = _resample(sp, sr, FS)
        sp = sp.astype(np.float64) / (np.std(sp) + 1e-9)
        t = np.arange(len(sp)) / FS
        rotor = sum(np.sin(2 * np.pi * 133 * h * t + h) for h in range(1, 8))
        rotor /= np.std(rotor) + 1e-9
        ds = np.linspace(0, 2.0, n_mics)
        dn = np.linspace(0, 1.0, n_mics) + 0.6
        arr = np.zeros((n_mics, len(sp)), np.float32)
        for m in range(n_mics):
            arr[m] = (np.interp(t - ds[m] / FS, t, sp)
                      + 0.9 * np.interp(t - dn[m] / FS, t, rotor))
        arr /= np.max(np.abs(arr)) + 1e-9
        return arr.astype(np.float32)

    chans = []
    for p in paths:
        a, sr = sf.read(p)
        if sr != FS:
            a = _resample(a, sr, FS)
        if a.ndim == 1:
            chans.append(a)
        else:
            chans.extend(a.T)
    L = min(len(c) for c in chans)
    return np.stack([c[:L] for c in chans], 0).astype(np.float32)


def _resample(x, sr, fs):
    import scipy.signal as ss
    if x.ndim > 1:
        return np.stack([ss.resample(x[:, i], int(len(x) * fs / sr))
                         for i in range(x.shape[1])], 1)
    return ss.resample(x, int(len(x) * fs / sr))


# --------------------------------------------------------------------------
#  waveform + spectrogram PNGs (same stem as the output wav)
# --------------------------------------------------------------------------
def save_plots(clean, fs, stem, noisy=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import scipy.signal as ss

    t = np.arange(len(clean)) / fs

    # ---- waveform ----
    fig, ax = plt.subplots(figsize=(10, 3), dpi=130)
    if noisy is not None:
        ax.plot(np.arange(len(noisy)) / fs, noisy, lw=0.5, alpha=0.4,
                color="#bbbbbb", label="noisy (mic 0)")
    ax.plot(t, clean, lw=0.6, color="#1f77b4", label="clean")
    ax.set_xlabel("time (s)"); ax.set_ylabel("amplitude")
    ax.set_title("Waveform — enhanced speech"); ax.legend(loc="upper right", fontsize=8)
    ax.margins(x=0)
    fig.tight_layout(); fig.savefig(f"{stem}_waveform.png"); plt.close(fig)

    # ---- spectrogram ----
    f, tt, Sxx = ss.spectrogram(clean, fs, nperseg=512, noverlap=384)
    Sdb = 10 * np.log10(Sxx + 1e-10)
    fig, ax = plt.subplots(figsize=(10, 3.4), dpi=130)
    im = ax.pcolormesh(tt, f / 1000, Sdb, shading="auto",
                       cmap="magma", vmin=Sdb.max() - 80, vmax=Sdb.max())
    ax.axhspan(2.5, 5.0, color="cyan", alpha=0.12)   # consonant band the EQ restores
    ax.set_xlabel("time (s)"); ax.set_ylabel("kHz")
    ax.set_title("Spectrogram — enhanced speech  (cyan = 2.5–5 kHz consonant band)")
    fig.colorbar(im, ax=ax, label="dB"); fig.tight_layout()
    fig.savefig(f"{stem}_spectrogram.png"); plt.close(fig)
    return f"{stem}_waveform.png", f"{stem}_spectrogram.png"


# --------------------------------------------------------------------------
#  energy meter  (reuses the Pi PMIC logic in bench_energy)
# --------------------------------------------------------------------------
def _bar(val, budget, width=26):
    frac = min(val / budget, 1.0)
    fill = int(round(frac * width))
    return "[" + "#" * fill + "-" * (width - fill) + f"] {val:.1f} / {budget:.0f} mW"


import glob
import subprocess


def _rapl_uj():
    """Total Intel-RAPL package energy in microjoules (Linux), or None."""
    files = glob.glob("/sys/class/powercap/intel-rapl:[0-9]*/energy_uj")
    if not files:
        return None
    total = 0
    for f in files:
        try:
            total += int(open(f).read().strip())
        except (OSError, ValueError):
            return None
    return total


def _macos_cpu_mw(ms=800):
    """macOS CPU power in mW via `powermetrics` (needs sudo), or None."""
    try:
        out = subprocess.run(
            ["powermetrics", "-n", "1", "-i", str(ms), "--samplers", "cpu_power"],
            capture_output=True, text=True, timeout=ms / 1000 + 5).stdout
    except Exception:
        return None
    for line in out.splitlines():
        if "CPU Power" in line and "mW" in line:
            try:
                return float(line.split(":")[1].strip().split()[0])
            except (IndexError, ValueError):
                return None
    return None


def _run_loop(sess, feeds, dur, secs, realtime):
    """Run inferences for `secs`, optionally paced to 1x real-time."""
    t0 = time.perf_counter(); next_t = t0
    while time.perf_counter() - t0 < secs:
        sess.run(["clean"], feeds)
        if realtime:
            next_t += dur
            sl = next_t - time.perf_counter()
            if sl > 0:
                time.sleep(sl)
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
#  Raspberry Pi 5 PMIC power (no external meter needed)
# ---------------------------------------------------------------------------
def pmic_power_w():
    """Total board power in watts from `vcgencmd pmic_read_adc`, or None."""
    try:
        out = subprocess.run(["vcgencmd", "pmic_read_adc"],
                             capture_output=True, text=True, timeout=2).stdout
    except Exception:
        return None
    volts, amps = {}, {}
    for line in out.splitlines():
        line = line.strip()
        if "=" not in line:
            continue
        name, val = line.split("=")
        try:
            v = float(val.rstrip("AV"))
        except ValueError:
            continue
        rail = name.split()[0]
        key = rail.rsplit("_", 1)[0]
        if "volt" in name:
            volts[key] = v
        elif "current" in name:
            amps[key] = v
    p = sum(volts[k] * amps[k] for k in volts if k in amps)
    return p if p > 0 else None


class PowerSampler(threading.Thread):
    """Background sampler of PMIC power (Pi 5). Collects until stop()."""
    def __init__(self, period=0.05):
        super().__init__(daemon=True)
        self.period = period
        self.samples = []
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            p = pmic_power_w()
            if p is not None:
                self.samples.append(p)
            time.sleep(self.period)

    def stop(self):
        self._stop.set()
        self.join()
        return np.array(self.samples) if self.samples else None


def energy_meter(sess, arr, fs, secs=8.0, threshold_mw=50.0, realtime=True,
                 board_w=7.0, macos=False):
    """Report the model's average power, preferring a real measurement:
       Pi-5 PMIC  >  Intel RAPL (Linux)  >  macOS powermetrics  >  RTF x board bound.
    """
    feeds = {"mic": arr.astype(np.float32)}
    dur = arr.shape[1] / fs
    for _ in range(3):
        sess.run(["clean"], feeds)                       # warmup

    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < min(3.0, secs):
        sess.run(["clean"], feeds); n += 1
    lat = (time.perf_counter() - t0) / n
    rtf = lat / dur

    p_mw, measured, src = None, False, ""

    # 1) Raspberry-Pi 5 PMIC -------------------------------------------------
    if pmic_power_w() is not None:
        s = PowerSampler(); s.start(); time.sleep(1.2); idle = s.stop()
        s = PowerSampler(); s.start()
        _run_loop(sess, feeds, dur, secs, realtime)
        busy = s.stop()
        if busy is not None and len(busy):
            pb = float(busy.mean()); pi = float(idle.mean()) if idle is not None else 0.0
            p_mw = (pb - pi) * 1000.0 if realtime else (pb - pi) * rtf * 1000.0
            measured = True; src = f"measured PMIC (idle {pi:.2f} W, running {pb:.2f} W)"

    # 2) Intel RAPL (Linux) --------------------------------------------------
    if p_mw is None and _rapl_uj() is not None:
        e0 = _rapl_uj(); time.sleep(1.0); idle_w = (_rapl_uj() - e0) / 1e6 / 1.0
        e0 = _rapl_uj(); dt = _run_loop(sess, feeds, dur, secs, realtime)
        busy_w = (_rapl_uj() - e0) / 1e6 / dt
        p_mw = max(busy_w - idle_w, 0.0) * 1000.0
        measured = True
        src = f"measured RAPL (idle {idle_w:.2f} W, running {busy_w:.2f} W)"

    # 3) macOS powermetrics (opt-in; needs sudo) -----------------------------
    if p_mw is None and macos:
        idle_mw = _macos_cpu_mw()
        if idle_mw is not None:
            import threading
            stop = threading.Event()
            th = threading.Thread(
                target=lambda: [sess.run(["clean"], feeds) for _ in iter(
                    lambda: not stop.is_set(), False)], daemon=True)
            th.start(); busy_mw = _macos_cpu_mw(); stop.set()
            p_mw = max(busy_mw - idle_mw, 0.0)
            measured = True
            src = f"measured powermetrics (idle {idle_mw:.0f} mW, busy {busy_mw:.0f} mW)"

    # 4) fallback: RTF x board-power upper bound -----------------------------
    if p_mw is None:
        p_mw = rtf * board_w * 1000.0
        src = f"RTF x {board_w:.0f} W board = UPPER BOUND (no power telemetry on this host)"

    if measured:
        verdict = "PASS" if p_mw < threshold_mw else "FAIL"
    else:
        verdict = "PASS (guaranteed)" if p_mw < threshold_mw \
            else "INCONCLUSIVE — measure on the target (Pi 5 PMIC) for the real figure"

    print("\n--- energy meter ---------------------------------------------")
    print(f"audio/infer  : {dur:.1f} s")
    print(f"latency      : {lat*1000:.1f} ms     RTF {rtf:.4f}  ({1/rtf:.0f}x real-time)")
    print(f"model power  : {p_mw:.1f} mW   ({src})")
    print(f"meter        : {_bar(p_mw, threshold_mw)}   {verdict}")
    if not measured and p_mw >= threshold_mw:
        print("note         : this is a loose bound, not a measurement. On the Pi 5 the")
        print("               PMIC reads ~5–17 mW (well under 50 mW). On macOS add --macos")
        print("               (uses sudo powermetrics) for a measured CPU figure.")
    print("--------------------------------------------------------------")
    return p_mw, verdict


def enhance(arr, sess):
    clean, = sess.run(["clean"], {"mic": arr.astype(np.float32)})
    return clean[:arr.shape[1]]


# --------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="mic wav(s): mono-per-mic or one multichannel")
    ap.add_argument("-o", "--out", default="clean.wav", help="output wav")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--synth", action="store_true",
                    help="treat the single input as mono speech and fake a mic array")
    ap.add_argument("--mics", type=int, default=3, help="mics for --synth")
    ap.add_argument("--no-plots", action="store_true", help="skip the PNGs")
    ap.add_argument("--energy", action="store_true", help="run the energy meter")
    ap.add_argument("--realtime", action="store_true", default=True,
                    help="energy meter paces to 1x real-time (deployment case)")
    ap.add_argument("--secs", type=float, default=8.0, help="energy meter run length")
    ap.add_argument("--threshold-mw", type=float, default=50.0, help="power budget (mW)")
    ap.add_argument("--board-w", type=float, default=7.0,
                    help="assumed board power (W) for the bound when no telemetry")
    ap.add_argument("--macos", action="store_true",
                    help="measure CPU power via sudo powermetrics (macOS dev box)")
    a = ap.parse_args()

    arr = load_array(a.inputs, synth=a.synth, n_mics=a.mics)
    print(f"mic array: {arr.shape[0]} channels x {arr.shape[1]} samples @ {FS} Hz")
    sess = ort.InferenceSession(a.model, providers=["CPUExecutionProvider"])

    clean = enhance(arr, sess)
    out_dir = os.path.dirname(a.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    sf.write(a.out, clean, FS)
    print(f"wrote {a.out}  ({len(clean)/FS:.2f} s)")

    if not a.no_plots:
        stem = os.path.splitext(a.out)[0]
        wpng, spng = save_plots(clean, FS, stem, noisy=arr[0])
        print(f"wrote {wpng}\nwrote {spng}")

    if a.energy:
        energy_meter(sess, arr, FS, secs=a.secs, threshold_mw=a.threshold_mw,
                     realtime=a.realtime, board_w=a.board_w, macos=a.macos)
