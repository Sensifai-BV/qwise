#!/usr/bin/env python3
"""
bench_energy.py — latency + power/energy benchmark for the merged ONNX.

Run ON the Raspberry Pi (where the real numbers matter):

    python bench_energy.py mic01.wav --single --secs 20
    python bench_energy.py mic01.wav mic02.wav mic03.wav --single --secs 20

Energy cannot be derived from the model alone — it depends on the hardware — so
this measures it on the device:

  * latency per inference and the real-time factor (RTF = compute_time / audio_time)
  * board power, two ways:
      - Raspberry Pi 5: read the on-board PMIC via `vcgencmd pmic_read_adc`
        (sums V*I of the supply rails) -> idle vs busy power, no extra hardware.
      - Other Pi / no PMIC: reports latency + CPU load and a TDP-based estimate;
        for a true figure use an inline USB power meter and
        Energy = (P_busy - P_idle) * compute_time.

  Energy per second of audio (J/s) = busy_power(W) * RTF.
"""

import argparse
import os
import subprocess
import sys
import threading
import time
import numpy as np
import soundfile as sf

import rankn as R
import merged_ref as MR


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


# ---------------------------------------------------------------------------
def load_array(paths):
    chans, fs = [], None
    for p in paths:
        xi, fi = sf.read(p, always_2d=True)
        fs = fs or fi
        chans.append(xi)
    if len(chans) == 1:
        return chans[0], fs
    L = min(c.shape[0] for c in chans)
    return np.concatenate([c[:L, :1] for c in chans], axis=1), fs


def main():
    ap = argparse.ArgumentParser(description="Pi latency/energy benchmark")
    ap.add_argument("input", nargs="+")
    ap.add_argument("--single", action="store_true", help="benchmark the fully merged onnx")
    ap.add_argument("--secs", type=float, default=20.0, help="sustained run length (s)")
    ap.add_argument("--tdp", type=float, default=None,
                    help="assumed board active power (W) for an estimate if no PMIC")
    args = ap.parse_args()

    import onnxruntime as ort
    here = os.path.dirname(os.path.abspath(__file__))
    x, fs = load_array(args.input)
    dur = x.shape[0] / fs
    cfg = R.config_for(fs)
    X = R.stft_multi(x, cfg.n_fft, cfg.hop)
    vad_in, _ = MR._vad_chunks(x[:, cfg.ref_mic], fs)
    model = os.path.join(here, f"rankn_vad_{fs//1000}k.onnx" if args.single
                         else f"rankn_vadmask_{fs//1000}k.onnx")
    sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    if args.single:
        feeds = {"vad_in": vad_in.astype(np.float32),
                 "state0": np.zeros((2, 1, 128), np.float32),
                 "k_Xr": np.real(X).astype(np.float32), "k_Xi": np.imag(X).astype(np.float32)}
        outs = ["k_Yr", "k_Yi", "k_vprob"]
    else:
        feeds = {"probs": np.zeros(len(vad_in), np.float32),
                 "Xr": np.real(X).astype(np.float32), "Xi": np.imag(X).astype(np.float32)}
        outs = ["Yr", "Yi", "vprob"]

    for _ in range(3):
        sess.run(outs, feeds)                       # warmup

    has_pmic = pmic_power_w() is not None
    idle = None
    if has_pmic:
        time.sleep(0.5)
        s = PowerSampler(); s.start(); time.sleep(1.5); idle = s.stop()

    sampler = PowerSampler() if has_pmic else None
    if sampler:
        sampler.start()
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < args.secs:
        sess.run(outs, feeds)
        n += 1
    elapsed = time.perf_counter() - t0
    busy = sampler.stop() if sampler else None

    lat = elapsed / n
    rtf = lat / dur
    print(f"\nmodel        : {os.path.basename(model)}  ({os.path.getsize(model)//1024} KB)")
    print(f"audio/infer  : {dur:.1f}s   inferences: {n}")
    print(f"latency      : {lat*1000:.1f} ms/inference")
    print(f"real-time    : RTF {rtf:.4f}  ({1/rtf:.0f}x faster than real-time)")
    if busy is not None and len(busy):
        pb, pi = float(busy.mean()), float(idle.mean()) if idle is not None else 0.0
        print(f"power (PMIC) : idle {pi:.2f} W   busy {pb:.2f} W   delta {pb-pi:.2f} W")
        print(f"energy       : {pb*lat:.3f} J/inference   {pb*rtf:.3f} J per second of audio")
        print(f"              (incremental over idle: {(pb-pi)*lat:.3f} J/inference)")
    else:
        print("power        : no PMIC telemetry (not a Pi 5).")
        if args.tdp:
            print(f"estimate     : @ {args.tdp:.1f} W active -> "
                  f"{args.tdp*lat:.3f} J/inference, {args.tdp*rtf:.3f} J/s-audio")
        print("              For a real figure use an inline USB power meter:")
        print("              Energy = (P_busy - P_idle) [W] x latency [s].")


if __name__ == "__main__":
    main()
