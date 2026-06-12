"""Q-WiSE enhancer ONNX inference helper for MATLAB's py.* bridge.

The self-contained enhancer ONNX maps a raw multi-channel mic array
``mic[M, L]`` (float32, 16 kHz) to a single enhanced channel ``clean[L]``.
Running it through ``onnxruntime`` here — rather than marshalling numpy
arrays by hand on the MATLAB side — keeps all the array/dtype handling in
Python.

A module-level registry caches one ORT session per model path so the
session survives between MATLAB calls without threading any state through
MATLAB.

Functions exposed to MATLAB:

    info  = load(path)
        -> {'input_names': [...], 'output_names': [...]}

    clean = enhance(path, mic_flat, n_samples, n_mics)
        -> list[float] of length n_samples (enhanced mono speech)

``mic_flat`` is the [L x N] MATLAB matrix flattened column-major
(``mic(:)``); we reshape it back with ``order='F'`` and transpose to the
``[M, L]`` layout the model expects.
"""

import glob
import platform
import time

import numpy as np
import onnxruntime as ort


_SESSIONS = {}
FS = 16000


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


def load(path):
    """Open the enhancer ONNX model and cache the ORT session."""
    key = str(path)
    if key not in _SESSIONS:
        sess = ort.InferenceSession(key, providers=["CPUExecutionProvider"])
        in_names  = [i.name for i in sess.get_inputs()]
        out_names = [o.name for o in sess.get_outputs()]
        _SESSIONS[key] = (sess, in_names, out_names)
    sess, in_names, out_names = _SESSIONS[key]
    return {"input_names": in_names, "output_names": out_names}


def enhance(path, mic_flat, n_samples, n_mics):
    """Run one enhancement pass and return the clean signal as a list.

    MATLAB serializes arrays column-major (Fortran order), so a [L x N]
    matrix flattened with ``mic(:)`` is recovered with
    ``reshape(L, N, order='F')``; transposing gives the ``[M, L]`` tensor
    the model wants. ``ascontiguousarray`` is required because ORT needs
    C-contiguous input.
    """
    key = str(path)
    if key not in _SESSIONS:
        load(key)
    sess, in_names, out_names = _SESSIONS[key]

    L = int(n_samples)
    N = int(n_mics)
    flat = np.asarray(mic_flat, dtype=np.float32).reshape(-1)
    if flat.size != L * N:
        raise ValueError(
            f"enhance got {flat.size} samples, expected L*N = {L*N} "
            f"(L={L}, N={N}).")

    mic = flat.reshape(L, N, order="F").T          # [N, L] == [M, L]
    mic = np.ascontiguousarray(mic, dtype=np.float32)

    in_name  = in_names[0]  if in_names  else "mic"
    out_name = out_names[0] if out_names else "clean"
    clean = sess.run([out_name], {in_name: mic})[0]
    clean = np.asarray(clean, dtype=np.float64).reshape(-1)
    return clean.tolist()


def enhance_metered(path, mic_flat, n_samples, n_mics,
                    board_w=7.0, threshold_mw=50.0):
    """Enhance + estimate model power (run_demo.py energy-meter algorithm).

    Times the actual ONNX inference to get the real-time factor (RTF), then
    estimates power, preferring a measured figure (Intel-RAPL on Linux,
    idle-subtracted) and falling back to the conservative RTF x board-power
    upper bound used by run_demo.py. Returns the clean signal plus the
    metrics so MATLAB can draw the meter.
    """
    key = str(path)
    if key not in _SESSIONS:
        load(key)
    sess, in_names, out_names = _SESSIONS[key]

    L = int(n_samples)
    N = int(n_mics)
    flat = np.asarray(mic_flat, dtype=np.float32).reshape(-1)
    if flat.size != L * N:
        raise ValueError(
            f"enhance_metered got {flat.size} samples, expected {L*N}.")
    mic = np.ascontiguousarray(flat.reshape(L, N, order="F").T, dtype=np.float32)

    in_name  = in_names[0]  if in_names  else "mic"
    out_name = out_names[0] if out_names else "clean"
    feeds = {in_name: mic}

    # --- the run we actually keep ------------------------------------
    for _ in range(2):                       # warmup
        sess.run([out_name], feeds)
    clean = sess.run([out_name], feeds)[0]

    # --- stable latency over a few repeats ---------------------------
    reps = 5
    t0 = time.perf_counter()
    for _ in range(reps):
        sess.run([out_name], feeds)
    latency = (time.perf_counter() - t0) / reps
    audio_s = L / float(FS)
    rtf = latency / audio_s if audio_s > 0 else 0.0

    # --- power estimate ----------------------------------------------
    power_mw = None
    src = ""
    e0 = _rapl_uj()
    if e0 is not None:                       # measured (Intel-RAPL, Linux)
        ti = time.perf_counter(); time.sleep(0.15)
        idle_w = (_rapl_uj() - e0) / 1e6 / (time.perf_counter() - ti)
        e1 = _rapl_uj(); tb = time.perf_counter()
        while time.perf_counter() - tb < 0.25:
            sess.run([out_name], feeds)
        busy_w = (_rapl_uj() - e1) / 1e6 / (time.perf_counter() - tb)
        power_mw = max(busy_w - idle_w, 0.0) * 1000.0
        src = "measured RAPL"
    if power_mw is None:                     # RTF x board-power upper bound
        power_mw = rtf * board_w * 1000.0
        src = "RTF x %.0fW bound" % board_w

    clean = np.asarray(clean, dtype=np.float64).reshape(-1)
    return {
        "clean":        clean.tolist(),
        "power_mw":     float(power_mw),
        "threshold_mw": float(threshold_mw),
        "rtf":          float(rtf),
        "latency_ms":   float(latency * 1000.0),
        "audio_s":      float(audio_s),
        "src":          src,
        "os":           platform.system(),
    }
