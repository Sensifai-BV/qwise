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

import numpy as np
import onnxruntime as ort


_SESSIONS = {}


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
