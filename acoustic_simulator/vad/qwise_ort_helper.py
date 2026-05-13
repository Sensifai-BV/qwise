"""Q-WiSE neural VAD ONNX inference helper for MATLAB's py.* bridge.

MATLAB's ``Deep Learning Toolbox Converter for ONNX Model Format`` support
package is not always available (importNetworkFromONNX /
importONNXFunction / importONNXNetwork can all be missing).  When that
happens we fall back to ``onnxruntime`` via the MATLAB<->Python bridge.

The module keeps a small registry of loaded sessions keyed by path so the
same session object survives between MATLAB ``step()`` calls without any
ORT state threading through MATLAB.

Audio-context quirk
-------------------
The v5 ONNX model expects ``CONTEXT_SIZE = 64`` samples of audio context
from the *previous* call prepended to each new 512-sample chunk; the
input tensor is therefore (1, 576), not (1, 512). Feeding only 512
samples returns near-zero probabilities for *every* chunk, including
clean voiced speech (verified empirically against this build of the
model). The helper threads the 64-sample context through MATLAB just like
the LSTM hidden state, so the simulator never has to worry about it.

Functions exposed to MATLAB:

    info = load(path)
        -> {'input_names':..., 'output_names':..., 'context_size': 64,
            'state_size': 256}

    out  = step(path, x, state, context, sr)
        -> {'prob':float, 'state':list[float], 'context':list[float]}

``x`` is a length-512 float iterable (the new chunk of audio).
``state`` is a length-256 flat iterable representing the [2,1,128] LSTM
hidden state. ``context`` is a length-64 flat iterable holding the tail
of the previous chunk that the model needs prepended; pass zeros on the
first call. ``sr`` is the sample rate in Hz (int).
"""

import numpy as np
import onnxruntime as ort


_SESSIONS = {}
CONTEXT_SIZE = 64
STATE_SIZE   = 2 * 1 * 128  # 256


def load(path):
    """Open the VAD ONNX model and cache the ORT session."""
    key = str(path)
    sess = ort.InferenceSession(key, providers=["CPUExecutionProvider"])
    in_names  = [i.name for i in sess.get_inputs()]
    out_names = [o.name for o in sess.get_outputs()]
    _SESSIONS[key] = (sess, in_names, out_names)
    return {
        "input_names":  in_names,
        "output_names": out_names,
        "context_size": CONTEXT_SIZE,
        "state_size":   STATE_SIZE,
    }


def _pick_feeds(in_names, x, state, sr):
    """Bind ORT feeds, normalising sr to a 0-D int64 ``ndarray``.

    onnxruntime rejects bare numpy scalars (``np.int64(sr)``) with
    ``RuntimeError: Unable to handle object of type <class
    'numpy.int64'>``; it requires ``numpy.ndarray`` for every feed.
    Wrapping ``sr`` with ``np.asarray(..., dtype=np.int64)`` returns a
    0-D ndarray that ORT accepts for the model's scalar ``sr`` input.
    """
    feeds = {}
    sr_arr = np.asarray(int(sr), dtype=np.int64)
    for n in in_names:
        nl = n.lower()
        if nl in ("h", "c", "h0", "c0") or "state" in nl or "hidden" in nl:
            feeds[n] = state
        elif "sr" in nl or "sample_rate" in nl or "rate" in nl:
            feeds[n] = sr_arr
        else:
            feeds[n] = x
    return feeds


def step(path, x, state, context, sr):
    """Run one Q-WiSE VAD inference step.

    MATLAB serializes arrays column-major (Fortran order); we reshape
    the incoming flat state with ``order='F'`` to recover the logical
    [layer, batch, hidden] layout, then re-serialize the new state the
    same way so MATLAB's default column-major reshape reconstructs it
    correctly. ``np.ascontiguousarray`` is required because onnxruntime
    needs C-contiguous tensors.

    The 64-sample audio context is prepended to ``x`` before inference,
    yielding the (1, 576) tensor the v5 model expects, and the trailing 64
    samples of ``x`` become the new context for the next call.
    """
    sess, in_names, out_names = _SESSIONS[str(path)]

    new_chunk = np.asarray(x, dtype=np.float32).reshape(-1)
    if new_chunk.size != 512:
        raise ValueError(
            f"qwise_ort_helper.step expects 512-sample chunks, got "
            f"{new_chunk.size}")

    ctx = np.asarray(context, dtype=np.float32).reshape(-1)
    if ctx.size != CONTEXT_SIZE:
        raise ValueError(
            f"qwise_ort_helper.step expects {CONTEXT_SIZE}-sample "
            f"context, got {ctx.size}")

    x_in = np.ascontiguousarray(
        np.concatenate([ctx, new_chunk]).reshape(1, -1).astype(np.float32))
    st = np.ascontiguousarray(
        np.asarray(state, dtype=np.float32).reshape(2, 1, 128, order="F"))

    feeds = _pick_feeds(in_names, x_in, st, sr)
    outs = sess.run(out_names, feeds)

    prob = float(np.asarray(outs[0]).reshape(-1)[0])
    if len(outs) > 1:
        new_state = np.asarray(outs[1], dtype=np.float32).reshape(2, 1, 128)
    else:
        new_state = st
    new_context = new_chunk[-CONTEXT_SIZE:].astype(np.float32)

    # Flatten column-major so MATLAB's reshape(flat,2,1,128) round-trips.
    return {
        "prob":    prob,
        "state":   new_state.flatten(order="F").astype(float).tolist(),
        "context": new_context.astype(float).tolist(),
    }
