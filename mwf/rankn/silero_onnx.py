"""
silero_onnx.py — rebuild the Silero-VAD v5 network as clean ONNX ops.

Loads the published weights from `silero_vad_16k.safetensors` (or the 8 k file)
and constructs the exact forward pass (per the repo's tinygrad_model.py) using
only standard ONNX operators — NO `If` wrapper and an EXPLICIT LSTM state. That
is what makes it embeddable in a single graph: it can be wrapped in a `Scan`
(unlike the shipped silero_vad.onnx, which is nested `If` subgraphs that crash
onnxruntime when scanned).

Per-chunk graph:
    inputs : chunk [1, ctx+win]   (64+512 @16k, 32+256 @8k)
             state [2, 1, 128]     (h = state[0], c = state[1])
    outputs: prob  [1, 1]
             stateN [2, 1, 128]

build_silero_chunk(fs) returns a spox build() dict-ready (Var) tuple; see
build_silero_chunk_model() for a standalone ONNX.
"""

import json
import struct
import numpy as np
import spox.opset.ai.onnx.v18 as op
from spox import argument, build, Tensor


def load_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        blob = f.read()
    out = {}
    for k, v in hdr.items():
        if k == "__metadata__":
            continue
        s, e = v["data_offsets"]
        out[k] = np.frombuffer(blob[s:e], np.float32).reshape(v["shape"]).copy()
    return out


def _c(v):
    return op.const(np.asarray(v, np.float32))


def _ci(v):
    return op.const(np.asarray(v, np.int64))


def silero_chunk_graph(W, fs, chunk, state):
    """Build the per-chunk Silero forward. chunk:[1, ctx+win], state:[2,1,128]
    -> prob:[1,1], stateN:[2,1,128]. W = safetensors dict."""
    ctx = 64 if fs == 16000 else 32

    # reflect-pad right by `ctx`, to [1,1,L+ctx]
    x = op.unsqueeze(chunk, axes=_ci([1]))                       # [1,1,L]
    x = op.pad(x, _ci([0, 0, 0, 0, 0, ctx]), mode="reflect")    # pad last dim end
    # STFT front-end (learned Conv1d), then magnitude of the 129 complex bins
    x = op.conv(x, _c(W["stft_conv.weight"]),
                kernel_shape=[256], strides=[128], pads=[0, 0])  # [1,258,T']
    re, im = op.split(x, axis=1, num_outputs=2)                 # 258 -> 129 + 129
    mag = op.sqrt(op.add(op.mul(re, re), op.mul(im, im)))        # [1,129,T']

    def conv(z, name, stride):
        z = op.conv(z, _c(W[name + ".weight"]), _c(W[name + ".bias"]),
                    kernel_shape=[3], strides=[stride], pads=[1, 1])
        return op.relu(z)
    a = conv(mag, "conv1", 1)
    a = conv(a, "conv2", 2)
    a = conv(a, "conv3", 2)
    a = conv(a, "conv4", 1)                                      # [1,128,1]
    xt = op.squeeze(a, axes=_ci([2]))                           # [1,128]

    # explicit LSTM cell
    h = op.squeeze(op.gather(state, _ci(0), axis=0), axes=_ci([]))  # [1,128]
    c = op.squeeze(op.gather(state, _ci(1), axis=0), axes=_ci([]))  # [1,128]
    Wih_T = _c(W["lstm_cell.weight_ih"].T)     # [128,512]
    Whh_T = _c(W["lstm_cell.weight_hh"].T)
    g = op.add(op.add(op.matmul(xt, Wih_T), _c(W["lstm_cell.bias_ih"])),
               op.add(op.matmul(h, Whh_T), _c(W["lstm_cell.bias_hh"])))   # [1,512]
    gi, gf, gg, go = op.split(g, axis=1, num_outputs=4)         # 512 -> 4 x 128
    c2 = op.add(op.mul(op.sigmoid(gf), c), op.mul(op.sigmoid(gi), op.tanh(gg)))
    h2 = op.mul(op.sigmoid(go), op.tanh(c2))                    # [1,128]

    z = op.unsqueeze(op.relu(h2), axes=_ci([2]))               # [1,128,1]
    z = op.conv(z, _c(W["final_conv.weight"]), _c(W["final_conv.bias"]),
                kernel_shape=[1], strides=[1], pads=[0, 0])     # [1,1,1]
    prob = op.reduce_mean(op.sigmoid(op.squeeze(z, axes=_ci([1]))),
                          axes=_ci([1]), keepdims=1)            # [1,1]
    stateN = op.concat([op.unsqueeze(h2, axes=_ci([0])),
                        op.unsqueeze(c2, axes=_ci([0]))], axis=0)  # [2,1,128]
    return prob, stateN


def chunk_len(fs):
    return (512 if fs == 16000 else 256) + (64 if fs == 16000 else 32)


def build_silero_chunk_model(safetensors_path, fs):
    W = load_safetensors(safetensors_path)
    L = chunk_len(fs)
    chunk = argument(Tensor(np.float32, (1, L)))
    state = argument(Tensor(np.float32, (2, 1, 128)))
    prob, stateN = silero_chunk_graph(W, fs, chunk, state)
    return build({"chunk": chunk, "state": state},
                 {"prob": prob, "stateN": stateN})


def build_silero_scan_model(safetensors_path, fs, opset=18):
    """Silero VAD over a whole utterance as ONE graph (the per-chunk net wrapped
    in a Scan that carries the LSTM state). No `If` nodes, so it is Scan-safe.

        inputs : state0 [2,1,128]   (zeros), vad_in [N, ctx+win]
        outputs: probs  [N]          per-chunk speech probability
                 final_state [2,1,128]
    """
    import onnx
    from onnx import helper, TensorProto as TP
    base = build_silero_chunk_model(safetensors_path, fs)
    g = base.graph
    L = chunk_len(fs)
    pre = [helper.make_node("Constant", [], ["__cs"],
                            value=helper.make_tensor("v", TP.INT64, [2], [1, L])),
           helper.make_node("Reshape", ["scan_chunk", "__cs"], ["chunk"])]
    post = [helper.make_node("Squeeze", ["prob"], ["prob1"])]   # [1,1] -> scalar
    body = helper.make_graph(
        pre + list(g.node) + post, "silero_body",
        [helper.make_tensor_value_info("state", TP.FLOAT, [2, 1, 128]),
         helper.make_tensor_value_info("scan_chunk", TP.FLOAT, [L])],
        [helper.make_tensor_value_info("stateN", TP.FLOAT, [2, 1, 128]),
         helper.make_tensor_value_info("prob1", TP.FLOAT, [])],
        list(g.initializer))
    scan = helper.make_node("Scan", ["state0", "vad_in"], ["final_state", "probs"],
                            body=body, num_scan_inputs=1)
    og = helper.make_graph(
        [scan], "silero_scan",
        [helper.make_tensor_value_info("state0", TP.FLOAT, [2, 1, 128]),
         helper.make_tensor_value_info("vad_in", TP.FLOAT, ["N", L])],
        [helper.make_tensor_value_info("probs", TP.FLOAT, ["N"]),
         helper.make_tensor_value_info("final_state", TP.FLOAT, [2, 1, 128])])
    m = helper.make_model(og, opset_imports=[helper.make_opsetid("", opset)])
    m.ir_version = base.ir_version
    return m
