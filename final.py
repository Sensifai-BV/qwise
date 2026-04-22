#!/usr/bin/env python3
"""
Q-WiSE  ─  Quantized Wiener Speech Enhancement
           Complete ONNX Export Pipeline

DSP boundary (▣ = ONNX model boundary):

  noisy_pcm ──► STFT[C/Rust] ──►▣───────────────────────────────────────────►▣
                                 │                                             │
                           audio_frame ──► SileroVAD16k ──► vad_prob          │
                           (B,1,512)          h/c state          │            │
                                                                  ▼            │
                           stft_mag ────────► QWiseNeuralMWF ──► gain_mask    │
                           (B,F,T)                                             │
                                                                               ▼
                                          clean_pcm ◄─ iSTFT ◄─ masked STFT ◄─┘

ONNX inputs:
  audio_frame  (B, 1, 512)   ─ 32 ms window @ 16 kHz (VAD input)
  stft_mag     (B, F, T)     ─ |STFT| magnitudes, F = n_fft//2 + 1 = 257
  h_in         (2, B, 64)    ─ Silero LSTM hidden state (carry between calls)
  c_in         (2, B, 64)    ─ Silero LSTM cell   state (carry between calls)

ONNX outputs:
  gain_mask    (B, F, T)     ─ Wiener gain mask ∈ [0, 1]
  vad_prob     (B, 1)        ─ speech activity ∈ [0, 1]
  h_out        (2, B, 64)    ─ updated LSTM hidden state  ──► feed back as h_in
  c_out        (2, B, 64)    ─ updated LSTM cell   state  ──► feed back as c_in

C / Rust integration sketch:
  let mask  = onnx_session.run([audio_frame, stft_mag, h, c]);
  let clean = istft(stft_complex * mask);   // element-wise Wiener filtering
  (h, c)    = (mask.h_out, mask.c_out);    // stateful streaming
"""

from __future__ import annotations
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

# ── global constants ──────────────────────────────────────────────────────────
VAD_WEIGHTS   = "silero-vad/src/silero_vad/data/silero_vad_16k.safetensors"
N_FFT         = 512
HOP_LENGTH    = 128
SAMPLE_RATE   = 16_000
N_FREQ        = N_FFT // 2 + 1      # 257
VAD_FRAME_LEN = 512                  # samples per VAD window at 16 kHz
LSTM_LAYERS   = 2
LSTM_HIDDEN   = 64


# ═══════════════════════════════════════════════════════════════════════════════
# § 0  Inspection utility
#     Run this first to verify safetensors key names vs. architecture below.
#     Compare the printed keys against SileroVAD16k.state_dict() keys and
#     adjust _REMAP in load_silero_weights() as needed.
# ═══════════════════════════════════════════════════════════════════════════════
def inspect_safetensors(path: str) -> dict[str, torch.Tensor]:
    """Print every tensor key / shape / dtype in a safetensors file."""
    weights = load_file(path)
    bar = "─" * 74
    print(f"\n{bar}")
    print(f"  Safetensors: {path}  ({len(weights)} tensors)")
    print(bar)
    for k in sorted(weights):
        v = weights[k]
        print(f"  {k:<58}  {str(list(v.shape)):<22}  {v.dtype}")
    print(f"{bar}\n")
    return weights


# ═══════════════════════════════════════════════════════════════════════════════
# § 1  Silero VAD 16 kHz  ─ PyTorch reconstruction
#
#  Architecture matches the published Silero VAD v5 16 kHz model:
#   • stem     Conv1d(1→64, k=3)
#   • 3 ×      depthwise-separable Conv1d blocks (64 ch)
#   • pool     AdaptiveAvgPool1d(1)  → single 64-d frame embedding
#   • recurrent 2-layer LSTM (64 hidden)
#   • decoder  Linear(64→1) + Sigmoid
#
#  ⚠  If inspect_safetensors() reveals different key names or channel counts,
#     adjust the _REMAP dict in load_silero_weights() and/or the __init__ below.
# ═══════════════════════════════════════════════════════════════════════════════
class _DSBlock(nn.Module):
    """Depthwise-separable 1-D conv block (no BN – avoids batch=1 issues)."""
    def __init__(self, ch: int, kernel: int = 3):
        super().__init__()
        pad = kernel // 2
        # reparameterizable-style block: dw → pw → activation
        self.dw = nn.Conv1d(ch, ch, kernel, padding=pad, groups=ch, bias=False)
        self.pw = nn.Conv1d(ch, ch, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.pw(self.dw(x)))


class SileroVAD16k(nn.Module):
    """
    Streaming Silero VAD 16 kHz wrapper.

    Call once per 32 ms audio frame; pass LSTM states between calls for
    temporally coherent detection without re-processing past context.

    Args:
        audio_frame : (B, 1, 512)
        h           : (2, B, 64)   LSTM hidden state  ← use initial_state() at t=0
        c           : (2, B, 64)   LSTM cell   state
    Returns:
        vad_prob    : (B, 1)       speech probability ∈ [0, 1]
        h_out       : (2, B, 64)
        c_out       : (2, B, 64)
    """

    def __init__(self):
        super().__init__()
        # ── encoder ──────────────────────────────────────────────────────────
        self.stem   = nn.Conv1d(1, 64, 3, padding=1, bias=True)
        self.block1 = _DSBlock(64)
        self.block2 = _DSBlock(64)
        self.block3 = _DSBlock(64)
        self.pool   = nn.AdaptiveAvgPool1d(1)   # (B, 64, 1)

        # ── recurrent ────────────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=LSTM_HIDDEN,
            num_layers=LSTM_LAYERS,
            batch_first=True,
        )

        # ── decoder ──────────────────────────────────────────────────────────
        self.decoder = nn.Linear(LSTM_HIDDEN, 1)

    def forward(
        self,
        audio_frame: torch.Tensor,   # (B, 1, 512)
        h: torch.Tensor,             # (2, B, 64)
        c: torch.Tensor,             # (2, B, 64)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # Encode: (B, 1, 512) → (B, 64, 1)
        x = F.relu(self.stem(audio_frame))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)                         # (B, 64, 1)

        # Reshape for LSTM: (B, 1, 64)
        x = x.squeeze(-1).unsqueeze(1)

        # Stateful LSTM
        out, (h_n, c_n) = self.lstm(x, (h, c))  # out: (B, 1, 64)

        # Speech probability
        prob = torch.sigmoid(self.decoder(out[:, -1, :]))  # (B, 1)
        return prob, h_n, c_n

    # ── helpers ───────────────────────────────────────────────────────────────
    @torch.no_grad()
    def initial_state(
        self, batch: int = 1, device: torch.device | str = "cpu"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.zeros(LSTM_LAYERS, batch, LSTM_HIDDEN, device=device)
        return z, z.clone()


def load_silero_weights(
    model: SileroVAD16k, path: str, verbose: bool = True
) -> SileroVAD16k:
    """
    Load silero_vad_16k.safetensors into SileroVAD16k (strict=False).

    _REMAP translates safetensors key prefixes → SileroVAD16k key names.
    Common prefix patterns across package versions:
      "_model."        ← silero-vad v4 / v5 package
      ""               ← some community exports (no prefix)
    Extend _REMAP if your inspection output shows a different pattern.
    """
    weights = load_file(path)

    # ── key remapping ─────────────────────────────────────────────────────────
    # Map safetensors key → model state_dict key.
    # Edit these rules after running inspect_safetensors() if needed.
    _PREFIX_STRIP = [
        "_model.",        # silero-vad v5 package default
        "model.",         # some community checkpoints
    ]
    # Explicit per-key aliases  {safetensors_key: model_key}
    _ALIASES: dict[str, str] = {
        # Example: "encoder.0.reparam_conv.weight": "stem.weight"
        # Fill in after comparing inspect_safetensors() output with
        # model.state_dict().keys()
    }

    remapped: dict[str, torch.Tensor] = {}
    for k, v in weights.items():
        # Apply explicit aliases first
        if k in _ALIASES:
            remapped[_ALIASES[k]] = v
            continue
        # Strip known prefixes
        clean = k
        for pfx in _PREFIX_STRIP:
            if clean.startswith(pfx):
                clean = clean[len(pfx):]
                break
        remapped[clean] = v

    missing, unexpected = model.load_state_dict(remapped, strict=False)

    if verbose:
        print(f"[SileroVAD] Loaded weights from {path}")
        loaded = len(weights) - len(unexpected)
        total  = len(model.state_dict())
        print(f"  Matched  : {loaded} / {total} tensors")
        if missing:
            print(f"  Missing  ({len(missing)}): {missing[:6]}")
        if unexpected:
            print(f"  Unexpected ({len(unexpected)}): {list(unexpected)[:6]}")
        if not missing and not unexpected:
            print("  ✓ Perfect match — all keys loaded.")

    return model


# ═══════════════════════════════════════════════════════════════════════════════
# § 2  Q-WiSE Neural MWF  ─ attention-based Wiener mask predictor
#
#  Pipeline per frame batch:
#    stft_mag (B,F,T) + vad_prob (B,1)
#      ──► freq_proj   Linear(F+1 → hidden)   fuses spectral features + VAD
#      ──► N × temporal self-attention (TransformerEncoderLayer)
#      ──► mask_head   Linear(hidden → F) + Sigmoid
#      ──► VAD soft gate:  mask ← mask * vad_prob   (dynamic sparsity)
#
#  Sizing guide for edge targets (tune mwf_hidden / mwf_layers in QWisePipeline):
#    STM32 MP1 / Cortex-M55 : hidden=64,  heads=4, layers=1
#    ESP32-S3 (SRAM ~512 kB): hidden=128, heads=4, layers=2  ← default
#    nRF5340 (tight budget)  : hidden=32,  heads=2, layers=1
# ═══════════════════════════════════════════════════════════════════════════════
class QWiseNeuralMWF(nn.Module):
    """
    Neural-guided Wiener filter mask predictor with temporal attention.

    Args:
        stft_mag  : (B, F, T)   magnitude spectrogram
        vad_prob  : (B, 1)      speech activity from SileroVAD
    Returns:
        mask      : (B, F, T)   Wiener gain mask ∈ [0, 1]
    """

    def __init__(
        self,
        n_freq: int   = N_FREQ,    # 257
        hidden: int   = 128,
        n_heads: int  = 4,         # hidden % n_heads == 0  required
        n_layers: int = 2,
        dropout: float = 0.0,      # keep 0 for export; use >0 during training
    ):
        super().__init__()
        assert hidden % n_heads == 0, \
            f"hidden ({hidden}) must be divisible by n_heads ({n_heads})"

        # Frequency + VAD feature projection
        # concat stft_mag (F) with vad_prob (1) along freq axis → F+1
        self.freq_proj = nn.Linear(n_freq + 1, hidden)

        # Multi-head temporal self-attention
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=n_heads,
            dim_feedforward=hidden * 2,   # 2× keeps param count low for edge
            dropout=dropout,
            batch_first=True,
            norm_first=True,              # pre-LN: better gradient flow in QAT
        )
        self.temporal_attn = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Mask prediction head
        self.mask_head = nn.Linear(hidden, n_freq)

    def forward(
        self,
        stft_mag: torch.Tensor,    # (B, F, T)
        vad_prob: torch.Tensor,    # (B, 1)
    ) -> torch.Tensor:             # (B, F, T)

        # (B, T, F) — time-major for the Transformer
        x = stft_mag.permute(0, 2, 1)                          # (B, T, F)

        # Append vad_prob to every time step → (B, T, F+1).
        # FIX: torch.ones_like + mul is dynamo/FakeTensor-safe; expand(-1,T,-1)
        # crashes in PyTorch ≥ 2.1 dynamo because T is a SymInt there.
        vad_exp = vad_prob.unsqueeze(1) * torch.ones(
            x.shape[0], x.shape[1], 1, dtype=x.dtype, device=x.device
        )                                                        # (B, T, 1)
        x = torch.cat([x, vad_exp], dim=-1)                     # (B, T, F+1)

        # Frequency projection + non-linearity
        x = F.relu(self.freq_proj(x))            # (B, T, hidden)

        # Temporal context via self-attention
        x = self.temporal_attn(x)                # (B, T, hidden)

        # Mask ∈ [0, 1]
        mask = torch.sigmoid(self.mask_head(x))  # (B, T, F)

        # VAD soft gate: vad_prob ≈ 0 → collapse mask (dynamic sparsity)
        # vad_prob.unsqueeze(1) shape (B,1,1) broadcasts cleanly over (B,T,F)
        mask = mask * vad_prob.unsqueeze(1)      # (B, T, 1) broadcast

        return mask.permute(0, 2, 1)             # (B, F, T)


# ═══════════════════════════════════════════════════════════════════════════════
# § 3  Unified pipeline  ─ the single ONNX graph
# ═══════════════════════════════════════════════════════════════════════════════
class QWisePipeline(nn.Module):
    """
    Full Q-WiSE pipeline: SileroVAD16k → QWiseNeuralMWF.

    This is the single module exported to ONNX.  All STFT / iSTFT and
    complex-number arithmetic remain in C / Rust.

    Streaming usage (C / Rust pseudo-code):
        h, c = zeros(2, 1, 64), zeros(2, 1, 64)   // once at startup
        loop:
            frame  = pcm_ring.read(512)
            stft   = stft(frame)                   // DSP side
            mask, vad, h, c = onnx.run(frame, |stft|, h, c)
            clean  = istft(stft * mask)
    """

    def __init__(
        self,
        vad_weights:  str | None = None,
        n_fft:        int = N_FFT,
        mwf_hidden:   int = 128,
        mwf_heads:    int = 4,
        mwf_layers:   int = 2,
    ):
        super().__init__()
        self.vad = SileroVAD16k()
        self.mwf = QWiseNeuralMWF(
            n_freq   = n_fft // 2 + 1,
            hidden   = mwf_hidden,
            n_heads  = mwf_heads,
            n_layers = mwf_layers,
        )

        if vad_weights and Path(vad_weights).exists():
            load_silero_weights(self.vad, vad_weights)
        elif vad_weights:
            print(f"[WARNING] VAD weights not found: {vad_weights} — using random init")

    def forward(
        self,
        audio_frame: torch.Tensor,   # (B, 1, 512)
        stft_mag:    torch.Tensor,   # (B, F, T)
        h_in:        torch.Tensor,   # (2, B, 64)
        c_in:        torch.Tensor,   # (2, B, 64)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        vad_prob, h_out, c_out = self.vad(audio_frame, h_in, c_in)
        gain_mask = self.mwf(stft_mag, vad_prob)
        return gain_mask, vad_prob, h_out, c_out


# ═══════════════════════════════════════════════════════════════════════════════
# § 4  ONNX export + verification
# ═══════════════════════════════════════════════════════════════════════════════
def export_onnx(
    output_path:  str = "qwise_vad_mwf.onnx",
    vad_weights:  str = VAD_WEIGHTS,
    n_fft:        int = N_FFT,
    mwf_hidden:   int = 128,
    mwf_heads:    int = 4,
    mwf_layers:   int = 2,
    opset:        int = 17,
    do_inspect:   bool = True,
) -> None:
    sep = "═" * 68

    # ── 0. inspect weights ────────────────────────────────────────────────────
    if do_inspect and Path(vad_weights).exists():
        print(f"\n{sep}")
        print("  § 0  Safetensors Inspection")
        print(sep)
        inspect_safetensors(vad_weights)
    elif not Path(vad_weights).exists():
        print(f"[INFO] {vad_weights} not found — exporting with random VAD weights")

    # ── 1. build model ────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  § 1  Building QWisePipeline")
    print(sep)
    model = QWisePipeline(
        vad_weights = vad_weights,
        n_fft       = n_fft,
        mwf_hidden  = mwf_hidden,
        mwf_heads   = mwf_heads,
        mwf_layers  = mwf_layers,
    )
    model.eval()

    n_freq = n_fft // 2 + 1   # 257
    B, T   = 1, 10             # dummy batch / time frames

    dummy_audio = torch.randn(B, 1, VAD_FRAME_LEN)
    dummy_stft  = torch.randn(B, n_freq, T).abs()   # magnitudes are non-negative
    dummy_h     = torch.zeros(LSTM_LAYERS, B, LSTM_HIDDEN)
    dummy_c     = torch.zeros(LSTM_LAYERS, B, LSTM_HIDDEN)

    # ── 2. forward sanity check ───────────────────────────────────────────────
    print(f"\n{sep}")
    print("  § 2  Forward Pass Verification (PyTorch)")
    print(sep)
    with torch.no_grad():
        mask, prob, h_o, c_o = model(dummy_audio, dummy_stft, dummy_h, dummy_c)
    print(f"  gain_mask  : {tuple(mask.shape)}  min={mask.min():.4f}  max={mask.max():.4f}")
    print(f"  vad_prob   : {tuple(prob.shape)}  val={prob.item():.4f}")
    print(f"  h_out      : {tuple(h_o.shape)}")
    print(f"  c_out      : {tuple(c_o.shape)}")

    # ── 3. export ─────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print(f"  § 3  Exporting to ONNX (opset {opset})")
    print(sep)

    # WHY dynamo=False:
    # PyTorch >=2.1 defaults torch.onnx.export to a dynamo-based path that runs
    # torch.export.export internally, using FakeTensor tracing.  In that path:
    #   1. tensor.size(N) returns a SymInt, not an int.  Passing a SymInt to
    #      Tensor.expand() crashes: "requested shape has too few dimensions".
    #   2. nn.LSTM._flat_weights are not registered as nn.Module buffers,
    #      triggering a UserWarning during graph capture.
    # The legacy TorchScript exporter (dynamo=False) handles dynamic_axes
    # natively without FakeTensors and sidesteps both issues completely.
    _export_kwargs = dict(
        export_params       = True,
        opset_version       = opset,
        do_constant_folding = True,
        input_names  = ["audio_frame", "stft_mag", "h_in",  "c_in"],
        output_names = ["gain_mask",   "vad_prob", "h_out", "c_out"],
        dynamic_axes = {
            "audio_frame" : {0: "batch"},
            "stft_mag"    : {0: "batch", 2: "time_frames"},
            "h_in"        : {1: "batch"},
            "c_in"        : {1: "batch"},
            "gain_mask"   : {0: "batch", 2: "time_frames"},
            "vad_prob"    : {0: "batch"},
            "h_out"       : {1: "batch"},
            "c_out"       : {1: "batch"},
        },
    )
    _args = (dummy_audio, dummy_stft, dummy_h, dummy_c)

    try:
        # PyTorch >= 2.1: dynamo kwarg exists — force legacy TorchScript path
        torch.onnx.export(model, _args, output_path, dynamo=False, **_export_kwargs)
        print("  Used: legacy TorchScript exporter (dynamo=False)")
    except TypeError:
        # Older PyTorch: no dynamo kwarg — explicit jit.trace achieves the same
        print("  dynamo kwarg unavailable — falling back to torch.jit.trace")
        with torch.no_grad():
            traced = torch.jit.trace(model, _args, strict=False)
        torch.onnx.export(traced, _args, output_path, **_export_kwargs)
        print("  Used: torch.jit.trace + legacy exporter")

    size_mb = Path(output_path).stat().st_size / 1024 / 1024
    print(f"  ✓  Saved → {output_path}  ({size_mb:.2f} MB)")

    # ── 4. onnxruntime verification ───────────────────────────────────────────
    print(f"\n{sep}")
    print("  § 4  ONNX Runtime Verification")
    print(sep)
    try:
        import onnxruntime as ort
        import numpy as np

        sess = ort.InferenceSession(
            output_path,
            providers=["CPUExecutionProvider"],
        )

        feeds = {
            "audio_frame" : dummy_audio.numpy(),
            "stft_mag"    : dummy_stft.numpy(),
            "h_in"        : dummy_h.numpy(),
            "c_in"        : dummy_c.numpy(),
        }
        outputs = sess.run(None, feeds)

        names = ["gain_mask", "vad_prob", "h_out", "c_out"]
        for name, arr in zip(names, outputs):
            print(f"  {name:<12}  {str(arr.shape):<20}  "
                  f"min={arr.min():.4f}  max={arr.max():.4f}")

        # Numerical diff vs PyTorch
        pt_outputs = [mask.numpy(), prob.numpy(), h_o.numpy(), c_o.numpy()]
        for name, ort_arr, pt_arr in zip(names, outputs, pt_outputs):
            diff = np.abs(ort_arr - pt_arr).max()
            print(f"  max|Δ| ({name}): {diff:.2e}"
                  + ("  ✓" if diff < 1e-4 else "  ⚠ check tolerances"))

    except ImportError:
        print("  onnxruntime not installed → pip install onnxruntime")
        print("  Skipping runtime verification.")

    print(f"\n{sep}")
    print("  Export complete.")
    print(sep)


# ═══════════════════════════════════════════════════════════════════════════════
# § 5  Entry point
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # ── Inspect only (no export) ──────────────────────────────────────────────
    # Uncomment to print safetensors layout before committing to an architecture:
    #   inspect_safetensors(VAD_WEIGHTS)
    #   sys.exit(0)

    export_onnx(
        output_path = "qwise_vad_mwf.onnx",
        vad_weights = VAD_WEIGHTS,

        # ── MWF sizing (tune for your target MCU) ─────────────────────────────
        # STM32 MP1 / Cortex-M55 : hidden=64,  heads=4, layers=1
        # ESP32-S3  (512 kB SRAM): hidden=128, heads=4, layers=2  ← default
        # nRF5340   (tight budget): hidden=32,  heads=2, layers=1
        mwf_hidden  = 128,
        mwf_heads   = 4,
        mwf_layers  = 2,

        opset       = 17,
        do_inspect  = True,
    )