#!/usr/bin/env python3
"""
Q-WiSE streaming inference demo  ─  Python stand-in for the C/Rust DSP layer.

Shows the exact calling convention the ONNX model expects in production.
Replace the Python STFT with CMSIS-DSP (ARM) or esp-dsp (Xtensa) on device.

Usage:
    python qwise_stream_demo.py  [path/to/noisy.wav]
"""

from __future__ import annotations
import numpy as np
from pathlib import Path

# ── constants must match the exported ONNX ────────────────────────────────────
N_FFT         = 512
HOP_LENGTH    = 128
SAMPLE_RATE   = 16_000
VAD_FRAME_LEN = 512   # one 32 ms window

# ─────────────────────────────────────────────────────────────────────────────
#  onnxruntime session
# ─────────────────────────────────────────────────────────────────────────────
def build_session(onnx_path: str = "qwise_vad_mwf.onnx"):
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    print(f"Loaded: {onnx_path}")
    for inp in sess.get_inputs():
        print(f"  in  {inp.name:<15} {inp.shape}")
    for out in sess.get_outputs():
        print(f"  out {out.name:<15} {out.shape}")
    return sess


# ─────────────────────────────────────────────────────────────────────────────
#  DSP helpers  (Python / NumPy stand-in for CMSIS-DSP / esp-dsp on device)
# ─────────────────────────────────────────────────────────────────────────────
def stft(frame: np.ndarray, n_fft: int = N_FFT, hop: int = HOP_LENGTH) -> np.ndarray:
    """Return complex STFT  (F, T)  for one mono frame."""
    window = np.hanning(n_fft).astype(np.float32)
    # pad frame to cover at least one hop
    padded = np.pad(frame, (n_fft // 2, n_fft // 2))
    n_frames = 1 + (len(padded) - n_fft) // hop
    out = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.complex64)
    for i in range(n_frames):
        seg = padded[i * hop : i * hop + n_fft] * window
        out[:, i] = np.fft.rfft(seg)
    return out


def istft(
    stft_complex: np.ndarray,
    n_fft: int = N_FFT,
    hop: int = HOP_LENGTH,
) -> np.ndarray:
    """Reconstruct time-domain signal from complex STFT  (F, T)."""
    window  = np.hanning(n_fft).astype(np.float32)
    n_frames = stft_complex.shape[1]
    length   = (n_frames - 1) * hop + n_fft
    out_buf  = np.zeros(length, dtype=np.float32)
    win_sum  = np.zeros(length, dtype=np.float32)
    for i in range(n_frames):
        frame = np.fft.irfft(stft_complex[:, i])[:n_fft] * window
        out_buf[i * hop : i * hop + n_fft] += frame
        win_sum[i * hop : i * hop + n_fft] += window ** 2
    # overlap-add normalise
    out_buf = np.where(win_sum > 1e-8, out_buf / np.maximum(win_sum, 1e-8), 0.0)
    return out_buf[n_fft // 2 : -(n_fft // 2)]  # trim padding


# ─────────────────────────────────────────────────────────────────────────────
#  Streaming enhancer
# ─────────────────────────────────────────────────────────────────────────────
class QWiseStreamEnhancer:
    """
    Stateful wrapper around the Q-WiSE ONNX model.

    Mirrors the C/Rust struct you would write on device:

        struct QWise {
            ort_session: *OrtSession,
            h: [f32; 2 * BATCH * 64],
            c: [f32; 2 * BATCH * 64],
        }
    """

    def __init__(self, onnx_path: str = "qwise_vad_mwf.onnx", batch: int = 1):
        self.sess  = build_session(onnx_path)
        self.batch = batch
        self._reset_state()

    def _reset_state(self):
        """Zero LSTM states — call at stream start or after long silence."""
        self.h = np.zeros((2, self.batch, 64), dtype=np.float32)
        self.c = np.zeros((2, self.batch, 64), dtype=np.float32)

    def process_frame(
        self,
        pcm_frame: np.ndarray,       # (512,) float32 @ 16 kHz  [−1, 1]
    ) -> tuple[np.ndarray, float]:
        """
        Process one 32 ms PCM frame.

        Returns:
            clean_frame  : (N,) enhanced PCM
            vad_prob     : float  speech activity ∈ [0, 1]
        """
        # ── DSP: STFT (this runs in C/Rust on device) ─────────────────────
        stft_complex = stft(pcm_frame)                           # (F, T)
        stft_mag     = np.abs(stft_complex).astype(np.float32)  # (F, T)

        # ── ONNX inputs ───────────────────────────────────────────────────
        feeds = {
            "audio_frame" : pcm_frame.reshape(self.batch, 1, -1),  # (B,1,512)
            "stft_mag"    : stft_mag[np.newaxis],                   # (B,F,T)
            "h_in"        : self.h,                                 # (2,B,64)
            "c_in"        : self.c,                                 # (2,B,64)
        }

        # ── ONNX inference ────────────────────────────────────────────────
        gain_mask, vad_prob, h_out, c_out = self.sess.run(None, feeds)

        # ── carry LSTM state forward (streaming) ──────────────────────────
        self.h = h_out
        self.c = c_out

        # ── apply Wiener mask and reconstruct ─────────────────────────────
        # gain_mask : (B, F, T)
        masked_stft = stft_complex * gain_mask[0]     # element-wise
        clean_frame = istft(masked_stft)

        return clean_frame, float(vad_prob[0, 0])


# ─────────────────────────────────────────────────────────────────────────────
#  Demo: process a WAV file frame-by-frame
# ─────────────────────────────────────────────────────────────────────────────
def process_wav(
    input_wav:  str = "noisy.wav",
    output_wav: str = "clean.wav",
    onnx_path:  str = "qwise_vad_mwf.onnx",
) -> None:
    try:
        import soundfile as sf
    except ImportError:
        print("pip install soundfile")
        return

    audio, sr = sf.read(input_wav, dtype="float32")
    assert sr == SAMPLE_RATE, f"Expected {SAMPLE_RATE} Hz, got {sr}"
    if audio.ndim > 1:
        audio = audio[:, 0]   # mono

    enhancer   = QWiseStreamEnhancer(onnx_path)
    clean_buf  = []
    hop        = VAD_FRAME_LEN

    # Process in 32 ms hops
    for start in range(0, len(audio) - hop + 1, hop):
        frame       = audio[start : start + hop]
        clean_frame, vad = enhancer.process_frame(frame)
        clean_buf.append(clean_frame)
        print(f"  t={start/SAMPLE_RATE:.3f}s  vad={vad:.3f}", end="\r")

    clean = np.concatenate(clean_buf)
    sf.write(output_wav, clean, SAMPLE_RATE)
    print(f"\n✓  Written: {output_wav}")


if __name__ == "__main__":
    import sys
    wav = sys.argv[1] if len(sys.argv) > 1 else "noisy.wav"
    if Path(wav).exists():
        process_wav(wav)
    else:
        print(f"[demo] {wav} not found — session load only")
        build_session()