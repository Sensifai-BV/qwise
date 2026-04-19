"""
demo_onnx.py – ONNX-only VAD + Wiener filter speech enhancement.

Uses the official Silero VAD ONNX model (with state I/O) for inference,
and a pure-numpy decision-directed Wiener filter for enhancement.

Usage:
    python demo_onnx.py <noisy.wav>
    python demo_onnx.py <noisy.wav> -o clean.wav --plot-save result.png
    python demo_onnx.py <noisy.wav> --vad-model silero_vad_16k_recon.onnx
"""

import argparse
import os
import numpy as np
import onnxruntime as ort


# ---------------------------------------------------------------------------
# Audio I/O  (uses soundfile or torchaudio as fallback)
# ---------------------------------------------------------------------------
def load_audio(path: str, target_sr: int = 16000) -> np.ndarray:
    """Load audio file → mono float32 numpy array at target_sr."""
    try:
        import soundfile as sf
        wav, sr = sf.read(path, dtype='float32')
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != target_sr:
            import soxr
            wav = soxr.resample(wav, sr, target_sr)
    except ImportError:
        import torch, torchaudio
        t, sr = torchaudio.load(path)
        if t.shape[0] > 1:
            t = t.mean(0, keepdim=True)
        if sr != target_sr:
            t = torchaudio.transforms.Resample(sr, target_sr)(t)
        wav = t.squeeze().numpy()
    return wav.astype(np.float32)


def save_audio(path: str, wav: np.ndarray, sr: int = 16000):
    try:
        import soundfile as sf
        sf.write(path, wav, sr)
    except ImportError:
        import torch, torchaudio
        t = torch.from_numpy(wav).unsqueeze(0)
        torchaudio.save(path, t, sr)


# ---------------------------------------------------------------------------
# ONNX VAD wrapper  (uses the official silero_vad.onnx with state I/O)
# ---------------------------------------------------------------------------
class OnnxVAD:
    """Streaming VAD using the official Silero ONNX model (with state + sr inputs)."""

    def __init__(self, model_path: str):
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(model_path, sess_options=opts,
                                            providers=['CPUExecutionProvider'])
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]

        # Detect model type: official (input, state, sr) vs recon (input only)
        self._has_state_io = len(self.input_names) >= 3
        self.reset()

    def reset(self):
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(64, dtype=np.float32)

    def __call__(self, chunk: np.ndarray, sr: int = 16000) -> float:
        """
        chunk: (512,) float32 audio samples
        Returns: speech probability float
        """
        # Prepend context
        x = np.concatenate([self._context, chunk])[np.newaxis, :]  # (1, 576)
        self._context = x[0, -64:]

        if self._has_state_io:
            # Official model: inputs = {input, state, sr}
            feeds = {
                self.input_names[0]: x,
                self.input_names[1]: self._state,
                self.input_names[2]: np.array(sr, dtype=np.int64),
            }
            outs = self.session.run(None, feeds)
            prob = outs[0]
            if len(outs) > 1:
                self._state = outs[1]
        else:
            # Recon ONNX: single input, no state I/O (stateless per-chunk)
            feeds = {self.input_names[0]: chunk[np.newaxis, :]}  # (1, 512)
            outs = self.session.run(None, feeds)
            prob = outs[0]

        return float(np.squeeze(prob))

    def process_audio(self, wav: np.ndarray, sr: int = 16000,
                      chunk_size: int = 512) -> np.ndarray:
        """Process full waveform → per-chunk VAD probabilities."""
        self.reset()
        probs = []
        for start in range(0, len(wav), chunk_size):
            chunk = wav[start:start + chunk_size]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            probs.append(self(chunk, sr))
        return np.array(probs, dtype=np.float32)


# ---------------------------------------------------------------------------
# Decision-directed Wiener filter  (pure numpy)
# ---------------------------------------------------------------------------
def wiener_filter(noisy: np.ndarray, vad_probs: np.ndarray,
                  sr: int = 16000, chunk_size: int = 512,
                  n_fft: int = 512, hop: int = 128,
                  alpha_s: float = 0.95, dd_alpha: float = 0.98,
                  gain_floor_db: float = -25.0) -> np.ndarray:
    """
    Decision-directed Wiener filter guided by VAD probabilities.

    Parameters
    ----------
    noisy       : (T,) float32 time-domain signal
    vad_probs   : (N,) per-chunk speech probabilities
    n_fft       : FFT size
    hop         : STFT hop length
    alpha_s     : noise PSD smoothing (higher = slower update during speech)
    dd_alpha    : decision-directed SNR smoothing
    gain_floor_db : minimum gain in dB

    Returns
    -------
    clean       : (T,) enhanced signal
    """
    window = np.hanning(n_fft).astype(np.float32)
    gain_floor = 10.0 ** (gain_floor_db / 20.0)

    # STFT
    # Pad signal so we get integer number of frames
    pad_len = (n_fft - len(noisy) % hop) % hop
    noisy_pad = np.pad(noisy, (0, pad_len))
    n_frames = 1 + (len(noisy_pad) - n_fft) // hop
    freq_bins = n_fft // 2 + 1

    X = np.zeros((freq_bins, n_frames), dtype=np.complex64)
    for t in range(n_frames):
        seg = noisy_pad[t * hop: t * hop + n_fft] * window
        X[:, t] = np.fft.rfft(seg)

    power = np.abs(X) ** 2  # (freq_bins, n_frames)

    # Interpolate VAD probs to STFT frame rate
    vad_t = np.linspace(0, len(vad_probs) - 1, n_frames)
    vad_interp = np.interp(vad_t, np.arange(len(vad_probs)), vad_probs)

    # --- Recursive noise estimation + Wiener gain ---
    # Init noise PSD from first few frames
    n_init = min(5, n_frames)
    noise_psd = power[:, :n_init].mean(axis=1)  # (freq_bins,)

    gain = np.zeros_like(power)
    prev_gain = np.zeros(freq_bins, dtype=np.float32)

    for t in range(n_frames):
        frame_power = power[:, t]
        sp = vad_interp[t]  # speech probability for this frame

        # Adaptive noise update: slow during speech, fast during noise
        alpha_t = alpha_s * sp + (1.0 - sp) * 0.5
        noise_psd = alpha_t * noise_psd + (1.0 - alpha_t) * frame_power

        # A-posteriori SNR
        gamma = frame_power / (noise_psd + 1e-10)

        # Decision-directed a-priori SNR
        xi_ml = np.maximum(gamma - 1.0, 0.0)
        xi = dd_alpha * (prev_gain ** 2) * gamma + (1.0 - dd_alpha) * xi_ml
        xi = np.maximum(xi, 1e-4)

        # Wiener gain
        G = xi / (xi + 1.0)
        G = np.maximum(G, gain_floor)

        gain[:, t] = G
        prev_gain = G

    # Apply gain
    Y = X * gain

    # Inverse STFT (overlap-add)
    out_len = len(noisy_pad)
    clean = np.zeros(out_len, dtype=np.float32)
    win_sum = np.zeros(out_len, dtype=np.float32)

    for t in range(n_frames):
        seg = np.fft.irfft(Y[:, t], n=n_fft).astype(np.float32) * window
        start = t * hop
        clean[start:start + n_fft] += seg
        win_sum[start:start + n_fft] += window ** 2

    # Normalise by window overlap
    win_sum = np.maximum(win_sum, 1e-8)
    clean = clean / win_sum
    clean = clean[:len(noisy)]

    return clean


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_result(noisy, clean, vad_probs, sr=16000, chunk_size=512, save_path=None):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(14, 10),
                             gridspec_kw={'height_ratios': [1, 0.5, 1, 1]})

    t_wav = np.arange(len(noisy)) / sr

    axes[0].plot(t_wav, noisy, linewidth=0.3)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("Noisy input")
    axes[0].set_xlim(0, t_wav[-1])

    t_vad = np.linspace(0, len(noisy) / sr, len(vad_probs))
    axes[1].fill_between(t_vad, vad_probs, alpha=0.6, color="orange")
    axes[1].axhline(0.5, color="red", ls="--", lw=0.8, label="threshold")
    axes[1].set_ylabel("Speech prob")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("VAD (ONNX)")
    axes[1].legend(loc="upper right")
    axes[1].set_xlim(0, t_wav[-1])

    nfft = 512
    axes[2].specgram(noisy, NFFT=nfft, Fs=sr, noverlap=nfft // 2,
                     cmap='magma', vmin=-80, vmax=0)
    axes[2].set_ylabel("Hz")
    axes[2].set_title("Noisy spectrogram")
    axes[2].set_xlim(0, t_wav[-1])

    axes[3].specgram(clean, NFFT=nfft, Fs=sr, noverlap=nfft // 2,
                     cmap='magma', vmin=-80, vmax=0)
    axes[3].set_ylabel("Hz")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_title("Enhanced spectrogram")
    axes[3].set_xlim(0, t_wav[-1])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"📊 Plot saved → {save_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="ONNX VAD + Wiener filter speech enhancement")
    parser.add_argument("input", help="Noisy WAV file")
    parser.add_argument("-o", "--output", default=None,
                        help="Output WAV (default: enhanced_<input>)")
    parser.add_argument("--vad-model",
                        default="silero-vad/src/silero_vad/data/silero_vad.onnx",
                        help="Path to ONNX VAD model")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-save", default=None)
    # Wiener params
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop", type=int, default=128)
    parser.add_argument("--alpha-s", type=float, default=0.95,
                        help="Noise smoothing (0.8-0.99)")
    parser.add_argument("--dd-alpha", type=float, default=0.98,
                        help="Decision-directed smoothing (0.8-0.995)")
    parser.add_argument("--gain-floor", type=float, default=-25.0,
                        help="Spectral floor in dB")
    args = parser.parse_args()

    if args.output is None:
        base = os.path.basename(args.input)
        d = os.path.dirname(args.input) or "."
        args.output = os.path.join(d, f"enhanced_{base}")

    # Load audio
    print(f"🎤 Loading: {args.input}")
    noisy = load_audio(args.input)
    dur = len(noisy) / 16000
    print(f"   Duration: {dur:.2f}s  Samples: {len(noisy)}")

    # VAD
    print(f"🔧 Loading ONNX VAD: {args.vad_model}")
    vad = OnnxVAD(args.vad_model)
    vad_probs = vad.process_audio(noisy)
    n_speech = np.sum(vad_probs > 0.5)
    print(f"   VAD: {n_speech}/{len(vad_probs)} chunks = speech "
          f"({100 * n_speech / len(vad_probs):.1f}%)")

    # Wiener filter
    print("🧹 Applying Wiener filter …")
    clean = wiener_filter(
        noisy, vad_probs,
        n_fft=args.n_fft, hop=args.hop,
        alpha_s=args.alpha_s, dd_alpha=args.dd_alpha,
        gain_floor_db=args.gain_floor,
    )

    # Save
    save_audio(args.output, clean)
    print(f"✅ Saved → {args.output}")

    # Plot
    if args.plot or args.plot_save:
        plot_result(noisy, clean, vad_probs, save_path=args.plot_save)


if __name__ == "__main__":
    main()
