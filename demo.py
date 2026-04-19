"""
demo.py – Take a noisy speech WAV, enhance it with VAD + Wiener filter, save clean WAV.

Usage:
    python demo.py <noisy.wav>                        # → output: enhanced_<noisy>.wav
    python demo.py <noisy.wav> -o clean_output.wav    # → output: clean_output.wav
    python demo.py <noisy.wav> --plot                  # also show VAD + spectrogram plot
"""

import argparse
import os
import torch
import torchaudio
import torch.nn.functional as F
from recon import load_silero_vad_pt, WienerFilter, VADWienerPipeline


def load_audio(path: str, target_sr: int = 16000) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return wav  # (1, T)


def save_audio(path: str, wav: torch.Tensor, sr: int = 16000):
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    torchaudio.save(path, wav.cpu(), sr)


def plot_result(noisy: torch.Tensor, clean: torch.Tensor,
                vad_probs: torch.Tensor, sr: int = 16000, save_path: str = None):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [1, 0.5, 1, 1]})

    t_noisy = np.arange(noisy.shape[-1]) / sr
    t_clean = np.arange(clean.shape[-1]) / sr

    # 1) Noisy waveform
    axes[0].plot(t_noisy, noisy.squeeze().numpy(), linewidth=0.3)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("Noisy input")
    axes[0].set_xlim(0, t_noisy[-1])

    # 2) VAD probabilities
    probs = vad_probs.squeeze().numpy()
    t_vad = np.linspace(0, noisy.shape[-1] / sr, len(probs))
    axes[1].fill_between(t_vad, probs, alpha=0.6, color="orange")
    axes[1].axhline(0.5, color="red", linestyle="--", linewidth=0.8, label="threshold=0.5")
    axes[1].set_ylabel("Speech prob")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("VAD probabilities")
    axes[1].legend(loc="upper right")
    axes[1].set_xlim(0, t_noisy[-1])

    # 3) Noisy spectrogram
    n_fft = 512
    noisy_np = noisy.squeeze().numpy()
    axes[2].specgram(noisy_np, NFFT=n_fft, Fs=sr, noverlap=n_fft//2,
                     cmap='magma', vmin=-80, vmax=0)
    axes[2].set_ylabel("Freq (Hz)")
    axes[2].set_title("Noisy spectrogram")
    axes[2].set_xlim(0, t_noisy[-1])

    # 4) Enhanced spectrogram
    clean_np = clean.squeeze().numpy()
    axes[3].specgram(clean_np, NFFT=n_fft, Fs=sr, noverlap=n_fft//2,
                     cmap='magma', vmin=-80, vmax=0)
    axes[3].set_ylabel("Freq (Hz)")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_title("Enhanced spectrogram")
    axes[3].set_xlim(0, t_noisy[-1])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"📊 Plot saved → {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="VAD + Wiener speech enhancement demo")
    parser.add_argument("input", help="Path to noisy speech WAV file")
    parser.add_argument("-o", "--output", default=None, help="Output WAV path (default: enhanced_<input>)")
    parser.add_argument("--plot", action="store_true", help="Show/save VAD + waveform plot")
    parser.add_argument("--plot-save", default=None, help="Save plot to this path instead of showing")
    args = parser.parse_args()

    if args.output is None:
        base = os.path.basename(args.input)
        args.output = os.path.join(os.path.dirname(args.input) or ".", f"enhanced_{base}")

    print(f"🎤 Loading: {args.input}")
    noisy = load_audio(args.input)
    duration = noisy.shape[-1] / 16000
    print(f"   Duration: {duration:.2f}s  Samples: {noisy.shape[-1]}")

    print("🔧 Loading Silero VAD + Wiener filter …")
    vad = load_silero_vad_pt()
    wiener = WienerFilter()
    pipe = VADWienerPipeline(vad, wiener)

    print("🧹 Enhancing …")
    with torch.no_grad():
        clean, vad_probs = pipe(noisy)

    save_audio(args.output, clean)
    print(f"✅ Saved enhanced audio → {args.output}")

    n_speech = (vad_probs > 0.5).sum().item()
    n_total = vad_probs.numel()
    print(f"   VAD: {n_speech}/{n_total} chunks classified as speech ({100*n_speech/n_total:.1f}%)")

    if args.plot or args.plot_save:
        plot_result(noisy, clean, vad_probs, save_path=args.plot_save)


if __name__ == "__main__":
    main()
