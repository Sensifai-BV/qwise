"""
improve_audio.py – Add silent padding before/after a clean-speech WAV and/or
                   boost its volume by 2× (or any custom multiplier).

Usage:
    # Add 1 s silence before and 0.5 s after, then double the volume:
    python improve_audio.py input.wav

    # Custom padding and gain:
    python improve_audio.py input.wav \\
        --pad-before 2.0 \\
        --pad-after  1.0 \\
        --gain 3.0    \\
        -o output.wav

Switches:
    --pad-before SECONDS   Seconds of silence to prepend  (default: 1.0)
    --pad-after  SECONDS   Seconds of silence to append   (default: 1.0)
    --gain       FACTOR    Volume multiplier               (default: 2.0)
    -o / --output PATH     Output file path               (default: improved_<input>)
    --no-clip              Soft-clip instead of hard-clipping when signal > 1.0
"""

import argparse
import os

import torch
import torchaudio


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_audio(path: str, target_sr: int = 16_000) -> tuple[torch.Tensor, int]:
    """Load a WAV, downmix to mono, resample to *target_sr*."""
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:                          # stereo/multi → mono
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return wav, target_sr                          # (1, T), sr


def save_audio(path: str, wav: torch.Tensor, sr: int) -> None:
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    torchaudio.save(path, wav.cpu(), sr)


def add_silence(wav: torch.Tensor, sr: int,
                before: float = 0.0, after: float = 0.0) -> torch.Tensor:
    """Prepend *before* seconds and append *after* seconds of silence."""
    n_before = int(round(before * sr))
    n_after  = int(round(after  * sr))
    parts = []
    if n_before > 0:
        parts.append(torch.zeros(wav.shape[0], n_before))
    parts.append(wav)
    if n_after > 0:
        parts.append(torch.zeros(wav.shape[0], n_after))
    return torch.cat(parts, dim=1)


def amplify(wav: torch.Tensor, gain: float = 2.0, soft_clip: bool = False) -> torch.Tensor:
    """Multiply amplitude by *gain*.  Clip to [-1, 1] to avoid distortion."""
    wav = wav * gain
    if soft_clip:
        # tanh soft-clip: preserves shape but rolls off gently above ±1
        wav = torch.tanh(wav)
    else:
        wav = torch.clamp(wav, -1.0, 1.0)
    return wav


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pad silence + boost volume of a clean-speech WAV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input",
                   help="Path to the input .wav file")
    p.add_argument("-o", "--output", default=None,
                   help="Output .wav path  (default: improved_<input>)")
    p.add_argument("--pad-before", type=float, default=1.0, metavar="SECONDS",
                   help="Seconds of silence to add BEFORE the speech")
    p.add_argument("--pad-after",  type=float, default=1.0, metavar="SECONDS",
                   help="Seconds of silence to add AFTER the speech")
    p.add_argument("--gain", type=float, default=2.0, metavar="FACTOR",
                   help="Volume multiplier applied to the speech  (2.0 = 2×)")
    p.add_argument("--no-clip", action="store_true",
                   help="Use soft (tanh) clipping instead of hard clamp to ±1")
    p.add_argument("--sr", type=int, default=16_000, metavar="HZ",
                   help="Target sample-rate for loading / saving")
    return p


def main() -> None:
    args = build_parser().parse_args()

    # ── default output path ─────────────────────────────────────────────────
    if args.output is None:
        base:      str = os.path.basename(str(args.input))
        directory: str = os.path.dirname(str(args.input)) or "."
        args.output    = os.path.join(directory, f"improved_{base}")

    # ── load ────────────────────────────────────────────────────────────────
    print(f"📂  Loading  : {args.input}")
    wav, sr = load_audio(args.input, target_sr=args.sr)
    original_duration = wav.shape[-1] / sr
    print(f"   Sample-rate : {sr} Hz")
    print(f"   Duration    : {original_duration:.3f} s  ({wav.shape[-1]} samples)")

    # ── amplify ─────────────────────────────────────────────────────────────
    print(f"🔊  Amplifying: ×{args.gain}  (soft-clip={args.no_clip})")
    wav = amplify(wav, gain=args.gain, soft_clip=args.no_clip)

    # ── pad silence ─────────────────────────────────────────────────────────
    print(f"🕐  Padding   : {args.pad_before} s before  |  {args.pad_after} s after")
    wav = add_silence(wav, sr, before=args.pad_before, after=args.pad_after)
    new_duration = wav.shape[-1] / sr
    print(f"   New duration: {new_duration:.3f} s  ({wav.shape[-1]} samples)")

    # ── save ─────────────────────────────────────────────────────────────────
    save_audio(str(args.output), wav, sr)
    print(f"✅  Saved     : {args.output}")


if __name__ == "__main__":
    main()

