"""
VAD Dataset Builder
====================
Mixes clean speech files with noise files at multiple SNR levels,
then generates per-frame binary VAD labels.

- Searches recursively for .wav / .WAV / .flac / .FLAC files
- Each speech file is mixed with EVERY noise file (full cross-product)
- Labels are generated from clean speech (SNR-independent ground truth)

Usage:
    python build_vad_dataset.py \
        --speech_dir  /path/to/clean_speech \
        --noise_dir   /path/to/noise \
        --output_dir  /path/to/output \
        --snr_levels  -5 0 5 10 15 20 \
        --frame_ms    20 \
        --sample_rate 16000 \
        --split       0.8 0.1 0.1

Install deps first:
    pip install librosa soundfile numpy tqdm
"""

import json
import random
import argparse
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1. File discovery  (case-insensitive, multi-format)
# ─────────────────────────────────────────────

AUDIO_EXTENSIONS = {".wav", ".flac"}


def find_audio_files(root_dir: str) -> list:
    """
    Recursively find all audio files under root_dir.
    Handles mixed case extensions (.wav / .WAV / .flac / .FLAC).
    Supports any depth of sub-directories.
    """
    root = Path(root_dir).resolve()

    if not root.exists():
        raise FileNotFoundError(f"Directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {root}")

    found = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            found.append(path)

    found = sorted(found)

    if not found:
        all_files = list(root.rglob("*"))
        extensions = {p.suffix.lower() for p in all_files if p.is_file()}
        raise FileNotFoundError(
            f"No audio files (.wav / .flac) found under: {root}\n"
            f"  Extensions found: {extensions or 'none'}\n"
            f"  Total items scanned: {len(all_files)}"
        )

    print(f"  Found {len(found)} audio files in: {root}")
    return found


# ─────────────────────────────────────────────
# 2. Audio loading & preprocessing
# ─────────────────────────────────────────────

def load_audio(path, sr: int = 16000) -> np.ndarray:
    """Load audio file, resample to sr Hz, convert to mono, peak-normalize."""
    audio, _ = librosa.load(str(path), sr=sr, mono=True)
    peak = np.max(np.abs(audio))
    if peak > 1e-9:
        audio = audio / peak
    return audio.astype(np.float32)


def loop_or_trim(audio: np.ndarray, target_len: int) -> np.ndarray:
    """Loop audio until it reaches target_len, then trim to exact length."""
    if len(audio) == 0:
        return np.zeros(target_len, dtype=np.float32)
    if len(audio) >= target_len:
        return audio[:target_len]
    reps = int(np.ceil(target_len / len(audio)))
    return np.tile(audio, reps)[:target_len]


# ─────────────────────────────────────────────
# 3. Mixing at target SNR
# ─────────────────────────────────────────────

def mix_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Mix speech + noise at a given SNR in dB.
    Noise is looped/trimmed to match speech length before mixing.
    Returns a clipped float32 mixture.
    """
    noise = loop_or_trim(noise, len(speech))

    speech_rms = np.sqrt(np.mean(speech ** 2) + 1e-9)
    noise_rms  = np.sqrt(np.mean(noise  ** 2) + 1e-9)

    target_noise_rms = speech_rms / (10 ** (snr_db / 20.0))
    noise_scaled = noise * (target_noise_rms / noise_rms)

    mixed = speech + noise_scaled
    mixed = np.clip(mixed, -1.0, 1.0)
    return mixed.astype(np.float32)


# ─────────────────────────────────────────────
# 4. VAD label generation
# ─────────────────────────────────────────────

def compute_vad_labels(
    speech: np.ndarray,
    sr: int = 16000,
    frame_ms: int = 20,
    energy_threshold: float = 0.01,
) -> list:
    """
    Generate per-frame binary VAD labels from the CLEAN speech signal.

    Labels are derived from clean speech (not the mixture) so that
    ground truth is independent of the noise level used for mixing.

    Args:
        speech:           Clean speech waveform (float32, normalized).
        sr:               Sample rate in Hz.
        frame_ms:         Frame duration in milliseconds (default 20 ms).
        energy_threshold: RMS threshold above which a frame is labeled 1.

    Returns:
        List of int (0 or 1), one entry per frame.
    """
    frame_size = int(sr * frame_ms / 1000)
    labels = []

    for start in range(0, len(speech), frame_size):
        frame = speech[start : start + frame_size]
        if len(frame) == 0:
            break
        rms = np.sqrt(np.mean(frame ** 2))
        labels.append(1 if rms > energy_threshold else 0)

    return labels


def compute_timestamps(labels: list, frame_ms: int = 20) -> list:
    """
    Convert a binary label list into speech segment timestamps (seconds).
    Used for RTTM / TextGrid export.
    """
    segments = []
    in_speech = False
    seg_start = 0.0

    for i, label in enumerate(labels):
        t = i * frame_ms / 1000.0
        if label == 1 and not in_speech:
            seg_start = t
            in_speech = True
        elif label == 0 and in_speech:
            segments.append({"start": round(seg_start, 4), "end": round(t, 4)})
            in_speech = False

    if in_speech:
        segments.append({
            "start": round(seg_start, 4),
            "end":   round(len(labels) * frame_ms / 1000.0, 4),
        })

    return segments


# ─────────────────────────────────────────────
# 5. Label export helpers
# ─────────────────────────────────────────────

def save_json(labels: list, segments: list, path, meta: dict):
    """Save JSON label file with metadata, per-frame labels, and segment list."""
    data = {
        "meta": meta,
        "frame_labels": labels,
        "speech_segments": segments,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def save_rttm(segments: list, path, speaker: str = "SPK"):
    """Save RTTM format label file (compatible with pyannote / kaldi)."""
    stem = Path(path).stem
    with open(path, "w") as f:
        for seg in segments:
            dur = round(seg["end"] - seg["start"], 4)
            f.write(
                f"SPEAKER {stem} 1 {seg['start']:.4f} {dur:.4f} "
                f"<NA> <NA> {speaker} <NA> <NA>\n"
            )


# ─────────────────────────────────────────────
# 6. Train / val / test split
# ─────────────────────────────────────────────

def split_files(files: list, ratios: tuple, seed: int = 42) -> tuple:
    """Shuffle and split a file list into train / val / test subsets."""
    random.seed(seed)
    shuffled = files.copy()
    random.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * ratios[0])
    n_val   = int(n * ratios[1])
    return (
        shuffled[:n_train],
        shuffled[n_train : n_train + n_val],
        shuffled[n_train + n_val :],
    )


# ─────────────────────────────────────────────
# 7. Main pipeline
# ─────────────────────────────────────────────

def build_dataset(args):
    print("\n=== VAD Dataset Builder ===\n")

    # ── Step 1: Discover files ──────────────────────────────────────────
    print("[1/4] Discovering files...")
    speech_files = find_audio_files(args.speech_dir)
    noise_files  = find_audio_files(args.noise_dir)

    total_combinations = len(speech_files) * len(noise_files) * len(args.snr_levels)
    print(f"\n  Speech files : {len(speech_files)}")
    print(f"  Noise files  : {len(noise_files)}")
    print(f"  SNR levels   : {args.snr_levels}")
    print(f"  Total output files will be: {total_combinations}\n")

    # ── Step 2: Pre-load all noise files into memory ────────────────────
    print("[2/4] Loading noise files...")
    # dict: noise_path_str -> np.ndarray
    noise_audios = {}
    for noise_path in tqdm(noise_files, desc="noise", unit="file"):
        noise_audios[str(noise_path)] = load_audio(noise_path, args.sample_rate)

    # ── Step 3: Split speech files ─────────────────────────────────────
    print("\n[3/4] Splitting speech files into train / val / test...")
    train_files, val_files, test_files = split_files(
        speech_files,
        ratios=(args.split[0], args.split[1], args.split[2]),
    )
    splits = {"train": train_files, "val": val_files, "test": test_files}
    for name, lst in splits.items():
        n_out = len(lst) * len(noise_files) * len(args.snr_levels)
        print(f"  {name}: {len(lst)} speech files  ->  {n_out} mixed files")

    # ── Step 4: Mix every speech x every noise x every SNR ─────────────
    print("\n[4/4] Mixing and labeling...\n")
    output_root = Path(args.output_dir)
    stats = {"total_files": 0, "total_duration_s": 0.0, "speech_ratio": []}

    for split_name, files in splits.items():

        audio_dir = output_root / split_name / "audio"
        label_dir = output_root / split_name / "labels"
        audio_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        for speech_path in tqdm(files, desc=split_name, unit="speech"):

            # Load speech once — reused for all noise files and SNR levels
            speech = load_audio(speech_path, args.sample_rate)

            # Compute VAD labels ONCE from clean speech
            # (same labels reused for every noise/SNR combination)
            labels   = compute_vad_labels(
                speech,
                sr=args.sample_rate,
                frame_ms=args.frame_ms,
                energy_threshold=args.energy_threshold,
            )
            segments = compute_timestamps(labels, args.frame_ms)

            duration_s   = round(len(speech) / args.sample_rate, 4)
            speech_ratio = round(sum(labels) / max(len(labels), 1), 4)

            # Cross-product: every noise file x every SNR level
            for noise_path_str, noise_audio in noise_audios.items():
                noise_stem = Path(noise_path_str).stem

                # Random offset into noise file for variety
                max_start  = max(0, len(noise_audio) - len(speech))
                noise_start = random.randint(0, max_start) if max_start > 0 else 0
                noise_segment = noise_audio[noise_start : noise_start + len(speech)]

                for snr in args.snr_levels:
                    mixed = mix_snr(speech, noise_segment, snr_db=snr)

                    # Filename: {speech_stem}__{noise_stem}__snrXXdB
                    snr_tag = f"snr{snr:+.0f}dB".replace("+", "p").replace("-", "m")
                    stem = f"{speech_path.stem}__{noise_stem}__{snr_tag}"

                    # Save mixed WAV
                    wav_path = audio_dir / f"{stem}.wav"
                    sf.write(str(wav_path), mixed, args.sample_rate, subtype="PCM_16")

                    # Save JSON labels
                    meta = {
                        "source_speech"      : str(speech_path),
                        "source_noise"       : noise_path_str,
                        "snr_db"             : snr,
                        "sample_rate"        : args.sample_rate,
                        "frame_ms"           : args.frame_ms,
                        "num_frames"         : len(labels),
                        "duration_s"         : duration_s,
                        "speech_frame_ratio" : speech_ratio,
                    }
                    json_path = label_dir / f"{stem}.json"
                    save_json(labels, segments, json_path, meta)

                    # Save RTTM labels (optional, for pyannote / kaldi)
                    rttm_path = label_dir / f"{stem}.rttm"
                    save_rttm(segments, rttm_path)

                    stats["total_files"]      += 1
                    stats["total_duration_s"] += duration_s
                    stats["speech_ratio"].append(speech_ratio)

    # ── Summary ────────────────────────────────────────────────────────
    print("\n=== Done ===")
    print(f"  Total mixed files : {stats['total_files']}")
    print(f"  Total audio       : {stats['total_duration_s'] / 3600:.2f} hours")
    print(f"  Avg speech ratio  : {np.mean(stats['speech_ratio']):.2%}")
    print(f"  Output directory  : {output_root.resolve()}\n")

    manifest = {
        "sample_rate"     : args.sample_rate,
        "frame_ms"        : args.frame_ms,
        "snr_levels"      : args.snr_levels,
        "num_noise_files" : len(noise_files),
        "splits"          : {k: len(v) for k, v in splits.items()},
        "total_files"     : stats["total_files"],
        "total_hours"     : round(stats["total_duration_s"] / 3600, 3),
    }
    manifest_path = output_root / "dataset_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest saved : {manifest_path}")


# ─────────────────────────────────────────────
# 8. CLI
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a VAD dataset by mixing clean speech with noise files.\n"
            "Each speech file is mixed with every noise file at every SNR level."
        )
    )
    parser.add_argument(
        "--speech_dir", required=True,
        help="Root directory of clean speech files (WAV/FLAC, searched recursively)."
    )
    parser.add_argument(
        "--noise_dir", required=True,
        help="Root directory of noise files (WAV/FLAC, searched recursively)."
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Root output directory for the dataset."
    )
    parser.add_argument(
        "--snr_levels", nargs="+", type=float,
        default=[-5, 0, 5, 10, 15, 20],
        help="SNR levels in dB (default: -5 0 5 10 15 20)."
    )
    parser.add_argument(
        "--frame_ms", type=int, default=20,
        help="VAD frame size in milliseconds (default: 20)."
    )
    parser.add_argument(
        "--sample_rate", type=int, default=16000,
        help="Target sample rate Hz (default: 16000)."
    )
    parser.add_argument(
        "--energy_threshold", type=float, default=0.01,
        help="RMS energy threshold for speech frame detection (default: 0.01)."
    )
    parser.add_argument(
        "--split", nargs=3, type=float, default=[0.8, 0.1, 0.1],
        metavar=("TRAIN", "VAL", "TEST"),
        help="Train/val/test split ratios (default: 0.8 0.1 0.1)."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if abs(sum(args.split) - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {sum(args.split):.4f}")

    build_dataset(args)