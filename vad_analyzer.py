#!/usr/bin/env python3
"""
VAD 4-Panel Evaluation – NOIZEUS Dataset
=========================================

Panels (per file):
  1 – Clean waveform (blue)
  2 – Noisy waveform (red)
  3 – VAD binary detection (0=noise, 1=speech) – step plot
  4 – Ground-Truth VAD binary (0=noise, 1=speech) – step plot

Ground truth is derived from the **clean audio energy envelope**
(not from running the neural VAD on the clean file).

Also produces:
  - Aggregated confusion matrix PNG
  - Per-SNR summary table (printed + saved as CSV)
  - When --model both: comparison confusion matrix for both models

Usage:
    python test2.py \
        --input_dir  /path/to/noizeus_dataset \
        [--output_dir ./cm_output] \
        [--device cpu] \
        [--model speechbrain|silero|both] \
        [--silero_model_path ./silero-vad/src/silero_vad/data/silero_vad.onnx] \
        [--activation_th   0.5] \
        [--deactivation_th 0.25] \
        [--close_th        0.250] \
        [--len_th          0.250] \
        [--energy_th_db   -40]
"""

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import torch
from hyperpyyaml import load_hyperpyyaml

from speechbrain.dataio import audio_io
from speechbrain.utils.checkpoints import Checkpointer
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)

SCRIPT_DIR   = Path(__file__).parent.resolve()
HPARAMS_FILE = SCRIPT_DIR / "hparams" / "train.yaml"
HFC_DIR      = SCRIPT_DIR / "hfc"
SAVE_DIR     = SCRIPT_DIR / "CRDNN_VAD" / "save"
SILERO_ONNX_DEFAULT = SCRIPT_DIR / "silero-vad" / "src" / "silero_vad" / "data" / "silero_vad.onnx"
BOUNDARY_MARKER = 1.0

RE_CLEAN = re.compile(r"^(sp\d+)\.wav$",               re.IGNORECASE)
RE_NOISY = re.compile(r"^(sp\d+)_(.+?)_sn(\d+)\.wav$", re.IGNORECASE)

# Colours
C_CLEAN      = "#1565C0"
C_NOISY      = "#C62828"
C_SPEECH     = "#43A047"
C_NOISE_FILL = "#E53935"
C_GT_SPEECH  = "#1B5E20"
C_GT_NOISE   = "#B71C1C"
C_BG         = "#F5F5F5"


# ────────────────────────────────────────────────────────────────────────────
# Dataset discovery
# ────────────────────────────────────────────────────────────────────────────
def discover_dataset(input_dir: Path) -> Dict[str, Dict]:
    raw: Dict = defaultdict(lambda: {"clean": None, "noisy": defaultdict(dict)})
    for f in sorted(input_dir.iterdir()):
        if not f.is_file():
            continue
        m = RE_CLEAN.match(f.name)
        if m:
            raw[m.group(1).lower()]["clean"] = f
            continue
        m = RE_NOISY.match(f.name)
        if m:
            spk, noise, snr = m.group(1).lower(), m.group(2).lower(), int(m.group(3))
            raw[spk]["noisy"][noise][snr] = f
    valid = {}
    for spk, data in sorted(raw.items()):
        if data["clean"] is None or not data["noisy"]:
            continue
        valid[spk] = {"clean": data["clean"], "noisy": dict(data["noisy"])}
    return valid


# ────────────────────────────────────────────────────────────────────────────
# Ground-truth VAD from clean audio energy
# ────────────────────────────────────────────────────────────────────────────
def energy_vad_gt(
    wav_path: Path,
    frame_dur: float = 0.01,
    energy_th_db: float = -55.0,
    merge_gap: float = 0.10,
    min_dur: float = 0.05,
    min_silence: float = 0.15,
    relative_th_db: float = 35.0,
) -> Tuple[torch.Tensor, float, np.ndarray, int]:
    """
    Energy-based ground-truth VAD on a clean audio file.

    Uses an **adaptive threshold**: the higher of ``energy_th_db`` and
    ``peak_rms_dB - relative_th_db``.  This handles quiet recordings
    (like NOIZEUS) where a fixed -40 dB misses most speech.

    Returns
    -------
    boundaries : (N, 2) tensor of [start_s, end_s]
    total_dur  : total duration in seconds
    frame_labels : bool array per frame (True=speech)
    sr : sample rate
    """
    wav, sr = audio_io.load(str(wav_path))
    wav = wav.float()
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    mono = wav.mean(dim=0)                       # (num_samples,)
    total_dur = mono.shape[0] / sr

    # Frame-level RMS energy
    frame_len = int(sr * frame_dur)
    n_frames = mono.shape[0] // frame_len
    frames = mono[:n_frames * frame_len].reshape(n_frames, frame_len)
    rms = torch.sqrt((frames ** 2).mean(dim=1) + 1e-12)
    rms_db = 20 * torch.log10(rms + 1e-12)

    # Adaptive threshold: peak - relative_th_db, but no lower than energy_th_db
    peak_db = float(rms_db.max())
    adaptive_th = max(energy_th_db, peak_db - relative_th_db)
    logger.debug(f"  GT energy: peak={peak_db:.1f} dB, adaptive_th={adaptive_th:.1f} dB")

    # Binary labels
    speech_mask = (rms_db >= adaptive_th).numpy()

    # Morphological smoothing: close small gaps, remove short bursts
    gap_frames = int(merge_gap / frame_dur)
    min_frames = int(min_dur / frame_dur)
    min_sil_frames = int(min_silence / frame_dur)

    labels = speech_mask.copy()

    # Pass 1: Close short gaps (merge nearby speech through brief silences)
    in_gap = 0
    for i in range(len(labels)):
        if not labels[i]:
            in_gap += 1
        else:
            if 0 < in_gap <= gap_frames:
                labels[i - in_gap:i] = True
            in_gap = 0

    # Pass 2: Remove short speech bursts (spurious detections)
    seg_start = None
    for i in range(len(labels)):
        if labels[i] and seg_start is None:
            seg_start = i
        elif not labels[i] and seg_start is not None:
            if (i - seg_start) < min_frames:
                labels[seg_start:i] = False
            seg_start = None
    if seg_start is not None and (len(labels) - seg_start) < min_frames:
        labels[seg_start:] = False

    # Pass 3: Remove short silence gaps that survived pass 1
    # (ensures we don't fragment speech at syllable boundaries)
    sil_start = None
    for i in range(len(labels)):
        if not labels[i] and sil_start is None:
            sil_start = i
        elif labels[i] and sil_start is not None:
            if (i - sil_start) < min_sil_frames:
                labels[sil_start:i] = True
            sil_start = None

    # Convert to boundaries
    boundaries = []
    in_speech = False
    start = 0
    for i in range(len(labels)):
        if labels[i] and not in_speech:
            start = i
            in_speech = True
        elif not labels[i] and in_speech:
            boundaries.append([start * frame_dur, i * frame_dur])
            in_speech = False
    if in_speech:
        boundaries.append([start * frame_dur, len(labels) * frame_dur])

    if boundaries:
        b_tensor = torch.FloatTensor(boundaries)
    else:
        b_tensor = torch.zeros(0, 2)

    return b_tensor, total_dur, labels, int(sr)


# ────────────────────────────────────────────────────────────────────────────
# Model loading
# ────────────────────────────────────────────────────────────────────────────
def load_vad_model(device: str = "cpu", ckpt_tag: str = "hfc") -> Dict:
    """Load SpeechBrain CRDNN VAD model.

    Parameters
    ----------
    ckpt_tag : str
        ``"hfc"``        – official HuggingFace pretrained model (hfc/ dir)
        ``"epoch_91"``   – local checkpoint CKPT+epoch_91
        ``"epoch_100"``  – local checkpoint CKPT+epoch_100
    """
    if ckpt_tag == "hfc":
        # ── Load from the HuggingFace-style pretrained dir ──
        hparams_file = HFC_DIR / "hyperparams.yaml"
        with open(hparams_file, encoding="utf-8") as fh:
            hparams = load_hyperpyyaml(fh)

        model         = hparams["model"]
        mean_var_norm = hparams["mean_var_norm"]

        # Use Pretrainer to load model.ckpt + normalizer.ckpt
        from speechbrain.utils.parameter_transfer import Pretrainer
        pretrainer = Pretrainer(
            loadables={"model": model, "mean_var_norm": mean_var_norm},
        )
        pretrainer.collect_files(default_source=str(HFC_DIR))
        pretrainer.load_collected()
        logger.info(f"Loaded HuggingFace pretrained model from {HFC_DIR}")
    else:
        # ── Load from a specific local checkpoint ──
        overrides = {
            "data_folder":           "/tmp",
            "musan_folder":          "/tmp",
            "commonlanguage_folder": "/tmp",
            "output_folder":         "/tmp",
        }
        with open(HPARAMS_FILE, encoding="utf-8") as fh:
            hparams = load_hyperpyyaml(fh, overrides)

        model         = hparams["model"]
        mean_var_norm = hparams["mean_var_norm"]

        ckpt_dir = SAVE_DIR / f"CKPT+{ckpt_tag}"
        if not ckpt_dir.exists():
            raise RuntimeError(f"Checkpoint dir not found: {ckpt_dir}")

        # Load specific checkpoint by filtering with ckpt_predicate
        ckpt = Checkpointer(
            checkpoints_dir=str(SAVE_DIR),
            recoverables={"model": model, "normalizer": mean_var_norm},
        )
        chosen = ckpt.recover_if_possible(
            ckpt_predicate=lambda c: str(ckpt_dir) in str(c.path),
        )
        if chosen is None:
            raise RuntimeError(f"Could not recover checkpoint from {ckpt_dir}")
        logger.info(f"Loaded local checkpoint: {chosen.path}")

    model.eval().to(device)
    mean_var_norm.to(device)
    hparams["compute_features"].to(device)

    return {
        "compute_features": hparams["compute_features"],
        "mean_var_norm":    mean_var_norm,
        "cnn":              hparams["cnn"],
        "rnn":              hparams["rnn"],
        "dnn":              hparams["dnn"],
        "sample_rate":      int(hparams["sample_rate"]),
        "time_resolution":  float(hparams["time_resolution"]),
    }


# ────────────────────────────────────────────────────────────────────────────
# VAD inference (SpeechBrain neural model)
# ────────────────────────────────────────────────────────────────────────────
def _speech_prob_chunk(mods, wavs: torch.Tensor, device: str) -> torch.Tensor:
    if wavs.dim() == 1:
        wavs = wavs.unsqueeze(0)
    wav_lens = torch.ones(wavs.shape[0], device=device)
    wavs  = wavs.float().to(device)
    feats = mods["compute_features"](wavs)
    feats = mods["mean_var_norm"](feats, wav_lens)
    out   = mods["cnn"](feats)
    out   = out.reshape(out.shape[0], out.shape[1], out.shape[2] * out.shape[3])
    out, _ = mods["rnn"](out)
    return torch.sigmoid(mods["dnn"](out))


def speech_prob_file(mods, path: Path, device: str) -> torch.Tensor:
    sr = mods["sample_rate"]; tr = mods["time_resolution"]
    large = 30.0; small = 10.0
    meta  = audio_io.info(str(path))
    total = meta.num_frames
    lc = int(sr * large); sc = int(sr * small)

    chunks = []; begin = 0
    while True:
        last   = (begin + lc >= total)
        wav, _ = audio_io.load(str(path), frame_offset=begin, num_frames=lc)
        wav    = wav.to(device)
        if last or wav.shape[-1] < sc:
            wav = torch.cat([wav, torch.zeros(1, sc, device=device)], dim=1)

        segs = torch.nn.functional.unfold(
            wav.unsqueeze(1).unsqueeze(2),
            kernel_size=(1, sc), stride=(1, sc),
        ).squeeze(0).transpose(0, 1)

        prob    = _speech_prob_chunk(mods, segs, device)[:, :-1, :]
        prob    = prob.permute(2, 1, 0)
        out_len = int(wav.shape[-1] / (sr * tr))
        kl      = int(small / tr)
        prob = torch.nn.functional.fold(
            prob, output_size=(1, out_len), kernel_size=(1, kl), stride=(1, kl),
        ).squeeze(1).transpose(-1, -2)
        chunks.append(prob)
        if last:
            break
        begin += lc

    prob_vad  = torch.cat(chunks, dim=1)
    last_elem = int(total / (tr * sr))
    return prob_vad[:, :last_elem, :]


def apply_threshold(prob: torch.Tensor, act: float, deact: float) -> torch.Tensor:
    no_deact = (prob >= deact).to("cpu")
    th       = (prob >= act).to("cpu")
    for i in range(1, prob.shape[1]):
        th[:, i, ...] |= th[:, i - 1, ...]
        th[:, i, ...]  &= no_deact[:, i, ...]
    return th.to(prob.device)


def get_boundaries(th: torch.Tensor, tr: float, sr: int) -> torch.Tensor:
    shifted          = torch.roll(th, dims=1, shifts=1)
    shifted[:, 0, :] = 0
    th               = th + shifted
    th[:, 0,  :]     = (th[:, 0,  :] >= BOUNDARY_MARKER).int()
    th[:, -1, :]     = (th[:, -1, :] >= BOUNDARY_MARKER).int()
    if (th == BOUNDARY_MARKER).nonzero().shape[0] % 2 == 1:
        th = torch.cat(
            (th, torch.tensor([[[BOUNDARY_MARKER]]]).to(th.device)), dim=1
        )
    idx = (th == BOUNDARY_MARKER).nonzero()[:, 1].reshape(-1, 2)
    beg = (idx[:, 0] * tr).float()
    end = (idx[:, 1] * tr).float()
    return torch.stack([beg, end], dim=1)


def merge_close(b: torch.Tensor, th: float) -> torch.Tensor:
    if b.shape[0] == 0:
        return b
    out = []; pb, pe = b[0, 0].float(), b[0, 1].float()
    for i in range(1, b.shape[0]):
        if b[i, 0] - pe <= th:
            pe = b[i, 1]
        else:
            out.append([pb, pe]); pb, pe = b[i, 0], b[i, 1]
    out.append([pb, pe])
    return torch.FloatTensor(out).to(b.device)


def remove_short(b: torch.Tensor, th: float) -> torch.Tensor:
    kept = [
        [b[i, 0], b[i, 1]]
        for i in range(b.shape[0])
        if (b[i, 1] - b[i, 0]) > th
    ]
    return torch.FloatTensor(kept).to(b.device) if kept else torch.zeros(0, 2)


def run_vad(mods, path, act, deact, close, lth, device):
    sr = mods["sample_rate"]; tr = mods["time_resolution"]
    with torch.no_grad():
        prob = speech_prob_file(mods, path, device)
    th = apply_threshold(prob, act, deact)
    b  = get_boundaries(th, tr, sr)
    if b.shape[0] > 0:
        b = merge_close(b, close)
        b = remove_short(b, lth)
    info = audio_io.info(str(path))
    return b, info.num_frames / info.sample_rate


# ────────────────────────────────────────────────────────────────────────────
# Silero VAD (ONNX) loading & inference
# ────────────────────────────────────────────────────────────────────────────
def load_silero_model(model_path: str) -> "OnnxWrapper":
    """Load the Silero VAD ONNX model."""
    silero_src = SCRIPT_DIR / "silero-vad" / "src"
    if str(silero_src) not in sys.path:
        sys.path.insert(0, str(silero_src))
    from silero_vad.utils_vad import OnnxWrapper
    model = OnnxWrapper(str(model_path), force_onnx_cpu=True)
    logger.info(f"Silero VAD model loaded from {model_path}")
    return model


def run_silero_vad(
    model, path: Path, act: float, deact: float, close: float, lth: float,
) -> Tuple[torch.Tensor, float]:
    """
    Run Silero VAD on an audio file. Returns boundaries and duration.
    Uses chunk-based processing at 16 kHz with 512-sample windows.
    """
    import torchaudio

    wav, sr = torchaudio.load(str(path))
    if wav.ndim > 1 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
        sr = 16000
    wav = wav.squeeze(0)  # (num_samples,)
    total_dur = wav.shape[0] / sr

    # Get per-chunk probabilities
    model.reset_states()
    num_samples = 512  # for 16kHz
    # Pad to multiple of num_samples
    if wav.shape[0] % num_samples:
        pad_num = num_samples - (wav.shape[0] % num_samples)
        wav = torch.nn.functional.pad(wav, (0, pad_num), 'constant', value=0.0)

    probs = []
    for i in range(0, wav.shape[0], num_samples):
        chunk = wav[i:i + num_samples].unsqueeze(0)
        out = model(chunk, sr)
        probs.append(out.item() if out.numel() == 1 else out.squeeze().item())

    # Each prob covers num_samples/sr seconds = 0.032s
    chunk_dur = num_samples / sr  # 0.032s per chunk

    # Apply activation/deactivation thresholds (hysteresis)
    triggered = False
    speech_frames = []
    for p in probs:
        if not triggered:
            if p >= act:
                triggered = True
                speech_frames.append(True)
            else:
                speech_frames.append(False)
        else:
            if p < deact:
                triggered = False
                speech_frames.append(False)
            else:
                speech_frames.append(True)

    # Convert to boundaries
    boundaries = []
    in_speech = False
    start = 0.0
    for i, is_speech in enumerate(speech_frames):
        t = i * chunk_dur
        if is_speech and not in_speech:
            start = t
            in_speech = True
        elif not is_speech and in_speech:
            boundaries.append([start, t])
            in_speech = False
    if in_speech:
        boundaries.append([start, len(speech_frames) * chunk_dur])

    if boundaries:
        b = torch.FloatTensor(boundaries)
        # Merge close and remove short
        b = merge_close(b, close)
        b = remove_short(b, lth)
    else:
        b = torch.zeros(0, 2)

    return b, total_dur


# ────────────────────────────────────────────────────────────────────────────
# Frame labels & confusion metrics
# ────────────────────────────────────────────────────────────────────────────
def boundaries_to_frame_labels(boundaries: torch.Tensor, n: int, tr: float) -> np.ndarray:
    labels = np.zeros(n, dtype=bool)
    for i in range(boundaries.shape[0]):
        s = max(0, round(boundaries[i, 0].item() / tr))
        e = min(n, round(boundaries[i, 1].item() / tr))
        labels[s:e] = True
    return labels


def compute_confusion(gt: np.ndarray, pred: np.ndarray, tr: float) -> Dict[str, float]:
    n = min(len(gt), len(pred))
    gt, pred = gt[:n], pred[:n]
    return {
        "tp": int(np.sum( gt &  pred)) * tr,
        "tn": int(np.sum(~gt & ~pred)) * tr,
        "fp": int(np.sum(~gt &  pred)) * tr,
        "fn": int(np.sum( gt & ~pred)) * tr,
    }


def derived_metrics(cm: Dict[str, float]) -> Dict[str, float]:
    tp, tn, fp, fn = cm["tp"], cm["tn"], cm["fp"], cm["fn"]
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    acc  = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "accuracy": acc}


def duration_stats(boundaries: torch.Tensor, total_dur: float):
    if boundaries.shape[0] == 0:
        return 0.0, total_dur
    speech = float(sum(boundaries[i, 1] - boundaries[i, 0] for i in range(boundaries.shape[0])))
    return speech, max(0.0, total_dur - speech)


# ────────────────────────────────────────────────────────────────────────────
# Binary step signal from boundaries
# ────────────────────────────────────────────────────────────────────────────
def boundaries_to_step(boundaries: torch.Tensor, total_dur: float, sr: int = 1000):
    """Create (time_array, binary_array) for step-plot display."""
    n = int(total_dur * sr)
    t = np.linspace(0, total_dur, n)
    y = np.zeros(n, dtype=np.float32)
    for i in range(boundaries.shape[0]):
        s = int(boundaries[i, 0].item() * sr)
        e = int(boundaries[i, 1].item() * sr)
        y[max(0, s):min(n, e)] = 1.0
    return t, y


# ────────────────────────────────────────────────────────────────────────────
# Audio loading
# ────────────────────────────────────────────────────────────────────────────
def load_mono_np(path: Path):
    wav, sr = audio_io.load(str(path))
    wav = wav.float()
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    return wav.mean(dim=0).numpy(), int(sr)


# ────────────────────────────────────────────────────────────────────────────
# Segment annotation helper
# ────────────────────────────────────────────────────────────────────────────
def _annotate_segments(ax, boundaries, total_dur):
    """Add small text annotations for each speech/noise segment."""
    if boundaries.shape[0] == 0:
        return
    prev = 0.0
    for i in range(boundaries.shape[0]):
        s = float(boundaries[i, 0])
        e = float(boundaries[i, 1])
        if s - prev > 0.15:
            mid = (prev + s) / 2
            ax.text(mid, 0.5, f"{prev:.1f}–{s:.1f}s", ha="center", va="center",
                    fontsize=5.5, color=C_GT_NOISE, style="italic",
                    bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.5))
        mid = (s + e) / 2
        ax.text(mid, 0.5, f"{s:.1f}–{e:.1f}s", ha="center", va="center",
                fontsize=5.5, color=C_GT_SPEECH, fontweight="bold",
                bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.5))
        prev = e
    if total_dur - prev > 0.15:
        mid = (prev + total_dur) / 2
        ax.text(mid, 0.5, f"{prev:.1f}–{total_dur:.1f}s", ha="center", va="center",
                fontsize=5.5, color=C_GT_NOISE, style="italic",
                bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.5))


# ────────────────────────────────────────────────────────────────────────────
# 4-Panel Plot  (panels 3 & 4 are binary 0/1 step functions)
# ────────────────────────────────────────────────────────────────────────────
def plot_4panel(
    spk, noise, snr_level,
    clean_path, noisy_path,
    gt_b, pred_b,
    audio_dur, pred_dur,
    output_dir,
    model_name: str = "SpeechBrain",
) -> Path:
    clean_wav, c_sr = load_mono_np(clean_path)
    noisy_wav, n_sr = load_mono_np(noisy_path)

    t_clean = np.arange(len(clean_wav)) / c_sr
    t_noisy = np.arange(len(noisy_wav)) / n_sr
    x_max = max(t_clean[-1] if len(t_clean) else 0,
                t_noisy[-1] if len(t_noisy) else 0)

    fig, axes = plt.subplots(4, 1, figsize=(14, 9),
                             gridspec_kw={"height_ratios": [1, 1, 0.7, 0.7]})
    fig.patch.set_facecolor("white")
    plt.subplots_adjust(hspace=0.45, top=0.91, bottom=0.08, left=0.07, right=0.97)

    # ── Panel 1: Clean waveform ──
    ax = axes[0]
    ax.plot(t_clean, clean_wav, color=C_CLEAN, linewidth=0.4, alpha=0.9)
    ax.set_xlim(0, x_max); ax.set_ylim(-1.1, 1.1)
    ax.set_ylabel("Amplitude", fontsize=9)
    ax.set_title("Panel 1 – Clean Audio Signal", fontsize=10, fontweight="bold", pad=4)
    ax.set_facecolor(C_BG)
    ax.axhline(0, color="#999", lw=0.4)
    sp_d, si_d = duration_stats(gt_b, audio_dur)
    ax.text(0.99, 0.95, f"Duration: {audio_dur:.2f}s  |  GT speech: {sp_d:.2f}s  silence: {si_d:.2f}s",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color="#444")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)

    # ── Panel 2: Noisy waveform ──
    ax = axes[1]
    ax.plot(t_noisy, noisy_wav, color=C_NOISY, linewidth=0.4, alpha=0.9)
    ax.set_xlim(0, x_max); ax.set_ylim(-1.1, 1.1)
    ax.set_ylabel("Amplitude", fontsize=9)
    ax.set_title(f"Panel 2 – Noisy Audio ({noise.capitalize()} @ {snr_level} dB SNR)",
                 fontsize=10, fontweight="bold", pad=4)
    ax.set_facecolor(C_BG)
    ax.axhline(0, color="#999", lw=0.4)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)

    # ── Panel 3: SpeechBrain VAD detection (binary 0/1) ──
    ax = axes[2]
    t_pred, y_pred = boundaries_to_step(pred_b, pred_dur)
    ax.fill_between(t_pred, y_pred, step="mid", color=C_SPEECH, alpha=0.5, label="Speech")
    ax.fill_between(t_pred, 1 - y_pred, step="mid", color=C_NOISE_FILL, alpha=0.25)
    ax.plot(t_pred, y_pred, color=C_SPEECH, linewidth=1.2, drawstyle="steps-mid")
    ax.set_xlim(0, x_max); ax.set_ylim(-0.05, 1.15)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Noise", "Speech"], fontsize=8)
    ax.set_ylabel("VAD Output", fontsize=9)
    ax.set_title(f"Panel 3 – {model_name} VAD Detection (on noisy audio)",
                 fontsize=10, fontweight="bold", pad=4)
    ax.set_facecolor("white")
    ax.axhline(0.5, color="#CCC", lw=0.5, ls="--")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    _annotate_segments(ax, pred_b, pred_dur)

    # ── Panel 4: Ground-Truth VAD (binary 0/1) ──
    ax = axes[3]
    t_gt, y_gt = boundaries_to_step(gt_b, audio_dur)
    ax.fill_between(t_gt, y_gt, step="mid", color=C_GT_SPEECH, alpha=0.45, label="Speech")
    ax.fill_between(t_gt, 1 - y_gt, step="mid", color=C_GT_NOISE, alpha=0.15)
    ax.plot(t_gt, y_gt, color=C_GT_SPEECH, linewidth=1.2, drawstyle="steps-mid")
    ax.set_xlim(0, x_max); ax.set_ylim(-0.05, 1.15)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Noise", "Speech"], fontsize=8)
    ax.set_ylabel("VAD Output", fontsize=9)
    ax.set_title("Panel 4 – Ground-Truth VAD (energy-based from clean audio)",
                 fontsize=10, fontweight="bold", pad=4)
    ax.set_facecolor("white")
    ax.axhline(0.5, color="#CCC", lw=0.5, ls="--")
    ax.set_xlabel("Time [s]", fontsize=9)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    _annotate_segments(ax, gt_b, audio_dur)

    # ── Suptitle ──
    fig.suptitle(
        f"VAD Analysis  –  {spk.upper()}  |  {noise.capitalize()} noise  |  {snr_level} dB SNR",
        fontsize=12, fontweight="bold", y=0.97)

    # ── Legend ──
    legend_handles = [
        mpatches.Patch(color=C_SPEECH, alpha=0.55, label="Predicted: Speech"),
        mpatches.Patch(color=C_NOISE_FILL, alpha=0.35, label="Predicted: Noise"),
        mpatches.Patch(color=C_GT_SPEECH, alpha=0.55, label="GT: Speech"),
        mpatches.Patch(color=C_GT_NOISE, alpha=0.35, label="GT: Noise/Silence"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, fontsize=8.5,
               framealpha=0.9, bbox_to_anchor=(0.5, 0.002))

    out = output_dir / f"{spk}_{noise}_sn{snr_level}.png"
    plt.savefig(str(out), dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def plot_5panel(
    spk, noise, snr_level,
    clean_path, noisy_path,
    gt_b, pred_b_sb, pred_b_si,
    audio_dur, pred_dur_sb, pred_dur_si,
    output_dir,
) -> Path:
    """5-panel plot: clean, noisy, Silero VAD, SpeechBrain VAD, Ground Truth."""
    clean_wav, c_sr = load_mono_np(clean_path)
    noisy_wav, n_sr = load_mono_np(noisy_path)

    t_clean = np.arange(len(clean_wav)) / c_sr
    t_noisy = np.arange(len(noisy_wav)) / n_sr
    x_max = max(t_clean[-1] if len(t_clean) else 0,
                t_noisy[-1] if len(t_noisy) else 0)

    fig, axes = plt.subplots(5, 1, figsize=(14, 11),
                             gridspec_kw={"height_ratios": [1, 1, 0.7, 0.7, 0.7]})
    fig.patch.set_facecolor("white")
    plt.subplots_adjust(hspace=0.50, top=0.92, bottom=0.07, left=0.07, right=0.97)

    # ── Panel 1: Clean waveform ──
    ax = axes[0]
    ax.plot(t_clean, clean_wav, color=C_CLEAN, linewidth=0.4, alpha=0.9)
    ax.set_xlim(0, x_max); ax.set_ylim(-1.1, 1.1)
    ax.set_ylabel("Amplitude", fontsize=9)
    ax.set_title("Panel 1 – Clean Audio Signal", fontsize=10, fontweight="bold", pad=4)
    ax.set_facecolor(C_BG)
    ax.axhline(0, color="#999", lw=0.4)
    sp_d, si_d = duration_stats(gt_b, audio_dur)
    ax.text(0.99, 0.95, f"Duration: {audio_dur:.2f}s  |  GT speech: {sp_d:.2f}s  silence: {si_d:.2f}s",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color="#444")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)

    # ── Panel 2: Noisy waveform ──
    ax = axes[1]
    ax.plot(t_noisy, noisy_wav, color=C_NOISY, linewidth=0.4, alpha=0.9)
    ax.set_xlim(0, x_max); ax.set_ylim(-1.1, 1.1)
    ax.set_ylabel("Amplitude", fontsize=9)
    ax.set_title(f"Panel 2 – Noisy Audio ({noise.capitalize()} @ {snr_level} dB SNR)",
                 fontsize=10, fontweight="bold", pad=4)
    ax.set_facecolor(C_BG)
    ax.axhline(0, color="#999", lw=0.4)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)

    # ── Panel 3: Silero VAD detection ──
    ax = axes[2]
    t_pred, y_pred = boundaries_to_step(pred_b_si, pred_dur_si)
    ax.fill_between(t_pred, y_pred, step="mid", color="#FF6F00", alpha=0.5, label="Speech")
    ax.fill_between(t_pred, 1 - y_pred, step="mid", color=C_NOISE_FILL, alpha=0.25)
    ax.plot(t_pred, y_pred, color="#FF6F00", linewidth=1.2, drawstyle="steps-mid")
    ax.set_xlim(0, x_max); ax.set_ylim(-0.05, 1.15)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Noise", "Speech"], fontsize=8)
    ax.set_ylabel("VAD Output", fontsize=9)
    ax.set_title("Panel 3 – Silero VAD Detection (on noisy audio)",
                 fontsize=10, fontweight="bold", pad=4)
    ax.set_facecolor("white")
    ax.axhline(0.5, color="#CCC", lw=0.5, ls="--")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    _annotate_segments(ax, pred_b_si, pred_dur_si)

    # ── Panel 4: SpeechBrain VAD detection ──
    ax = axes[3]
    t_pred, y_pred = boundaries_to_step(pred_b_sb, pred_dur_sb)
    ax.fill_between(t_pred, y_pred, step="mid", color=C_SPEECH, alpha=0.5, label="Speech")
    ax.fill_between(t_pred, 1 - y_pred, step="mid", color=C_NOISE_FILL, alpha=0.25)
    ax.plot(t_pred, y_pred, color=C_SPEECH, linewidth=1.2, drawstyle="steps-mid")
    ax.set_xlim(0, x_max); ax.set_ylim(-0.05, 1.15)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Noise", "Speech"], fontsize=8)
    ax.set_ylabel("VAD Output", fontsize=9)
    ax.set_title("Panel 4 – SpeechBrain VAD Detection (on noisy audio)",
                 fontsize=10, fontweight="bold", pad=4)
    ax.set_facecolor("white")
    ax.axhline(0.5, color="#CCC", lw=0.5, ls="--")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    _annotate_segments(ax, pred_b_sb, pred_dur_sb)

    # ── Panel 5: Ground-Truth VAD ──
    ax = axes[4]
    t_gt, y_gt = boundaries_to_step(gt_b, audio_dur)
    ax.fill_between(t_gt, y_gt, step="mid", color=C_GT_SPEECH, alpha=0.45, label="Speech")
    ax.fill_between(t_gt, 1 - y_gt, step="mid", color=C_GT_NOISE, alpha=0.15)
    ax.plot(t_gt, y_gt, color=C_GT_SPEECH, linewidth=1.2, drawstyle="steps-mid")
    ax.set_xlim(0, x_max); ax.set_ylim(-0.05, 1.15)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Noise", "Speech"], fontsize=8)
    ax.set_ylabel("VAD Output", fontsize=9)
    ax.set_title("Panel 5 – Ground-Truth VAD (energy-based from clean audio)",
                 fontsize=10, fontweight="bold", pad=4)
    ax.set_facecolor("white")
    ax.axhline(0.5, color="#CCC", lw=0.5, ls="--")
    ax.set_xlabel("Time [s]", fontsize=9)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    _annotate_segments(ax, gt_b, audio_dur)

    # ── Suptitle ──
    fig.suptitle(
        f"VAD Comparison  –  {spk.upper()}  |  {noise.capitalize()} noise  |  {snr_level} dB SNR",
        fontsize=12, fontweight="bold", y=0.97)

    # ── Legend ──
    legend_handles = [
        mpatches.Patch(color="#FF6F00", alpha=0.55, label="Silero: Speech"),
        mpatches.Patch(color=C_SPEECH, alpha=0.55, label="SpeechBrain: Speech"),
        mpatches.Patch(color=C_NOISE_FILL, alpha=0.35, label="Predicted: Noise"),
        mpatches.Patch(color=C_GT_SPEECH, alpha=0.55, label="GT: Speech"),
        mpatches.Patch(color=C_GT_NOISE, alpha=0.35, label="GT: Noise/Silence"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=5, fontsize=8.5,
               framealpha=0.9, bbox_to_anchor=(0.5, 0.002))

    out = output_dir / f"{spk}_{noise}_sn{snr_level}.png"
    plt.savefig(str(out), dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ────────────────────────────────────────────────────────────────────────────
# Confusion matrix plot
# ────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(cm, n_speakers, n_noisy, output_dir, ckpt_label: str = "CRDNN") -> Path:
    tp, tn, fp, fn = cm["tp"], cm["tn"], cm["fp"], cm["fn"]
    total = tp + tn + fp + fn
    dm = derived_metrics(cm)

    mat = np.array([[tn, fp], [fn, tp]])
    pct = 100.0 * mat / total if total > 0 else mat

    cell_meta = [
        [("True Noise",    "TN", "#1565C0"), ("False Speech", "FP", "#E53935")],
        [("Missed Speech", "FN", "#FB8C00"), ("True Speech",  "TP", "#2E7D32")],
    ]

    fig, ax = plt.subplots(figsize=(7, 6.5))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    for r in range(2):
        for c in range(2):
            v, p = mat[r, c], pct[r, c]
            label, tag, col = cell_meta[r][c]
            alpha = 0.15 + 0.60 * (v / total if total > 0 else 0)
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                       color=col, alpha=alpha, zorder=1))
            ax.text(c, r + 0.28, tag, ha="center", va="center",
                    fontsize=16, fontweight="bold", color=col, zorder=2)
            ax.text(c, r + 0.06, label, ha="center", va="center",
                    fontsize=10, color="#222", zorder=2)
            ax.text(c, r - 0.16, f"{v:,.2f} s", ha="center", va="center",
                    fontsize=12, fontweight="bold", color="#111", zorder=2)
            ax.text(c, r - 0.36, f"({p:.1f}%)", ha="center", va="center",
                    fontsize=9, color="#555", zorder=2)

    ax.set_xlim(-0.5, 1.5); ax.set_ylim(-0.5, 1.5)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted: Noise", "Predicted: Speech"], fontsize=11)
    ax.set_yticklabels(["GT: Noise", "GT: Speech"], fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=12, labelpad=10)
    ax.set_ylabel("Ground-Truth Label", fontsize=12, labelpad=10)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)

    ax.set_title(
        f"SpeechBrain VAD – Aggregated Confusion Matrix\n"
        f"{ckpt_label}  |  NOIZEUS dataset\n"
        "(GT = energy-based from clean audio)",
        fontsize=12, fontweight="bold", pad=16)

    fig.text(0.5, 0.01,
             f"Accuracy={dm['accuracy']:.4f}   Precision={dm['precision']:.4f}   "
             f"Recall={dm['recall']:.4f}   F1={dm['f1']:.4f}\n"
             f"Speakers={n_speakers}   Files={n_noisy}   Total={total:,.1f}s",
             ha="center", va="bottom", fontsize=8.5, color="#444")

    plt.tight_layout(rect=[0, 0.07, 1, 1])
    out = output_dir / "vad_confusion_matrix.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


# ────────────────────────────────────────────────────────────────────────────
# Comparison confusion matrix plot (both models side-by-side)
# ────────────────────────────────────────────────────────────────────────────
def plot_comparison_confusion_matrix(
    cm_sb: Dict[str, float],
    cm_si: Dict[str, float],
    n_speakers: int,
    n_noisy: int,
    output_dir: Path,
) -> Path:
    """Side-by-side confusion matrices for SpeechBrain vs Silero."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5))
    fig.patch.set_facecolor("#F8F9FA")

    cell_meta = [
        [("True Noise",    "TN", "#1565C0"), ("False Speech", "FP", "#E53935")],
        [("Missed Speech", "FN", "#FB8C00"), ("True Speech",  "TP", "#2E7D32")],
    ]

    for ax, cm, title in [
        (ax1, cm_sb, "SpeechBrain VAD (CRDNN)"),
        (ax2, cm_si, "Silero VAD (ONNX)"),
    ]:
        tp, tn, fp, fn = cm["tp"], cm["tn"], cm["fp"], cm["fn"]
        total = tp + tn + fp + fn
        dm = derived_metrics(cm)
        mat = np.array([[tn, fp], [fn, tp]])
        pct = 100.0 * mat / total if total > 0 else mat

        ax.set_facecolor("#F8F9FA")
        for r in range(2):
            for c in range(2):
                v, p = mat[r, c], pct[r, c]
                label, tag, col = cell_meta[r][c]
                alpha = 0.15 + 0.60 * (v / total if total > 0 else 0)
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                           color=col, alpha=alpha, zorder=1))
                ax.text(c, r + 0.28, tag, ha="center", va="center",
                        fontsize=14, fontweight="bold", color=col, zorder=2)
                ax.text(c, r + 0.06, label, ha="center", va="center",
                        fontsize=9, color="#222", zorder=2)
                ax.text(c, r - 0.16, f"{v:,.2f} s", ha="center", va="center",
                        fontsize=11, fontweight="bold", color="#111", zorder=2)
                ax.text(c, r - 0.36, f"({p:.1f}%)", ha="center", va="center",
                        fontsize=8, color="#555", zorder=2)

        ax.set_xlim(-0.5, 1.5); ax.set_ylim(-0.5, 1.5)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred: Noise", "Pred: Speech"], fontsize=9)
        ax.set_yticklabels(["GT: Noise", "GT: Speech"], fontsize=9)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(
            f"{title}\n"
            f"Acc={dm['accuracy']:.4f}  P={dm['precision']:.4f}  "
            f"R={dm['recall']:.4f}  F1={dm['f1']:.4f}",
            fontsize=10, fontweight="bold", pad=12)

    fig.suptitle(
        "VAD Model Comparison – Aggregated Confusion Matrices\n"
        "NOIZEUS dataset  (GT = energy-based from clean audio)",
        fontsize=12, fontweight="bold", y=1.02)

    dm_sb = derived_metrics(cm_sb)
    dm_si = derived_metrics(cm_si)
    fig.text(0.5, -0.02,
             f"Speakers={n_speakers}   Files={n_noisy}\n"
             f"SpeechBrain F1={dm_sb['f1']:.4f}   Silero F1={dm_si['f1']:.4f}   "
             f"Δ F1={dm_sb['f1'] - dm_si['f1']:+.4f}",
             ha="center", va="top", fontsize=9, color="#444")

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    out = output_dir / "vad_comparison_confusion_matrix.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def parse_args():
    p = argparse.ArgumentParser(description="VAD evaluation with binary step plots + confusion matrix")
    p.add_argument("--input_dir",  required=True, type=Path)
    p.add_argument("--output_dir", default=Path("cm_output"), type=Path)
    p.add_argument("--device",     default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--model",      default="speechbrain",
                   choices=["speechbrain", "silero", "both"],
                   help="Which VAD model to use (default: speechbrain)")
    p.add_argument("--silero_model_path", type=Path,
                   default=SILERO_ONNX_DEFAULT,
                   help="Path to Silero VAD ONNX model file")
    p.add_argument("--ckpt", default="hfc",
                   choices=["hfc", "epoch_91", "epoch_100"],
                   help="Which SpeechBrain checkpoint to use: "
                        "hfc = HuggingFace pretrained, "
                        "epoch_91 = local CKPT+epoch_91, "
                        "epoch_100 = local CKPT+epoch_100 (default: hfc)")
    p.add_argument("--activation_th",   type=float, default=0.5)
    p.add_argument("--deactivation_th", type=float, default=0.25)
    p.add_argument("--close_th",        type=float, default=0.25)
    p.add_argument("--len_th",          type=float, default=0.25)
    p.add_argument("--energy_th_db",    type=float, default=-55.0,
                   help="Min energy threshold floor (dB) for GT VAD on clean audio")
    return p.parse_args()


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        logger.error(f"--input_dir does not exist: {input_dir}")
        sys.exit(1)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    use_sb = args.model in ("speechbrain", "both")
    use_si = args.model in ("silero", "both")

    # 1. Discover dataset
    logger.info(f"Scanning: {input_dir}")
    dataset = discover_dataset(input_dir)
    if not dataset:
        logger.error("No valid speaker pairs found."); sys.exit(1)

    n_speakers = len(dataset)
    total_noisy = sum(len(s) for d in dataset.values() for s in d["noisy"].values())
    logger.info(f"Found {n_speakers} speakers, {total_noisy} noisy files")

    # 2. Load model(s)
    mods_sb = None
    mods_si = None
    tr = 0.01  # default time resolution

    if use_sb:
        logger.info(f"Loading SpeechBrain VAD model (ckpt={args.ckpt}) …")
        mods_sb = load_vad_model(device=args.device, ckpt_tag=args.ckpt)
        tr = mods_sb["time_resolution"]
        logger.info("SpeechBrain model loaded.")

    if use_si:
        logger.info("Loading Silero VAD model …")
        if not args.silero_model_path.exists():
            logger.error(f"Silero model not found: {args.silero_model_path}")
            sys.exit(1)
        mods_si = load_silero_model(str(args.silero_model_path))
        if not use_sb:
            tr = 0.032  # Silero chunk duration at 16kHz with 512 samples
        logger.info("Silero model loaded.")

    # 3. Evaluation
    # Aggregated confusion matrices per model
    agg_sb = {"tp": 0.0, "tn": 0.0, "fp": 0.0, "fn": 0.0}
    agg_si = {"tp": 0.0, "tn": 0.0, "fp": 0.0, "fn": 0.0}
    per_snr_sb: Dict[int, Dict[str, float]] = defaultdict(lambda: {"tp": 0, "tn": 0, "fp": 0, "fn": 0})
    per_snr_si: Dict[int, Dict[str, float]] = defaultdict(lambda: {"tp": 0, "tn": 0, "fp": 0, "fn": 0})
    per_noise_sb: Dict[str, Dict[str, float]] = defaultdict(lambda: {"tp": 0, "tn": 0, "fp": 0, "fn": 0})
    per_noise_si: Dict[str, Dict[str, float]] = defaultdict(lambda: {"tp": 0, "tn": 0, "fp": 0, "fn": 0})
    n_processed = 0

    gt_tr = mods_sb["time_resolution"] if mods_sb else 0.01

    for spk_idx, (spk, data) in enumerate(dataset.items(), 1):
        clean_path = data["clean"]

        # ── Ground truth from clean audio energy ──
        try:
            gt_b, audio_dur, gt_labels, _ = energy_vad_gt(
                clean_path, frame_dur=gt_tr, energy_th_db=args.energy_th_db)
        except Exception as exc:
            logger.warning(f"GT failed for {clean_path.name}: {exc} – skipping speaker")
            continue

        sp_d, si_d = duration_stats(gt_b, audio_dur)
        logger.info(f"[{spk_idx}/{n_speakers}] {spk}  dur={audio_dur:.2f}s  "
                     f"speech={sp_d:.2f}s  silence={si_d:.2f}s")

        for noise, snr_files in sorted(data["noisy"].items()):
            for snr_level, noisy_path in sorted(snr_files.items()):
                # --- SpeechBrain ---
                pred_b_sb_ok, pred_b_si_ok = None, None
                pred_dur_sb, pred_dur_si = 0.0, 0.0

                if use_sb:
                    try:
                        sb_tr = mods_sb["time_resolution"]
                        pred_b_sb, pred_dur_sb = run_vad(
                            mods_sb, noisy_path,
                            args.activation_th, args.deactivation_th,
                            args.close_th, args.len_th, args.device)

                        n_frames = min(len(gt_labels), max(1, round(pred_dur_sb / gt_tr)))
                        pred_labels = boundaries_to_frame_labels(pred_b_sb, n_frames, gt_tr)
                        cm = compute_confusion(gt_labels[:n_frames], pred_labels, gt_tr)

                        for k in agg_sb:
                            agg_sb[k] += cm[k]
                            per_snr_sb[snr_level][k] += cm[k]
                            per_noise_sb[noise][k] += cm[k]

                        dm = derived_metrics(cm)
                        p_sp, _ = duration_stats(pred_b_sb, pred_dur_sb)
                        logger.info(f"  [SB] {noisy_path.name:<40} Acc={dm['accuracy']:.3f} "
                                    f"F1={dm['f1']:.3f}  speech={p_sp:.2f}s")
                        pred_b_sb_ok = pred_b_sb

                        # 4-panel plot (only when not "both" mode)
                        if args.model != "both":
                            try:
                                plot_4panel(spk, noise, snr_level, clean_path, noisy_path,
                                            gt_b, pred_b_sb, audio_dur, pred_dur_sb, output_dir,
                                            model_name="SpeechBrain")
                            except Exception as e:
                                logger.warning(f"  Plot failed: {e}")

                    except Exception as exc:
                        logger.warning(f"  [SB] Failed {noisy_path.name}: {exc}")

                # --- Silero ---
                if use_si:
                    try:
                        pred_b_si, pred_dur_si = run_silero_vad(
                            mods_si, noisy_path,
                            args.activation_th, args.deactivation_th,
                            args.close_th, args.len_th)

                        n_frames = min(len(gt_labels), max(1, round(pred_dur_si / gt_tr)))
                        pred_labels = boundaries_to_frame_labels(pred_b_si, n_frames, gt_tr)
                        cm = compute_confusion(gt_labels[:n_frames], pred_labels, gt_tr)

                        for k in agg_si:
                            agg_si[k] += cm[k]
                            per_snr_si[snr_level][k] += cm[k]
                            per_noise_si[noise][k] += cm[k]

                        dm = derived_metrics(cm)
                        p_sp, _ = duration_stats(pred_b_si, pred_dur_si)
                        logger.info(f"  [SI] {noisy_path.name:<40} Acc={dm['accuracy']:.3f} "
                                    f"F1={dm['f1']:.3f}  speech={p_sp:.2f}s")
                        pred_b_si_ok = pred_b_si

                        # 4-panel plot (only when not "both" mode)
                        if args.model != "both":
                            try:
                                si_out_dir = output_dir / "silero"
                                si_out_dir.mkdir(parents=True, exist_ok=True)
                                plot_4panel(spk, noise, snr_level, clean_path, noisy_path,
                                            gt_b, pred_b_si, audio_dur, pred_dur_si, si_out_dir,
                                            model_name="Silero")
                            except Exception as e:
                                logger.warning(f"  Plot failed: {e}")

                    except Exception as exc:
                        logger.warning(f"  [SI] Failed {noisy_path.name}: {exc}")

                # --- 5-panel plot when both models succeeded ---
                if args.model == "both" and pred_b_sb_ok is not None and pred_b_si_ok is not None:
                    try:
                        plot_5panel(spk, noise, snr_level, clean_path, noisy_path,
                                    gt_b, pred_b_sb_ok, pred_b_si_ok,
                                    audio_dur, pred_dur_sb, pred_dur_si, output_dir)
                    except Exception as e:
                        logger.warning(f"  5-panel plot failed: {e}")

                n_processed += 1

    if n_processed == 0:
        logger.error("No files processed."); sys.exit(1)

    # 4. Confusion matrix plots
    def _print_summary(label, agg, per_snr, per_noise, out_dir, ckpt_label="CRDNN"):
        out_cm = plot_confusion_matrix(agg, n_speakers, n_processed, out_dir, ckpt_label=ckpt_label)
        dm_all = derived_metrics(agg)

        print("\n" + "=" * 80)
        print(f"  {label} VAD EVALUATION SUMMARY  (NOIZEUS)")
        print("  GT method: energy-based from clean audio")
        print(f"  Speakers: {n_speakers}   Files scored: {n_processed}")
        print("=" * 80)

        print(f"\n  {'SNR (dB)':>10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'TP(s)':>10} {'FP(s)':>10} {'FN(s)':>10} {'TN(s)':>10}")
        print("  " + "─" * 90)
        rows = []
        for snr in sorted(per_snr.keys()):
            dm = derived_metrics(per_snr[snr])
            c = per_snr[snr]
            print(f"  {snr:>10} {dm['accuracy']:>10.4f} {dm['precision']:>10.4f} "
                  f"{dm['recall']:>10.4f} {dm['f1']:>10.4f} "
                  f"{c['tp']:>10.2f} {c['fp']:>10.2f} {c['fn']:>10.2f} {c['tn']:>10.2f}")
            rows.append({"snr": snr, **dm, **c})

        print(f"\n  {'Noise Type':>14} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print("  " + "─" * 60)
        for nt in sorted(per_noise.keys()):
            dm = derived_metrics(per_noise[nt])
            print(f"  {nt:>14} {dm['accuracy']:>10.4f} {dm['precision']:>10.4f} "
                  f"{dm['recall']:>10.4f} {dm['f1']:>10.4f}")

        print(f"\n  {'OVERALL':>14} {dm_all['accuracy']:>10.4f} {dm_all['precision']:>10.4f} "
              f"{dm_all['recall']:>10.4f} {dm_all['f1']:>10.4f}")

        f1 = dm_all["f1"]
        print("\n" + "─" * 80)
        if f1 >= 0.90:
            verdict = f"✅ EXCELLENT – {label} VAD works very well on this dataset."
        elif f1 >= 0.80:
            verdict = f"✅ GOOD – {label} VAD performs well, minor errors at low SNR."
        elif f1 >= 0.65:
            verdict = f"⚠️  FAIR – VAD struggles at low SNR. Consider fine-tuning or lower thresholds."
        else:
            verdict = "❌ POOR – VAD fails on noisy data. Needs retraining or different model."
        print(f"  VERDICT (F1={f1:.4f}):  {verdict}")
        print("─" * 80)

        csv_path = out_dir / f"per_snr_results_{label.lower().replace(' ', '_')}.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["snr", "accuracy", "precision", "recall", "f1", "tp", "tn", "fp", "fn"])
            w.writeheader()
            w.writerows(rows)

        print(f"\n  Confusion matrix  → {out_cm}")
        print(f"  Per-SNR CSV       → {csv_path}")
        return out_cm

    ckpt_label = f"CRDNN {args.ckpt}"
    if use_sb:
        _print_summary("SpeechBrain", agg_sb, per_snr_sb, per_noise_sb, output_dir,
                        ckpt_label=ckpt_label)
    if use_si:
        si_out_dir = output_dir / "silero" if use_sb else output_dir
        si_out_dir.mkdir(parents=True, exist_ok=True)
        _print_summary("Silero", agg_si, per_snr_si, per_noise_si, si_out_dir)

    # 5. Comparison confusion matrix (both models)
    if args.model == "both":
        out_cmp = plot_comparison_confusion_matrix(
            agg_sb, agg_si, n_speakers, n_processed, output_dir)
        print(f"\n  Comparison matrix → {out_cmp}")

    print(f"  Waveform plots    → {output_dir}/")
    print(f"  Total files processed: {n_processed}\n")


if __name__ == "__main__":
    main()

