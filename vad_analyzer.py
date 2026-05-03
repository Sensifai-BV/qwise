#!/usr/bin/env python3
"""
VAD Multi-Panel Evaluation – NOIZEUS Dataset
=============================================

Panels (per file):
  1 – Clean waveform (blue)
  2 – Noisy waveform (red)
  3 – Silero VAD (orange)       [if enabled]
  4 – SpeechBrain VAD (green)   [if enabled]
  5 – SincQDR-VAD (purple)      [if enabled]
  6 – Tr-VAD (teal)             [if enabled]
  7 – Ground-Truth VAD (dark green)

Ground truth is derived from the **clean audio energy envelope**
(not from running the neural VAD on the clean file).

Also produces:
  - Aggregated confusion matrix PNG (per model + merged for all enabled)
  - Per-SNR summary table (printed + saved as CSV)

Usage:
    python vad_analyzer.py \
        --input_dir  /path/to/noizeus_dataset \
        [--output_dir ./cm_output] \
        [--device cpu] \
        [--enable_speechbrain] [--enable_silero] [--enable_sincqdr] [--enable_trvad] \
        [--silero_model_path ./silero-vad/src/silero_vad/data/silero_vad.onnx] \
        [--sincqdr_ckpt_path ./SincQDR-VAD/ckpt/sincqdr_vad.ckpt] \
        [--trvad_ckpt_path  ./Tr-VAD/checkpoint/weights_10_acc_97.09.pth] \
        [--activation_th   0.5] \
        [--deactivation_th 0.25] \
        [--close_th        0.250] \
        [--len_th          0.250] \
        [--energy_th_db   -40]
        [--save_vad_result]
        [--save_vad_result_as_matrix]
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
import soundfile as sf
import torch
from hyperpyyaml import load_hyperpyyaml

from speechbrain.dataio import audio_io
from speechbrain.utils.checkpoints import Checkpointer
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)

SCRIPT_DIR   = Path(__file__).parent.resolve()
HPARAMS_FILE = SCRIPT_DIR / "speechbrain" / "recipes" / "LibriParty" / "VAD" / "hparams" / "train.yaml"
SAVE_DIR     = SCRIPT_DIR / "speechbrain" / "CRDNN_VAD" / "save"
SILERO_ONNX_DEFAULT  = SCRIPT_DIR / "silero-vad" / "src" / "silero_vad" / "data" / "silero_vad.onnx"
SINCQDR_CKPT_DEFAULT = SCRIPT_DIR / "SincQDR-VAD" / "ckpt" / "sincqdr_vad.ckpt"
TRVAD_CKPT_DEFAULT   = SCRIPT_DIR / "Tr-VAD" / "checkpoint" / "weights_10_acc_97.09.pth"
BOUNDARY_MARKER = 1.0

# SincQDR-VAD hyper-parameters (must match training config)
SINCQDR_WINDOW_SIZE = 0.63   # seconds  (training used 0.63)
SINCQDR_OVERLAP     = 0.875  # fraction
SINCQDR_SR          = 16000
SINCQDR_MEDIAN_K    = 7

# Tr-VAD hyper-parameters
TRVAD_SR = 16000

RE_CLEAN = re.compile(r"^(sp\d+)\.wav$",               re.IGNORECASE)
RE_NOISY = re.compile(r"^(sp\d+)_(.+)_(mic\d+)\.wav$", re.IGNORECASE)

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
    # Search recursively so speaker files inside subdirectories are found
    for f in sorted(input_dir.rglob("*.wav")):
        if not f.is_file():
            continue
        m = RE_CLEAN.match(f.name)
        if m:
            raw[m.group(1).lower()]["clean"] = f
            continue
        m = RE_NOISY.match(f.name)
        if m:
            spk, noise, snr = m.group(1).lower(), m.group(2).lower(), m.group(3)
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
    # ── Load from a specific local checkpoint ──
    overrides = {
        "data_folder": "/tmp",
        "musan_folder": "/tmp",
        "commonlanguage_folder": "/tmp",
        "output_folder": "/tmp",
    }
    with open(HPARAMS_FILE, encoding="utf-8") as fh:
        hparams = load_hyperpyyaml(fh, overrides)

    model = hparams["model"]
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
        "mean_var_norm": mean_var_norm,
        "cnn": hparams["cnn"],
        "rnn": hparams["rnn"],
        "dnn": hparams["dnn"],
        "sample_rate": int(hparams["sample_rate"]),
        "time_resolution": float(hparams["time_resolution"]),
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
# SincQDR-VAD loading & inference
# ────────────────────────────────────────────────────────────────────────────
def load_sincqdr_model(ckpt_path: Path, device: str = "cpu"):
    """Load SincQDR-VAD model from a .ckpt file."""
    sincqdr_dir = SCRIPT_DIR / "SincQDR-VAD"
    if str(sincqdr_dir) not in sys.path:
        sys.path.insert(0, str(sincqdr_dir))
    from model.sincqdrvad import SincQDRVAD  # noqa: PLC0415

    model = SincQDRVAD(
        in_channels=1,
        hidden_channels=32,
        out_channels=64,
        patch_size=8,
        num_blocks=2,
        sinc_conv=True,
    )
    state = torch.load(str(ckpt_path), map_location=device)
    # checkpoint may be raw state_dict or wrapped
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval().to(device)
    logger.info(f"SincQDR-VAD model loaded from {ckpt_path}")
    return model


def run_sincqdr_vad(
    model,
    path: Path,
    act: float,
    deact: float,
    close: float,
    lth: float,
    device: str = "cpu",
    window_size: float = SINCQDR_WINDOW_SIZE,
    overlap: float = SINCQDR_OVERLAP,
    median_k: int = SINCQDR_MEDIAN_K,
) -> Tuple[torch.Tensor, float]:
    """
    Run SincQDR-VAD on an audio file using a sliding window.
    Returns (boundaries, total_duration_seconds).
    """
    import torchaudio

    wav, sr = torchaudio.load(str(path))
    if wav.ndim > 1 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SINCQDR_SR:
        wav = torchaudio.transforms.Resample(sr, SINCQDR_SR)(wav)
        sr = SINCQDR_SR
    wav = wav.squeeze(0)  # (num_samples,)
    total_dur = wav.shape[0] / sr

    win_samples = int(window_size * sr)
    hop_samples = int(win_samples * (1.0 - overlap))
    hop_samples = max(1, hop_samples)

    # Pad so every window is fully filled
    if wav.shape[0] < win_samples:
        wav = torch.nn.functional.pad(wav, (0, win_samples - wav.shape[0]))

    probs = []
    timestamps = []  # centre time of each window
    with torch.no_grad():
        start = 0
        while start + win_samples <= wav.shape[0]:
            chunk = wav[start: start + win_samples].unsqueeze(0).unsqueeze(0).to(device)  # (1,1,T)
            p = model.predict(chunk)
            probs.append(p.squeeze().item())
            timestamps.append((start + win_samples / 2) / sr)
            start += hop_samples
        # last partial window
        if start < wav.shape[0]:
            chunk = wav[start:].unsqueeze(0).unsqueeze(0)
            pad_len = win_samples - chunk.shape[-1]
            chunk = torch.nn.functional.pad(chunk, (0, pad_len)).to(device)
            p = model.predict(chunk)
            probs.append(p.squeeze().item())
            timestamps.append((start + (wav.shape[0] - start) / 2) / sr)

    # Median smoothing
    if median_k > 1 and len(probs) >= median_k:
        probs_t = torch.tensor(probs).unsqueeze(0).unsqueeze(0)  # (1,1,N)
        pad_k = median_k // 2
        probs_t = torch.nn.functional.pad(probs_t, (pad_k, pad_k), mode="reflect")
        smoothed = []
        for i in range(len(probs)):
            window_vals = probs_t[0, 0, i: i + median_k]
            smoothed.append(float(window_vals.median()))
        probs = smoothed

    # Hysteresis threshold → binary labels
    triggered = False
    speech_frames = []
    for p in probs:
        if not triggered:
            triggered = p >= act
        else:
            if p < deact:
                triggered = False
        speech_frames.append(triggered)

    # Convert to boundaries using per-window timestamps
    boundaries = []
    in_speech = False
    start_t = 0.0
    for i, is_sp in enumerate(speech_frames):
        t_start_win = timestamps[i] - (win_samples / 2) / sr
        t_end_win   = timestamps[i] + (win_samples / 2) / sr
        if is_sp and not in_speech:
            start_t = max(0.0, t_start_win)
            in_speech = True
        elif not is_sp and in_speech:
            boundaries.append([start_t, min(total_dur, t_end_win)])
            in_speech = False
    if in_speech:
        boundaries.append([start_t, total_dur])

    if boundaries:
        b = torch.FloatTensor(boundaries)
        b = merge_close(b, close)
        b = remove_short(b, lth)
    else:
        b = torch.zeros(0, 2)

    return b, total_dur


# ────────────────────────────────────────────────────────────────────────────
# Tr-VAD loading & inference
# ────────────────────────────────────────────────────────────────────────────
def load_trvad_model(ckpt_path: Path, device: str = "cpu"):
    """Load Tr-VAD model from a .pth checkpoint."""
    trvad_dir = SCRIPT_DIR / "Tr-VAD"
    if str(trvad_dir) not in sys.path:
        sys.path.insert(0, str(trvad_dir))
    from params import HParams
    from VAD_T import VADModel
    from utils import get_parameter_number

    hparams = HParams()
    model = VADModel(
        dim_in=hparams.dim_in, d_model=hparams.d_model,
        units_in=hparams.units_in, units=hparams.units,
        layers=hparams.layers, P=hparams.P,
        drop_rate=0, activation=hparams.activation,
    ).to(device)
    checkpoint = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    logger.info(f"Tr-VAD model loaded from {ckpt_path}")
    return model, hparams


def run_trvad_vad(
    model,
    hparams,
    path: Path,
    act: float,
    deact: float,
    close: float,
    lth: float,
    device: str = "cpu",
) -> Tuple[torch.Tensor, float]:
    """
    Run Tr-VAD on an audio file. Returns (boundaries, total_duration_seconds).
    """
    import librosa
    trvad_dir = SCRIPT_DIR / "Tr-VAD"
    if str(trvad_dir) not in sys.path:
        sys.path.insert(0, str(trvad_dir))
    from AFPC_feature import AFPC
    from utils import data_transform, bdnn_prediction
    import torch.nn.functional as F

    waveform, sr = librosa.load(str(path), sr=hparams.sample_rate)
    total_dur = len(waveform) / sr
    waveform = waveform / (np.abs(waveform).max() + 1e-12) * 0.999

    feature_input = AFPC.features(
        waveform, fs=sr, nfft=hparams.n_fft,
        winstep=hparams.winstep, winlen=hparams.winlen,
        nfilt=hparams.nfilt, ncoef=hparams.ncoef,
    )[:, :80]
    feature_input = (feature_input - np.mean(feature_input, axis=0)) / (
        np.std(feature_input, axis=0) + 1e-10
    )
    feature_input = torch.as_tensor(feature_input, dtype=torch.float32)

    window_size, unit_size = hparams.w, hparams.u
    feature_input = data_transform(
        feature_input, window_size, unit_size,
        feature_input.min(), DEVICE=torch.device('cpu'),
    )
    feature_input = feature_input[window_size: -window_size, :, :]

    with torch.inference_mode():
        train_data = feature_input.to(device)
        postnet_output = model(train_data)
        _, soft_vad = bdnn_prediction(
            F.sigmoid(postnet_output).cpu().detach().numpy(),
            w=window_size, u=unit_size, threshold=0.5,
        )

    # soft_vad shape: (N, 1) – per-frame soft probabilities
    probs = soft_vad[:, 0].tolist()

    # Pad with leading/trailing zeros to account for the window trimming
    probs = [0.0] * window_size + probs + [0.0] * window_size

    # Each frame covers winstep seconds
    frame_dur = hparams.winstep  # 0.016s

    # Apply hysteresis thresholding
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
        t = i * frame_dur
        if is_speech and not in_speech:
            start = t
            in_speech = True
        elif not is_speech and in_speech:
            boundaries.append([start, t])
            in_speech = False
    if in_speech:
        boundaries.append([start, min(len(speech_frames) * frame_dur, total_dur)])

    if boundaries:
        b = torch.FloatTensor(boundaries)
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
# VAD result saving helpers
# ────────────────────────────────────────────────────────────────────────────
def save_vad_result_audio(noisy_path: Path, boundaries: torch.Tensor,
                          vad_name: str, output_dir: Path) -> Path:
    """Extract and concatenate only the speech segments detected by VAD.

    Output: {original_file_name}_{vad_name}.wav
    Only speech regions are kept; leading/trailing silence and noise gaps
    between segments are removed so the result contains no empty padding.
    """
    wav, sr = load_mono_np(noisy_path)
    n = len(wav)
    chunks = []
    for i in range(boundaries.shape[0]):
        s = max(0, int(boundaries[i, 0].item() * sr))
        e = min(n, int(boundaries[i, 1].item() * sr))
        if e > s:
            chunks.append(wav[s:e])

    if chunks:
        speech_wav = np.concatenate(chunks)
    else:
        speech_wav = np.zeros(0, dtype=wav.dtype)

    stem = noisy_path.stem
    vad_tag = vad_name.lower().replace(" ", "_").replace("-", "")
    out_path = output_dir / f"{stem}_{vad_tag}.wav"
    sf.write(str(out_path), speech_wav, sr)
    return out_path


def save_vad_result_matrix(noisy_path: Path, boundaries: torch.Tensor,
                           dur: float, vad_name: str, output_dir: Path,
                           frame_dur: float = 0.01) -> Path:
    """Save VAD binary frame labels as a numpy .npy file.

    Output: {original_file_name}_{vad_name}.npy
    Each element is a boolean (True=speech, False=noise) per frame of frame_dur seconds.
    """
    n_frames = max(1, round(dur / frame_dur))
    labels = boundaries_to_frame_labels(boundaries, n_frames, frame_dur)

    stem = noisy_path.stem
    vad_tag = vad_name.lower().replace(" ", "_").replace("-", "")
    out_path = output_dir / f"{stem}_{vad_tag}.npy"
    np.save(str(out_path), labels)
    return out_path


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
# Flexible N-Panel Plot  (panels 1–2 are waveforms, middle are VAD models, last is GT)
# ────────────────────────────────────────────────────────────────────────────

# Per-model colours
C_SILERO    = "#FF6F00"
C_SINCQDR   = "#6A1B9A"
C_TRVAD     = "#00796B"


def _draw_vad_panel(ax, boundaries, dur, color, title, x_max, annotate=False):
    t, y = boundaries_to_step(boundaries, dur)
    ax.fill_between(t, y, step="mid", color=color, alpha=0.5)
    ax.fill_between(t, 1 - y, step="mid", color=C_NOISE_FILL, alpha=0.25)
    ax.plot(t, y, color=color, linewidth=1.2, drawstyle="steps-mid")
    ax.set_xlim(0, x_max); ax.set_ylim(-0.05, 1.15)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Noise", "Speech"], fontsize=8)
    ax.set_ylabel("VAD Output", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=4)
    ax.set_facecolor("white")
    ax.axhline(0.5, color="#CCC", lw=0.5, ls="--")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    if annotate:
        _annotate_segments(ax, boundaries, dur)


def plot_panels(
    spk, noise, snr_level,
    clean_path, noisy_path,
    gt_b, audio_dur,
    model_results: List[Dict],   # list of {name, color, boundaries, dur, panel_num}
    output_dir: Path,
) -> Path:
    """
    Generic multi-panel plot.
    model_results: list of dicts with keys name, color, boundaries, dur
    Panels:
      1 – clean waveform
      2 – noisy waveform
      3..N-1 – VAD models (in order given)
      N – Ground-Truth
    """
    clean_wav, c_sr = load_mono_np(clean_path)
    noisy_wav, n_sr = load_mono_np(noisy_path)

    t_clean = np.arange(len(clean_wav)) / c_sr
    t_noisy = np.arange(len(noisy_wav)) / n_sr
    x_max = max(t_clean[-1] if len(t_clean) else 0,
                t_noisy[-1] if len(t_noisy) else 0)

    n_model_panels = len(model_results)
    n_panels = 2 + n_model_panels + 1  # waveforms + models + GT

    h_ratios = [1, 1] + [0.7] * n_model_panels + [0.7]
    fig_h = 3 + 2.2 * n_model_panels

    fig, axes = plt.subplots(n_panels, 1, figsize=(14, fig_h),
                             gridspec_kw={"height_ratios": h_ratios})
    fig.patch.set_facecolor("white")
    plt.subplots_adjust(hspace=0.50, top=0.93, bottom=0.07, left=0.07, right=0.97)

    sp_d, si_d = duration_stats(gt_b, audio_dur)

    # Panel 1 – Clean
    ax = axes[0]
    ax.plot(t_clean, clean_wav, color=C_CLEAN, linewidth=0.4, alpha=0.9)
    ax.set_xlim(0, x_max); ax.set_ylim(-1.1, 1.1)
    ax.set_ylabel("Amplitude", fontsize=9)
    ax.set_title("Panel 1 – Clean Audio Signal", fontsize=10, fontweight="bold", pad=4)
    ax.set_facecolor(C_BG)
    ax.axhline(0, color="#999", lw=0.4)
    ax.text(0.99, 0.95, f"Duration: {audio_dur:.2f}s  |  GT speech: {sp_d:.2f}s  silence: {si_d:.2f}s",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color="#444")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)

    # Panel 2 – Noisy
    ax = axes[1]
    ax.plot(t_noisy, noisy_wav, color=C_NOISY, linewidth=0.4, alpha=0.9)
    ax.set_xlim(0, x_max); ax.set_ylim(-1.1, 1.1)
    ax.set_ylabel("Amplitude", fontsize=9)
    ax.set_title(f"Panel 2 – Noisy Audio ({noise.capitalize()} @ {snr_level} dB SNR)",
                 fontsize=10, fontweight="bold", pad=4)
    ax.set_facecolor(C_BG)
    ax.axhline(0, color="#999", lw=0.4)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)

    # Model panels
    for idx, mres in enumerate(model_results):
        panel_num = 3 + idx
        ax = axes[2 + idx]
        _draw_vad_panel(ax, mres["boundaries"], mres["dur"], mres["color"],
                        f"Panel {panel_num} – {mres['name']} VAD Detection (on noisy audio)",
                        x_max, annotate=True)

    # GT panel (last)
    gt_panel_num = 3 + n_model_panels
    ax = axes[-1]
    t_gt, y_gt = boundaries_to_step(gt_b, audio_dur)
    ax.fill_between(t_gt, y_gt, step="mid", color=C_GT_SPEECH, alpha=0.45)
    ax.fill_between(t_gt, 1 - y_gt, step="mid", color=C_GT_NOISE, alpha=0.15)
    ax.plot(t_gt, y_gt, color=C_GT_SPEECH, linewidth=1.2, drawstyle="steps-mid")
    ax.set_xlim(0, x_max); ax.set_ylim(-0.05, 1.15)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Noise", "Speech"], fontsize=8)
    ax.set_ylabel("VAD Output", fontsize=9)
    ax.set_title(f"Panel {gt_panel_num} – Ground-Truth VAD (energy-based from clean audio)",
                 fontsize=10, fontweight="bold", pad=4)
    ax.set_facecolor("white")
    ax.axhline(0.5, color="#CCC", lw=0.5, ls="--")
    ax.set_xlabel("Time [s]", fontsize=9)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    _annotate_segments(ax, gt_b, audio_dur)

    # Suptitle
    fig.suptitle(
        f"VAD Analysis  –  {spk.upper()}  |  {noise.capitalize()} noise  |  {snr_level} dB SNR",
        fontsize=12, fontweight="bold", y=0.99)

    # Legend
    legend_handles = []
    for mres in model_results:
        legend_handles.append(mpatches.Patch(color=mres["color"], alpha=0.55, label=f"{mres['name']}: Speech"))
    legend_handles += [
        mpatches.Patch(color=C_NOISE_FILL, alpha=0.35, label="Predicted: Noise"),
        mpatches.Patch(color=C_GT_SPEECH,  alpha=0.55, label="GT: Speech"),
        mpatches.Patch(color=C_GT_NOISE,   alpha=0.35, label="GT: Noise/Silence"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=min(len(legend_handles), 5), fontsize=8.5,
               framealpha=0.9, bbox_to_anchor=(0.5, 0.002))

    out = output_dir / f"{spk}_{noise}_sn{snr_level}.png"
    plt.savefig(str(out), dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# Keep backward-compatible wrappers
def plot_4panel(spk, noise, snr_level, clean_path, noisy_path,
                gt_b, pred_b, audio_dur, pred_dur, output_dir,
                model_name: str = "SpeechBrain") -> Path:
    color = C_SPEECH if "Speech" in model_name else (
            C_SILERO if "Silero" in model_name else C_SINCQDR)
    return plot_panels(spk, noise, snr_level, clean_path, noisy_path,
                       gt_b, audio_dur,
                       [{"name": model_name, "color": color, "boundaries": pred_b, "dur": pred_dur}],
                       output_dir)


def plot_5panel(spk, noise, snr_level, clean_path, noisy_path,
                gt_b, pred_b_sb, pred_b_si,
                audio_dur, pred_dur_sb, pred_dur_si, output_dir) -> Path:
    return plot_panels(spk, noise, snr_level, clean_path, noisy_path,
                       gt_b, audio_dur,
                       [{"name": "Silero",      "color": C_SILERO, "boundaries": pred_b_si, "dur": pred_dur_si},
                        {"name": "SpeechBrain", "color": C_SPEECH, "boundaries": pred_b_sb, "dur": pred_dur_sb}],
                       output_dir)



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
# Generic N-model comparison confusion matrix plot
# ────────────────────────────────────────────────────────────────────────────
def plot_comparison_confusion_matrix(
    model_cms: List[Tuple[str, Dict[str, float]]],
    n_speakers: int,
    n_noisy: int,
    output_dir: Path,
) -> Path:
    """Side-by-side confusion matrices for all enabled models."""
    n_models = len(model_cms)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 6.5))
    if n_models == 1:
        axes = [axes]
    fig.patch.set_facecolor("#F8F9FA")

    cell_meta = [
        [("True Noise",    "TN", "#1565C0"), ("False Speech", "FP", "#E53935")],
        [("Missed Speech", "FN", "#FB8C00"), ("True Speech",  "TP", "#2E7D32")],
    ]

    for ax, (title, cm) in zip(axes, model_cms):
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

    model_names = [name for name, _ in model_cms]
    fig.suptitle(
        "VAD Model Comparison – Aggregated Confusion Matrices\n"
        "NOIZEUS dataset  (GT = energy-based from clean audio)",
        fontsize=12, fontweight="bold", y=1.02)

    f1_parts = "   ".join(
        f"{name} F1={derived_metrics(cm)['f1']:.4f}" for name, cm in model_cms
    )
    fig.text(0.5, -0.02,
             f"Speakers={n_speakers}   Files={n_noisy}\n{f1_parts}",
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
    # Enable flags – enable only the models you want
    p.add_argument("--enable_speechbrain", action="store_true",
                   help="Enable SpeechBrain VAD panel")
    p.add_argument("--enable_silero",      action="store_true",
                   help="Enable Silero VAD panel")
    p.add_argument("--enable_sincqdr",     action="store_true",
                   help="Enable SincQDR-VAD panel")
    p.add_argument("--enable_trvad",       action="store_true",
                   help="Enable Tr-VAD panel")
    p.add_argument("--silero_model_path", type=Path,
                   default=SILERO_ONNX_DEFAULT,
                   help="Path to Silero VAD ONNX model file")
    p.add_argument("--sincqdr_ckpt_path", type=Path,
                   default=SINCQDR_CKPT_DEFAULT,
                   help="Path to SincQDR-VAD checkpoint (.ckpt)")
    p.add_argument("--trvad_ckpt_path", type=Path,
                   default=TRVAD_CKPT_DEFAULT,
                   help="Path to Tr-VAD checkpoint (.pth)")
    p.add_argument("--ckpt", default="epoch_100",
                   choices=["epoch_91", "epoch_100"],
                   help="Which SpeechBrain checkpoint to use (default: epoch_100)")
    p.add_argument("--activation_th",   type=float, default=0.5)
    p.add_argument("--deactivation_th", type=float, default=0.25)
    p.add_argument("--close_th",        type=float, default=0.25)
    p.add_argument("--len_th",          type=float, default=0.25)
    p.add_argument("--energy_th_db",    type=float, default=-55.0,
                   help="Min energy threshold floor (dB) for GT VAD on clean audio")
    p.add_argument("--save_vad_result", action="store_true",
                   help="Save VAD-masked audio as {original_file_name}_{vad_name}.wav in output_dir")
    p.add_argument("--save_vad_result_as_matrix", action="store_true",
                   help="Save VAD binary frame labels as {original_file_name}_{vad_name}.npy in output_dir")
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

    # Determine which models are active
    use_sb = args.enable_speechbrain
    use_si = args.enable_silero
    use_sq = args.enable_sincqdr
    use_tv = args.enable_trvad

    if not (use_sb or use_si or use_sq or use_tv):
        logger.error("No models enabled – use --enable_speechbrain, --enable_silero, "
                      "--enable_sincqdr, and/or --enable_trvad."); sys.exit(1)

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
    mods_sq = None
    mods_tv = None
    tv_hparams = None
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

    if use_sq:
        logger.info("Loading SincQDR-VAD model …")
        if not args.sincqdr_ckpt_path.exists():
            logger.error(f"SincQDR checkpoint not found: {args.sincqdr_ckpt_path}")
            sys.exit(1)
        mods_sq = load_sincqdr_model(args.sincqdr_ckpt_path, device=args.device)
        logger.info("SincQDR-VAD model loaded.")

    if use_tv:
        logger.info("Loading Tr-VAD model …")
        if not args.trvad_ckpt_path.exists():
            logger.error(f"Tr-VAD checkpoint not found: {args.trvad_ckpt_path}")
            sys.exit(1)
        mods_tv, tv_hparams = load_trvad_model(args.trvad_ckpt_path, device=args.device)
        logger.info("Tr-VAD model loaded.")

    # 3. Evaluation
    # Aggregated confusion matrices per model
    agg_sb = {"tp": 0.0, "tn": 0.0, "fp": 0.0, "fn": 0.0}
    agg_si = {"tp": 0.0, "tn": 0.0, "fp": 0.0, "fn": 0.0}
    agg_sq = {"tp": 0.0, "tn": 0.0, "fp": 0.0, "fn": 0.0}
    agg_tv = {"tp": 0.0, "tn": 0.0, "fp": 0.0, "fn": 0.0}
    per_snr_sb:    Dict[int, Dict[str, float]] = defaultdict(lambda: {"tp": 0, "tn": 0, "fp": 0, "fn": 0})
    per_snr_si:    Dict[int, Dict[str, float]] = defaultdict(lambda: {"tp": 0, "tn": 0, "fp": 0, "fn": 0})
    per_snr_sq:    Dict[int, Dict[str, float]] = defaultdict(lambda: {"tp": 0, "tn": 0, "fp": 0, "fn": 0})
    per_snr_tv:    Dict[int, Dict[str, float]] = defaultdict(lambda: {"tp": 0, "tn": 0, "fp": 0, "fn": 0})
    per_noise_sb:  Dict[str, Dict[str, float]] = defaultdict(lambda: {"tp": 0, "tn": 0, "fp": 0, "fn": 0})
    per_noise_si:  Dict[str, Dict[str, float]] = defaultdict(lambda: {"tp": 0, "tn": 0, "fp": 0, "fn": 0})
    per_noise_sq:  Dict[str, Dict[str, float]] = defaultdict(lambda: {"tp": 0, "tn": 0, "fp": 0, "fn": 0})
    per_noise_tv:  Dict[str, Dict[str, float]] = defaultdict(lambda: {"tp": 0, "tn": 0, "fp": 0, "fn": 0})
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
                pred_b_sb_ok, pred_b_si_ok, pred_b_sq_ok, pred_b_tv_ok = None, None, None, None
                pred_dur_sb, pred_dur_si, pred_dur_sq, pred_dur_tv = 0.0, 0.0, 0.0, 0.0

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

                    except Exception as exc:
                        logger.warning(f"  [SI] Failed {noisy_path.name}: {exc}")

                # --- SincQDR ---
                if use_sq:
                    try:
                        pred_b_sq, pred_dur_sq = run_sincqdr_vad(
                            mods_sq, noisy_path,
                            args.activation_th, args.deactivation_th,
                            args.close_th, args.len_th, device=args.device)

                        n_frames = min(len(gt_labels), max(1, round(pred_dur_sq / gt_tr)))
                        pred_labels = boundaries_to_frame_labels(pred_b_sq, n_frames, gt_tr)
                        cm = compute_confusion(gt_labels[:n_frames], pred_labels, gt_tr)

                        for k in agg_sq:
                            agg_sq[k] += cm[k]
                            per_snr_sq[snr_level][k] += cm[k]
                            per_noise_sq[noise][k] += cm[k]

                        dm = derived_metrics(cm)
                        p_sp, _ = duration_stats(pred_b_sq, pred_dur_sq)
                        logger.info(f"  [SQ] {noisy_path.name:<40} Acc={dm['accuracy']:.3f} "
                                    f"F1={dm['f1']:.3f}  speech={p_sp:.2f}s")
                        pred_b_sq_ok = pred_b_sq

                    except Exception as exc:
                        logger.warning(f"  [SQ] Failed {noisy_path.name}: {exc}")

                # --- Tr-VAD ---
                if use_tv:
                    try:
                        pred_b_tv, pred_dur_tv = run_trvad_vad(
                            mods_tv, tv_hparams, noisy_path,
                            args.activation_th, args.deactivation_th,
                            args.close_th, args.len_th, device=args.device)

                        n_frames = min(len(gt_labels), max(1, round(pred_dur_tv / gt_tr)))
                        pred_labels = boundaries_to_frame_labels(pred_b_tv, n_frames, gt_tr)
                        cm = compute_confusion(gt_labels[:n_frames], pred_labels, gt_tr)

                        for k in agg_tv:
                            agg_tv[k] += cm[k]
                            per_snr_tv[snr_level][k] += cm[k]
                            per_noise_tv[noise][k] += cm[k]

                        dm = derived_metrics(cm)
                        p_sp, _ = duration_stats(pred_b_tv, pred_dur_tv)
                        logger.info(f"  [TV] {noisy_path.name:<40} Acc={dm['accuracy']:.3f} "
                                    f"F1={dm['f1']:.3f}  speech={p_sp:.2f}s")
                        pred_b_tv_ok = pred_b_tv

                    except Exception as exc:
                        logger.warning(f"  [TV] Failed {noisy_path.name}: {exc}")

                # --- Combined / per-model plot ---
                model_results = []
                if use_si and pred_b_si_ok is not None:
                    model_results.append({"name": "Silero",       "color": C_SILERO,
                                          "boundaries": pred_b_si_ok, "dur": pred_dur_si})
                if use_sb and pred_b_sb_ok is not None:
                    model_results.append({"name": "SpeechBrain",  "color": C_SPEECH,
                                          "boundaries": pred_b_sb_ok, "dur": pred_dur_sb})
                if use_sq and pred_b_sq_ok is not None:
                    model_results.append({"name": "SincQDR-VAD",  "color": C_SINCQDR,
                                          "boundaries": pred_b_sq_ok, "dur": pred_dur_sq})
                if use_tv and pred_b_tv_ok is not None:
                    model_results.append({"name": "Tr-VAD",       "color": C_TRVAD,
                                          "boundaries": pred_b_tv_ok, "dur": pred_dur_tv})

                if model_results:
                    try:
                        plot_panels(spk, noise, snr_level, clean_path, noisy_path,
                                    gt_b, audio_dur, model_results, output_dir)
                    except Exception as e:
                        logger.warning(f"  Plot failed: {e}")

                    # Save VAD result audio / matrix if requested
                    for mres in model_results:
                        if args.save_vad_result:
                            try:
                                out_wav = save_vad_result_audio(
                                    noisy_path, mres["boundaries"],
                                    mres["name"], output_dir)
                                logger.info(f"    Saved VAD audio: {out_wav.name}")
                            except Exception as e:
                                logger.warning(f"    Save VAD audio failed ({mres['name']}): {e}")
                        if args.save_vad_result_as_matrix:
                            try:
                                out_npy = save_vad_result_matrix(
                                    noisy_path, mres["boundaries"],
                                    mres["dur"], mres["name"], output_dir)
                                logger.info(f"    Saved VAD matrix: {out_npy.name}")
                            except Exception as e:
                                logger.warning(f"    Save VAD matrix failed ({mres['name']}): {e}")

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
        si_out_dir = output_dir / "silero" if (use_sb or use_sq or use_tv) else output_dir
        si_out_dir.mkdir(parents=True, exist_ok=True)
        _print_summary("Silero", agg_si, per_snr_si, per_noise_si, si_out_dir)
    if use_sq:
        sq_out_dir = output_dir / "sincqdr" if (use_sb or use_si or use_tv) else output_dir
        sq_out_dir.mkdir(parents=True, exist_ok=True)
        _print_summary("SincQDR", agg_sq, per_snr_sq, per_noise_sq, sq_out_dir)
    if use_tv:
        tv_out_dir = output_dir / "trvad" if (use_sb or use_si or use_sq) else output_dir
        tv_out_dir.mkdir(parents=True, exist_ok=True)
        _print_summary("Tr-VAD", agg_tv, per_snr_tv, per_noise_tv, tv_out_dir)

    # 5. Comparison confusion matrix (all enabled models side-by-side)
    active_cms = []
    if use_sb: active_cms.append(("SpeechBrain (CRDNN)", agg_sb))
    if use_si: active_cms.append(("Silero (ONNX)",       agg_si))
    if use_sq: active_cms.append(("SincQDR-VAD",         agg_sq))
    if use_tv: active_cms.append(("Tr-VAD",              agg_tv))
    if len(active_cms) > 1:
        out_cmp = plot_comparison_confusion_matrix(
            active_cms, n_speakers, n_processed, output_dir)
        print(f"\n  Comparison matrix → {out_cmp}")

    print(f"  Waveform plots    → {output_dir}/")
    print(f"  Total files processed: {n_processed}\n")


if __name__ == "__main__":
    main()

