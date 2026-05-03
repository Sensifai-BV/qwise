#!/usr/bin/env python3
"""
mwf.py – Multi-channel Wiener Filter (MWF) for speech enhancement.

Uses VAD (Voice Activity Detection) results to estimate speech and noise
spatial covariance matrices from multi-microphone recordings, then applies
an SDW-MWF / MVDR / GEV beamformer to produce a clean speech output.

Supports any number of microphones (3, 5, 10, ...).

Usage Examples:
    # 3 microphones, distance 0.8m
    python mwf.py \
        --vad datasets/noise/drone-sample/vad_result/sp01_man_0.8_mic1_silero.wav \
        --mics datasets/noise/drone-sample/man/sp01_man_0.8_mic1.wav \
              datasets/noise/drone-sample/man/sp01_man_0.8_mic2.wav \
              datasets/noise/drone-sample/man/sp01_man_0.8_mic3.wav \
        -o output_clean.wav

    # Process all distances for sp01_man
    python mwf.py \
        --vad datasets/noise/drone-sample/vad_result/sp01_man_1.0_mic1_silero.wav \
        --mics datasets/noise/drone-sample/man/sp01_man_1.0_mic1.wav \
              datasets/noise/drone-sample/man/sp01_man_1.0_mic2.wav \
              datasets/noise/drone-sample/man/sp01_man_1.0_mic3.wav \
        -o sp01_man_1.0_clean.wav

Algorithm:
    1. Load VAD result (extracted speech) + microphone signals
    2. Cross-correlate VAD with reference mic to find temporal alignment
    3. Build proper speech/noise mask aligned to mic signals
    4. Compute STFT for all channels
    5. Estimate noise covariance Φ_nn from noise-only frames (VAD=0)
    6. Estimate speech+noise covariance Φ_yy from speech frames (VAD=1)
    7. Compute speech covariance Φ_ss = Φ_yy - Φ_nn (PSD enforced)
    8. Compute beamformer: MWF / MVDR / GEV
    9. Apply beamformer + Wiener post-filter → iSTFT → save clean WAV
"""

from __future__ import annotations

import argparse

import numpy as np
import soundfile as sf
from scipy import signal as sig
from scipy.linalg import eigh as generalized_eigh


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16_000
N_FFT = 1024
HOP_LENGTH = 256
EPSILON = 1e-10  # regularization for matrix inversion


# ─────────────────────────────────────────────────────────────────────────────
# Audio I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_wav(path: str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load a WAV file, resample if needed, return mono float64 array."""
    data, sr = sf.read(path, dtype="float64")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        num_samples = int(len(data) * target_sr / sr)
        data = sig.resample(data, num_samples)
    return data


def save_wav(path: str, data: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    """Save a numpy array as a WAV file."""
    # Normalize to -1..1 range with headroom
    peak = np.abs(data).max()
    if peak > 0:
        data = data / peak * 0.9
    sf.write(path, data.astype(np.float64), sr, subtype="PCM_16")


# ─────────────────────────────────────────────────────────────────────────────
# STFT / iSTFT
# ─────────────────────────────────────────────────────────────────────────────

def compute_stft(x: np.ndarray, n_fft: int = N_FFT, hop: int = HOP_LENGTH) -> np.ndarray:
    """Compute STFT. Returns complex array of shape (n_freq, n_frames)."""
    window = sig.windows.hann(n_fft, sym=False)
    _, _, Zxx = sig.stft(x, fs=SAMPLE_RATE, window=window,
                         nperseg=n_fft, noverlap=n_fft - hop,
                         boundary="zeros", padded=True)
    return Zxx  # (n_freq, n_frames)


def compute_istft(X: np.ndarray, n_fft: int = N_FFT, hop: int = HOP_LENGTH) -> np.ndarray:
    """Compute inverse STFT. Input: (n_freq, n_frames) → output: (n_samples,)."""
    window = sig.windows.hann(n_fft, sym=False)
    _, xrec = sig.istft(X, fs=SAMPLE_RATE, window=window,
                        nperseg=n_fft, noverlap=n_fft - hop,
                        boundary=True)
    return xrec


# ─────────────────────────────────────────────────────────────────────────────
# VAD Alignment & Mask Generation
# ─────────────────────────────────────────────────────────────────────────────

def align_vad_to_mic(
    vad_signal: np.ndarray,
    mic_signal: np.ndarray,
) -> tuple[int, np.ndarray]:
    """
    Find where the VAD-extracted speech occurs within the full mic signal
    using cross-correlation.

    Returns:
        lag: sample offset where VAD signal aligns in mic signal
        aligned_vad: full-length signal with VAD placed at correct position
    """
    # Cross-correlate to find best alignment
    corr = sig.correlate(mic_signal, vad_signal, mode="valid")
    lag = int(np.argmax(np.abs(corr)))

    # Place VAD signal at correct position
    aligned_vad = np.zeros_like(mic_signal)
    end_idx = min(lag + len(vad_signal), len(mic_signal))
    copy_len = end_idx - lag
    aligned_vad[lag:end_idx] = vad_signal[:copy_len]

    return lag, aligned_vad


def build_speech_mask(
    aligned_vad: np.ndarray,
    n_frames: int,
    hop: int = HOP_LENGTH,
    n_fft: int = N_FFT,
    threshold: float = 0.01,
    context_frames: int = 3,
) -> np.ndarray:
    """
    Build a frame-level speech/noise mask from the aligned VAD signal.

    Uses RMS energy per frame with adaptive thresholding and adds
    context frames around detected speech for better covariance estimation.

    Returns:
        mask: (n_frames,) boolean array. True = speech, False = noise.
    """
    frame_energy = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop
        end = start + n_fft
        if end > len(aligned_vad):
            break
        frame = aligned_vad[start:end]
        frame_energy[i] = np.sqrt(np.mean(frame ** 2))

    # Adaptive threshold: use a fraction of the max energy
    max_energy = frame_energy.max()
    if max_energy > 0:
        norm_energy = frame_energy / max_energy
        mask = norm_energy > threshold
    else:
        mask = np.zeros(n_frames, dtype=bool)
        return mask

    # Add context frames (speech transitions)
    expanded_mask = mask.copy()
    for i in range(n_frames):
        if mask[i]:
            start = max(0, i - context_frames)
            end = min(n_frames, i + context_frames + 1)
            expanded_mask[start:end] = True

    return expanded_mask


# ─────────────────────────────────────────────────────────────────────────────
# Covariance Estimation (improved)
# ─────────────────────────────────────────────────────────────────────────────

def estimate_covariance(
    X_multi: np.ndarray,
    mask: np.ndarray,
    exponential_decay: float = 0.0,
) -> np.ndarray:
    """
    Estimate spatial covariance matrix from multi-channel STFT.

    Args:
        X_multi: (n_channels, n_freq, n_frames) complex STFT
        mask: (n_frames,) boolean – which frames to use
        exponential_decay: if > 0, weight recent frames more heavily

    Returns:
        Phi: (n_freq, n_channels, n_channels) complex covariance matrix per frequency
    """
    n_channels, n_freq, n_frames = X_multi.shape
    Phi = np.zeros((n_freq, n_channels, n_channels), dtype=np.complex128)

    frame_indices = np.where(mask)[0]
    n_valid = len(frame_indices)

    if n_valid == 0:
        return Phi + EPSILON * np.eye(n_channels, dtype=np.complex128)[np.newaxis, :, :]

    # Compute weights (optional exponential decay)
    if exponential_decay > 0 and n_valid > 1:
        weights = np.exp(exponential_decay * np.arange(n_valid) / n_valid)
        weights /= weights.sum()
    else:
        weights = np.ones(n_valid) / n_valid

    for f in range(n_freq):
        obs = X_multi[:, f, frame_indices]  # (n_channels, n_valid)
        # Weighted covariance
        weighted_obs = obs * np.sqrt(weights)[np.newaxis, :]
        Phi[f] = weighted_obs @ weighted_obs.conj().T

    # Clean up numerical issues
    Phi = np.nan_to_num(Phi, nan=0.0, posinf=0.0, neginf=0.0)

    # Add small diagonal loading for numerical stability
    Phi += EPSILON * np.eye(n_channels, dtype=np.complex128)[np.newaxis, :, :]

    return Phi


# ─────────────────────────────────────────────────────────────────────────────
# Beamformer Weights
# ─────────────────────────────────────────────────────────────────────────────

def compute_mwf_weights(
    Phi_ss: np.ndarray,
    Phi_nn: np.ndarray,
    ref_channel: int = 0,
    mu: float = 1.0,
) -> np.ndarray:
    """
    Speech Distortion Weighted MWF:
    W(f) = Φ_ss(f) @ inv(Φ_ss(f) + μ·Φ_nn(f)) @ e_ref
    """
    n_freq, n_ch, _ = Phi_ss.shape
    W = np.zeros((n_freq, n_ch), dtype=np.complex128)

    e_ref = np.zeros(n_ch, dtype=np.complex128)
    e_ref[ref_channel] = 1.0

    for f in range(n_freq):
        Phi_yy = Phi_ss[f] + mu * Phi_nn[f]
        # Diagonal loading proportional to trace for robustness
        diag_load = max(EPSILON, 1e-4 * np.real(np.trace(Phi_yy)) / n_ch)
        Phi_yy += diag_load * np.eye(n_ch, dtype=np.complex128)

        try:
            Phi_yy_inv = np.linalg.inv(Phi_yy)
        except np.linalg.LinAlgError:
            Phi_yy_inv = np.linalg.pinv(Phi_yy)

        W[f] = Phi_ss[f] @ Phi_yy_inv @ e_ref

    return W


def compute_mvdr_weights(
    Phi_ss: np.ndarray,
    Phi_nn: np.ndarray,
    ref_channel: int = 0,
) -> np.ndarray:
    """
    MVDR beamformer with steering vector from Φ_ss principal eigenvector.
    W(f) = Φ_nn^{-1} @ a / (a^H @ Φ_nn^{-1} @ a)
    """
    n_freq, n_ch, _ = Phi_ss.shape
    W = np.zeros((n_freq, n_ch), dtype=np.complex128)

    for f in range(n_freq):
        # Steering vector from principal eigenvector of Φ_ss
        eigvals, eigvecs = np.linalg.eigh(Phi_ss[f])
        a = eigvecs[:, -1]

        # Normalize to ref channel
        if np.abs(a[ref_channel]) > EPSILON:
            a = a / a[ref_channel]

        # Invert noise covariance with diagonal loading
        diag_load = max(EPSILON, 1e-4 * np.real(np.trace(Phi_nn[f])) / n_ch)
        Phi_nn_reg = Phi_nn[f] + diag_load * np.eye(n_ch, dtype=np.complex128)

        try:
            Phi_nn_inv = np.linalg.inv(Phi_nn_reg)
        except np.linalg.LinAlgError:
            Phi_nn_inv = np.linalg.pinv(Phi_nn_reg)

        num = Phi_nn_inv @ a
        denom = a.conj() @ num
        if np.abs(denom) > EPSILON:
            W[f] = num / denom
        else:
            W[f, ref_channel] = 1.0

    return W


def compute_gev_weights(
    Phi_ss: np.ndarray,
    Phi_nn: np.ndarray,
    ref_channel: int = 0,
) -> np.ndarray:
    """
    Generalized Eigenvalue (GEV / Max-SNR) beamformer.

    Finds w that maximizes w^H Φ_ss w / w^H Φ_nn w
    by solving the generalized eigenvalue problem: Φ_ss w = λ Φ_nn w

    This is the most robust beamformer for spatially non-stationary noise
    like drone propeller noise.
    """
    n_freq, n_ch, _ = Phi_ss.shape
    W = np.zeros((n_freq, n_ch), dtype=np.complex128)

    for f in range(n_freq):
        # Diagonal loading for both matrices
        diag_ss = max(EPSILON, 1e-4 * np.real(np.trace(Phi_ss[f])) / n_ch)
        diag_nn = max(EPSILON, 1e-4 * np.real(np.trace(Phi_nn[f])) / n_ch)
        A = Phi_ss[f] + diag_ss * np.eye(n_ch, dtype=np.complex128)
        B = Phi_nn[f] + diag_nn * np.eye(n_ch, dtype=np.complex128)

        try:
            # Solve generalized eigenvalue problem: A w = λ B w
            eigvals, eigvecs = generalized_eigh(A, B)
            # Select eigenvector with largest eigenvalue (max SNR direction)
            w = eigvecs[:, -1]
        except (np.linalg.LinAlgError, ValueError):
            # Fallback to standard eigendecomposition
            try:
                B_inv = np.linalg.inv(B)
                eigvals, eigvecs = np.linalg.eigh(B_inv @ A)
                w = eigvecs[:, -1]
            except np.linalg.LinAlgError:
                w = np.zeros(n_ch, dtype=np.complex128)
                w[ref_channel] = 1.0

        # Normalize: fix phase to reference channel
        if np.abs(w[ref_channel]) > EPSILON:
            w = w / w[ref_channel]

        W[f] = w

    return W


# ─────────────────────────────────────────────────────────────────────────────
# Apply beamformer
# ─────────────────────────────────────────────────────────────────────────────

def apply_beamformer(
    W: np.ndarray,
    X_multi: np.ndarray,
) -> np.ndarray:
    """
    Apply beamformer weights to multi-channel STFT.
    Y(f,t) = W(f)^H @ X(f,t)
    """
    n_channels, n_freq, n_frames = X_multi.shape
    Y = np.zeros((n_freq, n_frames), dtype=np.complex128)

    for f in range(n_freq):
        Y[f, :] = W[f].conj() @ X_multi[:, f, :]

    return Y


# ─────────────────────────────────────────────────────────────────────────────
# Post-filter (improved single-channel Wiener with spectral smoothing)
# ─────────────────────────────────────────────────────────────────────────────

def wiener_postfilter(
    Y: np.ndarray,
    speech_mask: np.ndarray,
    noise_floor_alpha: float = 0.98,
    gain_floor: float = 0.08,
) -> np.ndarray:
    """
    Apply an adaptive single-channel Wiener post-filter based on
    estimated SNR per frequency bin.

    Uses decision-directed approach with noise floor tracking.

    Args:
        Y: (n_freq, n_frames) beamformer output
        speech_mask: (n_frames,) boolean speech mask
        noise_floor_alpha: smoothing for noise estimate (0.9-0.99)
        gain_floor: minimum gain to avoid musical noise (0.05-0.15)

    Returns:
        Y_filtered: (n_freq, n_frames) post-filtered output
    """
    n_freq, n_frames = Y.shape
    Y_mag = np.abs(Y)
    Y_phase = np.angle(Y)

    # Estimate noise spectrum from noise-only frames
    noise_indices = np.where(~speech_mask)[0]
    if len(noise_indices) > 0:
        noise_spectrum = np.mean(Y_mag[:, noise_indices] ** 2, axis=1)
    else:
        noise_spectrum = np.mean(Y_mag ** 2, axis=1) * 0.1

    # Smooth noise spectrum
    noise_spectrum = np.maximum(noise_spectrum, EPSILON)

    # Apply frame-by-frame Wiener filter
    G = np.ones_like(Y_mag)
    noise_est = noise_spectrum.copy()

    for t in range(n_frames):
        frame_power = Y_mag[:, t] ** 2

        # Update noise estimate during noise frames
        if not speech_mask[t]:
            noise_est = noise_floor_alpha * noise_est + (1 - noise_floor_alpha) * frame_power

        # A priori SNR estimate
        snr = np.maximum(frame_power - noise_est, 0.0) / (noise_est + EPSILON)

        # Wiener gain
        gain = snr / (snr + 1.0)

        # Apply gain floor to avoid musical noise
        gain = np.maximum(gain, gain_floor)

        G[:, t] = gain

    # Smooth gain across frequency (reduce musical noise)
    kernel_size = 3
    for t in range(n_frames):
        G[:, t] = np.convolve(G[:, t], np.ones(kernel_size) / kernel_size, mode="same")

    # Reconstruct
    Y_filtered = Y_mag * G * np.exp(1j * Y_phase)

    return Y_filtered


# ─────────────────────────────────────────────────────────────────────────────
# Main MWF Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def multichannel_wiener_filter(
    mic_signals: list[np.ndarray],
    vad_signal: np.ndarray,
    method: str = "gev",
    ref_channel: int = 0,
    mu: float = 1.0,
    postfilter: bool = True,
    gain_floor: float = 0.08,
) -> np.ndarray:
    """
    Complete multi-channel Wiener filter pipeline.

    Args:
        mic_signals: list of N microphone signals, each (n_samples,) float
        vad_signal: VAD result signal (extracted speech audio from one mic)
        method: "mwf", "mvdr", or "gev" (recommended for drone noise)
        ref_channel: reference microphone index (0-based)
        mu: MWF distortion-vs-noise tradeoff (only for method="mwf")
        postfilter: whether to apply Wiener post-filter
        gain_floor: minimum gain for post-filter (prevents musical noise)

    Returns:
        clean_signal: (n_samples,) enhanced audio
    """
    n_channels = len(mic_signals)
    print(f"\n{'═' * 60}")
    print(f"  Multi-channel Wiener Filter ({method.upper()})")
    print(f"  Channels: {n_channels} | Ref: mic{ref_channel + 1}")
    print(f"  N_FFT: {N_FFT} | Hop: {HOP_LENGTH} | SR: {SAMPLE_RATE} Hz")
    print(f"{'═' * 60}")

    # ── Ensure all mic signals have the same length ──────────────────────────
    min_mic_len = min(len(s) for s in mic_signals)
    mic_signals = [s[:min_mic_len] for s in mic_signals]

    print(f"  Signal length: {min_mic_len} samples ({min_mic_len / SAMPLE_RATE:.2f} s)")
    print(f"  VAD length: {len(vad_signal)} samples ({len(vad_signal) / SAMPLE_RATE:.2f} s)")

    # ── Step 1: Align VAD to reference mic via cross-correlation ─────────────
    print("  [1/6] Aligning VAD to reference mic (cross-correlation)...")
    lag, aligned_vad = align_vad_to_mic(vad_signal, mic_signals[ref_channel])
    print(f"        Alignment lag: {lag} samples ({lag / SAMPLE_RATE:.3f} s)")

    # ── Step 2: Compute STFT for all channels ────────────────────────────────
    print("  [2/6] Computing multi-channel STFT...")
    stft_list = [compute_stft(s) for s in mic_signals]
    n_freq, n_frames = stft_list[0].shape
    X_multi = np.stack(stft_list, axis=0)  # (n_channels, n_freq, n_frames)
    print(f"        Shape: ({n_channels}, {n_freq}, {n_frames})")

    # ── Step 3: Build aligned speech mask ────────────────────────────────────
    print("  [3/6] Building speech/noise mask...")
    speech_mask = build_speech_mask(aligned_vad, n_frames)
    noise_mask = ~speech_mask

    n_speech = int(speech_mask.sum())
    n_noise = int(noise_mask.sum())
    print(f"        Speech frames: {n_speech} ({100*n_speech/n_frames:.1f}%)")
    print(f"        Noise frames:  {n_noise} ({100*n_noise/n_frames:.1f}%)")

    # Safety checks
    if n_speech < 5:
        print("  ⚠ Very few speech frames! Trying lower threshold...")
        speech_mask = build_speech_mask(aligned_vad, n_frames, threshold=0.003)
        noise_mask = ~speech_mask
        n_speech = int(speech_mask.sum())
        n_noise = int(noise_mask.sum())
        print(f"        Speech frames: {n_speech} | Noise frames: {n_noise}")

    if n_noise < 10:
        print("  ⚠ Very few noise frames! Using frames far from speech.")
        noise_mask = np.ones(n_frames, dtype=bool)
        noise_mask[speech_mask] = False
        if noise_mask.sum() < 10:
            # Force first and last 15% as noise
            n15 = max(10, n_frames // 7)
            noise_mask[:n15] = True
            noise_mask[-n15:] = True
            speech_mask = ~noise_mask

    # ── Step 4: Estimate covariance matrices ─────────────────────────────────
    print("  [4/6] Estimating covariance matrices...")
    Phi_nn = estimate_covariance(X_multi, noise_mask)
    Phi_yy = estimate_covariance(X_multi, speech_mask)

    # Speech covariance: Φ_ss = Φ_yy - Φ_nn (PSD enforced)
    Phi_ss = Phi_yy - Phi_nn

    # Enforce positive semi-definiteness
    for f in range(n_freq):
        eigvals, eigvecs = np.linalg.eigh(Phi_ss[f])
        eigvals = np.maximum(eigvals, 0.0)
        Phi_ss[f] = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
        # Add small regularization
        Phi_ss[f] += EPSILON * np.eye(n_channels, dtype=np.complex128)

    # Diagnostic
    ss_rank = np.linalg.matrix_rank(Phi_ss[n_freq // 4])
    nn_rank = np.linalg.matrix_rank(Phi_nn[n_freq // 4])
    print(f"        Φ_ss rank (f={n_freq//4}): {ss_rank} | Φ_nn rank: {nn_rank}")

    # ── Step 5: Compute beamformer weights ───────────────────────────────────
    print(f"  [5/6] Computing {method.upper()} beamformer weights...")
    if method == "mvdr":
        W = compute_mvdr_weights(Phi_ss, Phi_nn, ref_channel=ref_channel)
    elif method == "gev":
        W = compute_gev_weights(Phi_ss, Phi_nn, ref_channel=ref_channel)
    else:
        W = compute_mwf_weights(Phi_ss, Phi_nn, ref_channel=ref_channel, mu=mu)

    # ── Step 6: Apply beamformer + post-filter + iSTFT ───────────────────────
    print("  [6/6] Applying beamformer + post-filter + iSTFT...")
    Y = apply_beamformer(W, X_multi)

    if postfilter:
        # Stretch mask to match STFT frame count (account for boundary padding)
        pf_mask = speech_mask[:n_frames] if len(speech_mask) >= n_frames else \
            np.pad(speech_mask, (0, n_frames - len(speech_mask)), constant_values=False)
        Y = wiener_postfilter(Y, pf_mask, gain_floor=gain_floor)
        print(f"        Post-filter applied (gain_floor={gain_floor})")

    # Inverse STFT
    clean_signal = compute_istft(Y)

    # Ensure real output
    clean_signal = np.real(clean_signal)

    # Trim to original length
    clean_signal = clean_signal[:min_mic_len]

    # Normalize
    peak = np.abs(clean_signal).max()
    if peak > 0:
        clean_signal = clean_signal / peak * 0.9

    print(f"  ✓ Output: {len(clean_signal)} samples ({len(clean_signal) / SAMPLE_RATE:.2f} s)")
    print(f"{'═' * 60}\n")

    return clean_signal


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Multi-channel Wiener Filter for drone-noise speech enhancement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 3 microphones at 0.8m distance (GEV beamformer - recommended)
  python mwf.py \\
      --vad datasets/noise/drone-sample/vad_result/sp01_man_0.8_mic1_silero.wav \\
      --mics datasets/noise/drone-sample/man/sp01_man_0.8_mic1.wav \\
            datasets/noise/drone-sample/man/sp01_man_0.8_mic2.wav \\
            datasets/noise/drone-sample/man/sp01_man_0.8_mic3.wav \\
      -o sp01_man_0.8_clean.wav

  # Use MWF with custom mu
  python mwf.py \\
      --vad vad_result.wav \\
      --mics mic1.wav mic2.wav mic3.wav \\
      --method mwf --mu 0.5 -o clean.wav

  # MVDR beamformer
  python mwf.py \\
      --vad vad_result.wav \\
      --mics mic1.wav mic2.wav mic3.wav \\
      --method mvdr -o clean.wav
        """,
    )
    p.add_argument("--vad", required=True,
                   help="Path to VAD result WAV (extracted speech audio)")
    p.add_argument("--mics", nargs="+", required=True,
                   help="Paths to microphone WAV files (2 or more)")
    p.add_argument("-o", "--output", default="mwf_clean.wav",
                   help="Output clean speech WAV path (default: mwf_clean.wav)")
    p.add_argument("--method", choices=["mwf", "mvdr", "gev"], default="gev",
                   help="Beamforming method: 'gev' (default, best for drone noise), 'mwf', or 'mvdr'")
    p.add_argument("--ref", type=int, default=0,
                   help="Reference microphone index, 0-based (default: 0)")
    p.add_argument("--mu", type=float, default=1.0,
                   help="MWF distortion trade-off: 0=max denoise, 1=balanced (default: 1.0)")
    p.add_argument("--no-postfilter", action="store_true",
                   help="Disable Wiener post-filter")
    p.add_argument("--gain-floor", type=float, default=0.08,
                   help="Post-filter minimum gain to prevent musical noise (default: 0.08)")
    p.add_argument("--sr", type=int, default=SAMPLE_RATE,
                   help=f"Target sample rate (default: {SAMPLE_RATE})")
    return p


def main() -> None:
    args = build_parser().parse_args()

    global SAMPLE_RATE
    SAMPLE_RATE = args.sr

    # Validate inputs
    if len(args.mics) < 2:
        print("⚠  Warning: MWF requires at least 2 microphones for spatial filtering.")

    # Load VAD signal
    print(f"📂 Loading VAD result: {args.vad}")
    vad_signal = load_wav(args.vad, target_sr=SAMPLE_RATE)
    print(f"   Duration: {len(vad_signal) / SAMPLE_RATE:.2f} s")

    # Load microphone signals
    mic_signals = []
    for i, mic_path in enumerate(args.mics):
        print(f"📂 Loading mic {i + 1}: {mic_path}")
        mic_sig = load_wav(mic_path, target_sr=SAMPLE_RATE)
        mic_signals.append(mic_sig)
        print(f"   Duration: {len(mic_sig) / SAMPLE_RATE:.2f} s")

    # Run MWF
    clean = multichannel_wiener_filter(
        mic_signals=mic_signals,
        vad_signal=vad_signal,
        method=args.method,
        ref_channel=args.ref,
        mu=args.mu,
        postfilter=not args.no_postfilter,
        gain_floor=args.gain_floor,
    )

    # Save output
    save_wav(args.output, clean, sr=SAMPLE_RATE)
    print(f"✅ Clean speech saved: {args.output}")
    print(f"   Duration: {len(clean) / SAMPLE_RATE:.2f} s")


if __name__ == "__main__":
    main()

