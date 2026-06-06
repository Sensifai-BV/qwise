"""Tests for the batch MWF pipeline.

Direct port of ``tests/test_mwf_batch.m``. Exercises the full
``mwf.py``-equivalent flow on synthetic 3-mic data plus the individual
kernels (STFT round-trip, VAD alignment, speech-mask, covariance
estimate, weight shapes, batch process matrix + list inputs, postfilter
toggle, method dispatch).
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.config import default
from backend.mwf import (
    Mwf,
    align_vad,
    apply_beamformer,
    build_speech_mask,
    compute_gev_weights,
    compute_mvdr_weights,
    compute_mwf_weights,
    estimate_covariance,
    istft,
    stft,
)


# --------------------------------------------------------------------- #
# Fixtures + helpers
# --------------------------------------------------------------------- #
def _synth_scene(fs: int, T: float):
    """Deterministic 3-mic scene with a 350 Hz speech burst + noise."""
    rng = np.random.default_rng(42)
    N = int(round(fs * T))
    t = np.arange(N) / fs

    speech_burst = np.sin(2 * np.pi * 350 * t)
    gate = np.zeros(N, dtype=bool)
    gate[int(0.2 * N) : int(0.8 * N)] = True
    speech = speech_burst * gate

    noise = 0.3 * rng.standard_normal((N, 3))
    mic = np.column_stack([speech, 0.6 * speech, 0.3 * speech]) + noise
    return speech, mic


def _make_psd(n_ch: int, n_freq: int):
    rng = np.random.default_rng(7)
    Phi = np.zeros((n_freq, n_ch, n_ch), dtype=np.complex128)
    for f in range(n_freq):
        A = rng.standard_normal((n_ch, n_ch)) + 1j * rng.standard_normal((n_ch, n_ch))
        Phi[f] = A @ A.conj().T + 0.01 * np.eye(n_ch)
    return Phi


# --------------------------------------------------------------------- #
# STFT round-trip
# --------------------------------------------------------------------- #
def test_stft_istft_round_trip() -> None:
    rng = np.random.default_rng(0)
    fs = 16000
    x = rng.standard_normal(fs)
    n_fft, hop = 1024, 256
    X = stft(x, n_fft, hop)
    y = istft(X, n_fft, hop)
    L = min(x.size, y.size)
    err = np.linalg.norm(x[:L] - y[:L]) / np.linalg.norm(x[:L])
    assert err < 1e-6


# --------------------------------------------------------------------- #
# VAD alignment
# --------------------------------------------------------------------- #
def test_align_vad_finds_known_offset() -> None:
    rng = np.random.default_rng(1)
    fs = 16000
    s = np.sin(2 * np.pi * 440 * np.arange(fs // 2) / fs)
    lag0 = 4000
    mic = np.concatenate(
        [0.05 * rng.standard_normal(lag0), s, 0.05 * rng.standard_normal(fs)]
    )
    lag, aligned = align_vad(s, mic)
    assert abs(lag - lag0) <= 50          # cross-corr within ±3 ms
    assert aligned.shape == mic.shape


# --------------------------------------------------------------------- #
# Speech mask
# --------------------------------------------------------------------- #
def test_speech_mask_finds_speech_frames() -> None:
    rng = np.random.default_rng(2)        # not actually used — keeps determinism note
    fs = 16000
    T = 1.0
    N = int(round(fs * T))
    t = np.arange(N) / fs
    aligned_vad = np.zeros(N)
    a, b = int(0.3 * N), int(0.7 * N)
    aligned_vad[a:b] = np.sin(2 * np.pi * 350 * t[a:b])

    n_fft, hop = 1024, 256
    n_frames = (N + n_fft) // hop
    mask = build_speech_mask(aligned_vad, n_frames, n_fft, hop, 0.01, 3)
    assert mask.shape == (n_frames,)
    assert int(mask.sum()) > int(0.25 * n_frames)
    assert int((~mask).sum()) > int(0.25 * n_frames)
    _ = rng    # silence "unused" if pytest greps for fixtures


# --------------------------------------------------------------------- #
# Covariance estimate
# --------------------------------------------------------------------- #
def test_covariance_is_hermitian_psd() -> None:
    rng = np.random.default_rng(3)
    n_ch, n_freq, n_frames = 3, 17, 40
    X = rng.standard_normal((n_ch, n_freq, n_frames)) + 1j * rng.standard_normal(
        (n_ch, n_freq, n_frames)
    )
    mask = np.ones(n_frames, dtype=bool)
    Phi = estimate_covariance(X, mask, 1e-10)
    assert Phi.shape == (n_freq, n_ch, n_ch)
    for f in (0, n_freq // 2, n_freq - 1):
        M = Phi[f]
        assert np.linalg.norm(M - M.conj().T, "fro") < 1e-9
        d = np.linalg.eigvalsh((M + M.conj().T) / 2)
        assert float(d.real.min()) >= -1e-9


# --------------------------------------------------------------------- #
# Beamformer weight shapes
# --------------------------------------------------------------------- #
def test_each_method_returns_expected_shape() -> None:
    n_ch, n_freq = 3, 11
    Phi_ss = _make_psd(n_ch, n_freq)
    Phi_nn = _make_psd(n_ch, n_freq)

    W_mwf = compute_mwf_weights(Phi_ss, Phi_nn, 1, 1.0, 1e-10, 1e-4)
    W_mvdr = compute_mvdr_weights(Phi_ss, Phi_nn, 1, 1e-10, 1e-4)
    W_gev = compute_gev_weights(Phi_ss, Phi_nn, 1, 1e-10, 1e-4)

    for W in (W_mwf, W_mvdr, W_gev):
        assert W.shape == (n_freq, n_ch)
        assert np.all(np.isfinite(W))


def test_apply_beamformer_shape() -> None:
    n_ch, n_freq, n_frames = 3, 11, 20
    W = np.ones((n_freq, n_ch), dtype=np.complex128)
    X = np.ones((n_ch, n_freq, n_frames), dtype=np.complex128)
    Y = apply_beamformer(W, X)
    assert Y.shape == (n_freq, n_frames)
    # W^H · X with all-ones gives n_ch * 1.0 per (f, t).
    np.testing.assert_allclose(Y, float(n_ch), atol=1e-12)


# --------------------------------------------------------------------- #
# Batch ``process`` end-to-end
# --------------------------------------------------------------------- #
def test_process_matrix_input_returns_matching_length() -> None:
    cfg = default()
    cfg.mwf.method = "gev"
    cfg.mwf.postfilter = True
    vad_audio, mic_mat = _synth_scene(cfg.fs, 1.0)
    y = Mwf(cfg).process(vad_audio, mic_mat)
    assert y.ndim == 1
    assert y.size <= mic_mat.shape[0]
    assert np.all(np.isfinite(y))


def test_process_list_input_truncates_to_shortest() -> None:
    cfg = default()
    cfg.mwf.method = "gev"
    vad_audio, mic_mat = _synth_scene(cfg.fs, 1.0)
    mic_list = [mic_mat[:, 0], mic_mat[:-100, 1], mic_mat[:, 2]]
    y = Mwf(cfg).process(vad_audio, mic_list)
    assert y.ndim == 1
    assert np.all(np.isfinite(y))


def test_method_dispatch_runs_all_three() -> None:
    fs = default().fs
    vad_audio, mic_signals = _synth_scene(fs, 0.8)
    for method in ("gev", "mwf", "mvdr"):
        cfg = default()
        cfg.mwf.method = method
        y = Mwf(cfg).process(vad_audio, mic_signals)
        assert np.all(np.isfinite(y)), f"{method} produced non-finite samples"
        assert float(np.max(np.abs(y))) > 0


def test_postfilter_toggle_changes_output() -> None:
    fs = default().fs
    vad_audio, mic_signals = _synth_scene(fs, 0.8)

    cfg1 = default()
    cfg1.mwf.method = "gev"
    cfg1.mwf.postfilter = True
    cfg2 = default()
    cfg2.mwf.method = "gev"
    cfg2.mwf.postfilter = False
    y1 = Mwf(cfg1).process(vad_audio, mic_signals)
    y2 = Mwf(cfg2).process(vad_audio, mic_signals)
    L = min(y1.size, y2.size)
    assert np.linalg.norm(y1[:L] - y2[:L]) > 1e-6
