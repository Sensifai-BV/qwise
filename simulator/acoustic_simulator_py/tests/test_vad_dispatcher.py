"""Tests for :class:`backend.vad.Vad` (dispatcher).

Covers backend selection (``'auto'`` / ``'qwise'`` / ``'energy'``), the
ring-buffer trace, and the ``reset`` propagation. Neural-backend tests
skip when ``onnxruntime`` or the model file aren't available.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.config import default
from backend.vad import EnergyVad, QwiseVad, Vad


# --------------------------------------------------------------------- #
# Backend selection
# --------------------------------------------------------------------- #
def test_energy_forces_energy_backend() -> None:
    cfg = default()
    cfg.vad.backend = "energy"
    v = Vad(cfg)
    assert isinstance(v.backend, EnergyVad)
    assert v.backend_name == "energy"


def test_auto_picks_qwise_when_available() -> None:
    cfg = default()
    cfg.vad.backend = "auto"
    if not QwiseVad(cfg).ready:
        pytest.skip("onnxruntime or the VAD ONNX model is unavailable")
    v = Vad(cfg)
    assert isinstance(v.backend, QwiseVad)
    assert v.backend_name == "qwise-vad"


def test_qwise_falls_back_to_energy_when_unavailable() -> None:
    """Force the path through QwiseVad with a deliberately broken model
    path; the dispatcher must downgrade to EnergyVad without raising."""
    cfg = default()
    cfg.vad.backend = "qwise"
    cfg.vad.onnx_path = "models/__missing__.onnx"
    v = Vad(cfg)
    assert isinstance(v.backend, EnergyVad)
    assert v.backend_name == "energy"


# --------------------------------------------------------------------- #
# History ring + trace
# --------------------------------------------------------------------- #
def test_hist_len_follows_cfg() -> None:
    """``hist_len = max(8, round(vad_hist_sec * fs / frame_size))``."""
    cfg = default()
    expected = max(8, round(cfg.ui.vad_hist_sec * cfg.fs / cfg.frame_size))
    v = Vad(cfg)
    assert v.hist_len == expected
    assert v.history.shape == (expected,)
    assert v.flags.shape == (expected,)


def test_trace_returns_blocks_in_chronological_order() -> None:
    cfg = default()
    cfg.vad.backend = "energy"
    v = Vad(cfg)
    rng = np.random.default_rng(6)
    n = cfg.frame_size
    # Push three identifiable blocks: silence, voiced, silence.
    t = np.arange(n) / cfg.fs
    voiced = 0.35 * np.sin(2 * np.pi * 220 * t)
    seq = [1e-6 * rng.standard_normal(n), voiced, 1e-6 * rng.standard_normal(n)]
    pushed_scores = []
    for blk in seq:
        _, score = v.step(blk)
        pushed_scores.append(score)

    scores, _ = v.trace()
    # The last three entries of the chronological trace must be the
    # most-recently pushed scores in order.
    np.testing.assert_allclose(scores[-3:], pushed_scores, atol=1e-12)
    # The earlier entries are the zeros that were pre-filled at __init__.
    assert np.all(scores[: v.hist_len - 3] == 0.0)


def test_trace_is_a_copy_not_a_view() -> None:
    """Mutating the returned trace must not corrupt the dispatcher's
    internal ring — the UI may want to scale or smooth the array."""
    cfg = default()
    cfg.vad.backend = "energy"
    v = Vad(cfg)
    v.step(np.zeros(cfg.frame_size))
    scores, _ = v.trace()
    scores[0] = 12345.0
    fresh_scores, _ = v.trace()
    assert fresh_scores[0] != 12345.0


def test_reset_clears_ring_and_resets_backend() -> None:
    cfg = default()
    cfg.vad.backend = "energy"
    v = Vad(cfg)
    n = cfg.frame_size
    t = np.arange(n) / cfg.fs
    loud = 0.30 * np.sin(2 * np.pi * 220 * t) + 0.20 * np.sin(2 * np.pi * 480 * t)
    for _ in range(3):
        v.step(loud)
    v.reset()
    scores, flags = v.trace()
    assert np.all(scores == 0.0)
    assert not flags.any()
