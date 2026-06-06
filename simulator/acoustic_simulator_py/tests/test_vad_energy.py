"""Tests for :class:`backend.vad.EnergyVad`.

Direct port of ``tests/test_vad_energy.m``: silence is not speech, a
voiced-speech proxy is detected, the hangover release keeps the
``state_speech`` flag high for one frame after a quiet drop.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.config import default
from backend.vad import EnergyVad


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_silence_is_not_speech(rng) -> None:
    cfg = default()
    v = EnergyVad(cfg)
    n = cfg.frame_size
    for _ in range(5):
        is_speech, score = v.step(1e-6 * rng.standard_normal(n))
        assert is_speech is False
        assert score < 0.5


def test_tonal_speech_proxy_is_detected() -> None:
    """A formant-like sinusoid mix + light noise — the same fixture as the
    MATLAB suite. Must trigger within six frames."""
    cfg = default()
    v = EnergyVad(cfg)
    n = cfg.frame_size
    t = np.arange(n) / cfg.fs
    sig = (
        0.30 * np.sin(2 * np.pi * 220 * t)
        + 0.18 * np.sin(2 * np.pi * 480 * t)
        + 0.12 * np.sin(2 * np.pi * 920 * t)
        + 0.02 * np.random.default_rng(1).standard_normal(n)
    )
    detected = False
    for _ in range(6):
        if v.step(sig)[0]:
            detected = True
            break
    assert detected, "Energy+SFM VAD must detect the voiced-speech proxy."


def test_hangover_smooths_brief_drops() -> None:
    """After a stretch of voiced frames, the first quiet frame within
    the hangover window must still report speech."""
    cfg = default()
    cfg.vad.hang_frames = 5
    v = EnergyVad(cfg)
    n = cfg.frame_size
    t = np.arange(n) / cfg.fs
    loud = 0.30 * np.sin(2 * np.pi * 220 * t) + 0.20 * np.sin(2 * np.pi * 480 * t)
    quiet = 1e-6 * np.random.default_rng(2).standard_normal(n)
    for _ in range(4):
        v.step(loud)
    sp, _ = v.step(quiet)
    assert sp, "hangover should keep the speech state high for one quiet frame"


def test_reset_clears_score_and_state() -> None:
    cfg = default()
    v = EnergyVad(cfg)
    n = cfg.frame_size
    t = np.arange(n) / cfg.fs
    loud = 0.30 * np.sin(2 * np.pi * 220 * t) + 0.20 * np.sin(2 * np.pi * 480 * t)
    for _ in range(3):
        v.step(loud)
    v.reset()
    # The next quiet frame returns to silence + score == 0 (no EMA carry-over).
    sp, score = v.step(1e-6 * np.random.default_rng(3).standard_normal(n))
    assert sp is False
    assert score == pytest.approx(0.0, abs=1e-12)
