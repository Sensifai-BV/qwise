"""Tests for :meth:`SourceMixer.update_geometry`.

Mirrors ``tests/test_source_mixer_update_geometry.m``. The GUI scene
sliders call this method on every drag; we guarantee that the gains and
fractional delays actually swap in, the per-source history rings resize
without clicking, and a mic-count change is rejected loudly.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.config import default
from backend.core import NMicChangeError, SourceMixer, build_geometry


def test_update_changes_gains() -> None:
    cfg = default()
    geo1 = build_geometry(cfg)
    mixer = SourceMixer(cfg, geo1)

    cfg2 = default()
    cfg2.slant_dist = 6.0    # human further → smaller speech gain
    geo2 = build_geometry(cfg2)
    mixer.update_geometry(geo2)

    ref0 = cfg.mwf.ref_mic - 1
    g_old = float(geo1.gains_speech[ref0])
    g_new = float(geo2.gains_speech[ref0])
    assert g_new < g_old, "speech gain must drop when the speaker moves farther"

    # Next block must actually use the new gain.
    n = cfg.frame_size
    rng = np.random.default_rng(1)
    src = rng.standard_normal(n)
    out = mixer.mix(src, np.zeros(n), np.zeros(n))
    m0 = int(np.argmin(geo2.frac_delays_speech))
    ratio = float(np.max(np.abs(out[:, m0])) / np.max(np.abs(src)))
    assert ratio == pytest.approx(g_new, abs=1e-6)


def test_update_resizes_history_without_click() -> None:
    """Prime the history, then shrink (slant 0.5 m) and grow (slant 8.5 m)
    the geometry. Output stays finite and bounded across the boundary."""
    cfg = default()
    mixer = SourceMixer(cfg, build_geometry(cfg))
    n = cfg.frame_size
    s = np.sin(2 * np.pi * 250 * np.arange(n) / cfg.fs)
    mixer.mix(s, np.zeros(n), np.zeros(n))    # prime

    cfg_close = default()
    cfg_close.slant_dist = 0.5
    mixer.update_geometry(build_geometry(cfg_close))
    out_small = mixer.mix(s, np.zeros(n), np.zeros(n))
    assert np.all(np.isfinite(out_small)), "history-shrink produced NaN/Inf"
    assert float(np.max(np.abs(out_small))) <= 2.0, "history-shrink blew up"

    cfg_far = default()
    cfg_far.slant_dist = 8.5
    mixer.update_geometry(build_geometry(cfg_far))
    out_big = mixer.mix(s, np.zeros(n), np.zeros(n))
    assert np.all(np.isfinite(out_big)), "history-grow produced NaN/Inf"
    assert float(np.max(np.abs(out_big))) <= 2.0, "history-grow blew up"


def test_update_rejects_n_mic_change() -> None:
    cfg = default()
    mixer = SourceMixer(cfg, build_geometry(cfg))
    cfg2 = default()
    cfg2.n_mics = cfg.n_mics + 2
    geo2 = build_geometry(cfg2)
    with pytest.raises(NMicChangeError):
        mixer.update_geometry(geo2)
