"""Tests for :mod:`backend.core.source_mixer`.

Direct port of ``tests/test_source_mixer.m``. The contract is the same
as the MATLAB suite: every microphone receives all three sources with a
fractional sample delay and a 1/r spreading gain clamped at
``cfg.distance_ref``. The legacy perChannel wiring is gone — there is
only one physical model now.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.config import default
from backend.core import SourceMixer, build_geometry


# --------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------- #
@pytest.fixture
def cfg():
    """Default config with mic-1 composite (matches the MATLAB suite)."""
    c = default()
    c.mixer.composite = "mic1"
    return c


@pytest.fixture
def geo(cfg):
    return build_geometry(cfg)


@pytest.fixture
def rng():
    """Deterministic generator so failures are reproducible."""
    return np.random.default_rng(0)


# --------------------------------------------------------------------- #
# Shape + first-block correctness
# --------------------------------------------------------------------- #
def test_output_shape(cfg, geo, rng) -> None:
    mixer = SourceMixer(cfg, geo)
    n = cfg.frame_size
    out = mixer.mix(rng.standard_normal(n), rng.standard_normal(n), rng.standard_normal(n))
    assert out.shape == (n, cfg.n_mics)


def test_drone_only_gain_is_unity_on_closest(cfg, geo) -> None:
    """At the mic with frac-delay 0 for the drone, the peak ratio must
    equal ``geo.gains_drone`` to machine precision (no interpolation)."""
    mixer = SourceMixer(cfg, geo)
    n = cfg.frame_size
    src = np.sin(2 * np.pi * 440 * np.arange(n) / cfg.fs)
    out = mixer.mix(np.zeros(n), src, np.zeros(n))
    m0 = int(np.argmin(geo.frac_delays_drone))
    assert geo.frac_delays_drone[m0] == pytest.approx(0.0, abs=1e-12)
    expected = geo.gains_drone[m0]
    pk = float(np.max(np.abs(out[:, m0])) / np.max(np.abs(src)))
    assert pk == pytest.approx(expected, abs=1e-9)


def test_speech_attenuates_at_distance(cfg, geo) -> None:
    """At the default 2.5 m slant the ref-mic gain must be < 0.5 and
    match the geometry prediction."""
    mixer = SourceMixer(cfg, geo)
    n = cfg.frame_size
    src = 0.7 * np.sin(2 * np.pi * 200 * np.arange(n) / cfg.fs)
    out = mixer.mix(src, np.zeros(n), np.zeros(n))
    m0 = int(np.argmin(geo.frac_delays_speech))
    ratio = float(np.max(np.abs(out[:, m0])) / np.max(np.abs(src)))
    assert ratio == pytest.approx(geo.gains_speech[m0], abs=1e-6)
    assert ratio < 0.5


def test_speech_impulse_smears_across_fractional_tap(cfg, geo) -> None:
    """A unit impulse on speech must split across ``floor(tau)`` and
    ``floor(tau)+1`` with weights ``(1-frac)`` and ``frac``.

    Indices below are 0-based, so the bin that the MATLAB suite called
    ``i_a = floor(tau)+1`` is ``floor(tau)`` here.
    """
    mixer = SourceMixer(cfg, geo)
    n = cfg.frame_size
    src = np.zeros(n)
    src[0] = 1.0
    out = mixer.mix(src, np.zeros(n), np.zeros(n))

    for m in range(cfg.n_mics):
        tau = float(geo.frac_delays_speech[m])
        g = float(geo.gains_speech[m])
        if tau >= n - 1:
            continue
        i_a = int(np.floor(tau))         # 0-based — MATLAB's floor(tau)+1
        i_b = i_a + 1
        frac = tau - np.floor(tau)
        assert out[i_a, m] == pytest.approx(g * (1.0 - frac), abs=1e-9), (
            f"mic {m + 1}: leading bin mismatch at tau={tau:.4f}"
        )
        if i_b < n:
            assert out[i_b, m] == pytest.approx(g * frac, abs=1e-9), (
                f"mic {m + 1}: trailing bin mismatch at tau={tau:.4f}"
            )
        # Every leading sample must be exactly zero.
        if i_a > 0:
            np.testing.assert_allclose(out[:i_a, m], 0.0, atol=1e-12)


def test_history_carries_across_blocks(cfg, geo) -> None:
    """Impulse at the last sample of block 1 must reappear at the
    correct fractional position inside block 2, proving the history
    ring stitches blocks together cleanly."""
    mixer = SourceMixer(cfg, geo)
    n = cfg.frame_size
    src1 = np.zeros(n)
    src1[-1] = 1.0
    src2 = np.zeros(n)
    mixer.mix(src1, np.zeros(n), np.zeros(n))
    out2 = mixer.mix(src2, np.zeros(n), np.zeros(n))

    for m in range(cfg.n_mics):
        tau = float(geo.frac_delays_speech[m])
        g = float(geo.gains_speech[m])
        if tau < 1 or tau >= n:
            continue
        # MATLAB used i_a = floor(tau) (1-based); in 0-based that is floor(tau)-1.
        i_a = int(np.floor(tau)) - 1
        i_b = i_a + 1
        frac = tau - np.floor(tau)
        if i_a >= 0:
            assert out2[i_a, m] == pytest.approx(g * (1.0 - frac), abs=1e-9), (
                f"mic {m + 1}: block-2 leading bin mismatch"
            )
        if i_b < n:
            assert out2[i_b, m] == pytest.approx(g * frac, abs=1e-9), (
                f"mic {m + 1}: block-2 trailing bin mismatch"
            )


# --------------------------------------------------------------------- #
# Algebraic properties
# --------------------------------------------------------------------- #
def test_additivity_of_sources(cfg, geo, rng) -> None:
    """``mix(s, d, e) == mix(s,0,0) + mix(0,d,0) + mix(0,0,e)`` to 1e-10.

    Linear superposition is the easiest fingerprint that no source path
    is leaking into the wrong channel.
    """
    n = cfg.frame_size
    s = rng.standard_normal(n)
    d = rng.standard_normal(n)
    e = rng.standard_normal(n)
    full = SourceMixer(cfg, geo).mix(s, d, e)
    only_s = SourceMixer(cfg, geo).mix(s, np.zeros(n), np.zeros(n))
    only_d = SourceMixer(cfg, geo).mix(np.zeros(n), d, np.zeros(n))
    only_e = SourceMixer(cfg, geo).mix(np.zeros(n), np.zeros(n), e)
    np.testing.assert_allclose(full, only_s + only_d + only_e, atol=1e-10)


def test_every_mic_receives_every_source(cfg, geo, rng) -> None:
    """No mic is noise-only — each of the N channels carries some
    contribution from each source (with their respective gains)."""
    n = cfg.frame_size
    src = rng.standard_normal(n)
    out_s = SourceMixer(cfg, geo).mix(src, np.zeros(n), np.zeros(n))
    out_d = SourceMixer(cfg, geo).mix(np.zeros(n), src, np.zeros(n))
    out_e = SourceMixer(cfg, geo).mix(np.zeros(n), np.zeros(n), src)
    for m in range(cfg.n_mics):
        assert float(np.max(np.abs(out_s[:, m]))) > 0, f"mic {m + 1}: no speech"
        assert float(np.max(np.abs(out_d[:, m]))) > 0, f"mic {m + 1}: no drone"
        assert float(np.max(np.abs(out_e[:, m]))) > 0, f"mic {m + 1}: no env"


# --------------------------------------------------------------------- #
# Composite reductions
# --------------------------------------------------------------------- #
def test_composite_default_is_ref_mic(cfg, geo, rng) -> None:
    mixer = SourceMixer(cfg, geo)
    n = cfg.frame_size
    out = mixer.mix(rng.standard_normal(n), rng.standard_normal(n), rng.standard_normal(n))
    comp = mixer.composite(out)
    ref0 = cfg.mwf.ref_mic - 1
    np.testing.assert_allclose(comp, out[:, ref0], atol=1e-12)


def test_composite_sum_and_mean(cfg, geo, rng) -> None:
    n = cfg.frame_size
    s = rng.standard_normal(n)
    d = rng.standard_normal(n)
    e = rng.standard_normal(n)

    cfg_sum = default()
    cfg_sum.mixer.composite = "sum"
    cfg_mean = default()
    cfg_mean.mixer.composite = "mean"

    m_sum = SourceMixer(cfg_sum, geo)
    m_mean = SourceMixer(cfg_mean, geo)
    out_sum = m_sum.mix(s, d, e)
    out_mean = m_mean.mix(s, d, e)
    np.testing.assert_allclose(m_sum.composite(out_sum), out_sum.sum(axis=1), atol=1e-12)
    np.testing.assert_allclose(m_mean.composite(out_mean), out_mean.mean(axis=1), atol=1e-12)


def test_composite_empty_block_returns_silence(cfg, geo) -> None:
    """Edge case: an empty mic array returns ``zeros(frame_size)``,
    mirroring the MATLAB safety net."""
    mixer = SourceMixer(cfg, geo)
    comp = mixer.composite(np.zeros((0, cfg.n_mics)))
    assert comp.shape == (cfg.frame_size,)
    assert np.all(comp == 0.0)


# --------------------------------------------------------------------- #
# Reset
# --------------------------------------------------------------------- #
def test_reset_clears_history_rings(cfg, geo) -> None:
    mixer = SourceMixer(cfg, geo)
    n = cfg.frame_size
    impulse = np.zeros(n)
    impulse[-1] = 1.0
    mixer.mix(impulse, np.zeros(n), np.zeros(n))
    mixer.reset()
    # After reset the next block must be identically zero (no leakage).
    out = mixer.mix(np.zeros(n), np.zeros(n), np.zeros(n))
    np.testing.assert_array_equal(out, np.zeros((n, cfg.n_mics)))
