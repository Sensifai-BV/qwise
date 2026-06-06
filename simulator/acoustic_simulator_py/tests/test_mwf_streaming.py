"""Tests for the streaming MWF path + bookkeeping.

Mirrors ``tests/test_mwf_passthrough.m`` and adds two extra sanity tests
(streaming output shape + reset restores covariance EMA).
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.config import default
from backend.mwf import BadMethodError, Mwf, get_tf_gain_map


def test_passthrough_returns_reference_mic() -> None:
    cfg = default()
    cfg.mwf.passthrough = True
    m = Mwf(cfg)
    N = cfg.frame_size
    rng = np.random.default_rng(0)
    x = rng.standard_normal((N, cfg.n_mics))
    y = m.step(x, is_speech=False)
    assert y.shape == (N,)
    np.testing.assert_allclose(y, x[:, cfg.mwf.ref_mic - 1], atol=1e-12)


def test_covariance_buffers_initialised() -> None:
    cfg = default()
    m = Mwf(cfg)
    assert m.Rnn.shape == (m.nbin, cfg.n_mics, cfg.n_mics)
    assert m.Rss.shape == (m.nbin, cfg.n_mics, cfg.n_mics)
    np.testing.assert_allclose(
        m.Rnn[0],
        cfg.mwf.eps_reg * np.eye(cfg.n_mics, dtype=np.complex128),
        atol=1e-12,
    )


def test_gain_map_placeholder_is_ones() -> None:
    cfg = default()
    nbin = cfg.mwf.stft_win // 2 + 1
    X = np.random.default_rng(1).standard_normal((nbin, cfg.n_mics))
    G = get_tf_gain_map(X, cfg)
    assert G.shape == (nbin,)
    np.testing.assert_allclose(G, np.ones(nbin), atol=1e-12)


def test_reset_restores_initial_covariances() -> None:
    cfg = default()
    cfg.mwf.passthrough = False
    m = Mwf(cfg)
    N = cfg.frame_size
    rng = np.random.default_rng(2)
    m.step(rng.standard_normal((N, cfg.n_mics)), is_speech=True)
    # Speech step updates Rss away from the eps_reg identity.
    assert not np.allclose(
        m.Rss[0], cfg.mwf.eps_reg * np.eye(cfg.n_mics), atol=1e-12
    )
    m.reset()
    np.testing.assert_allclose(
        m.Rnn[0],
        cfg.mwf.eps_reg * np.eye(cfg.n_mics, dtype=np.complex128),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        m.Rss[0],
        cfg.mwf.eps_reg * np.eye(cfg.n_mics, dtype=np.complex128),
        atol=1e-12,
    )


def test_invalid_method_rejected() -> None:
    cfg = default()
    # Skip Pydantic validation by mutating attribute after construction.
    # ``Config.validate_assignment`` would catch this for normal usage —
    # the test guards the constructor's own defense-in-depth path.
    cfg.mwf.__dict__["method"] = "not-a-method"
    with pytest.raises(BadMethodError):
        Mwf(cfg)


# --------------------------------------------------------------------- #
# Streaming end-to-end (no passthrough)
# --------------------------------------------------------------------- #
def test_streaming_step_shape_and_finite() -> None:
    """When passthrough is off, ``step`` still returns ``[N]`` floats."""
    cfg = default()
    cfg.mwf.passthrough = False
    m = Mwf(cfg)
    N = cfg.frame_size
    rng = np.random.default_rng(3)
    # Two blocks: speech then noise. Covariance EMAs converge gently.
    for is_sp in (True, False):
        y = m.step(rng.standard_normal((N, cfg.n_mics)), is_speech=is_sp)
        assert y.shape == (N,)
        assert np.all(np.isfinite(y))


def test_streaming_speech_updates_rss_not_rnn() -> None:
    """One speech step must shift Rss away from the eps_reg identity
    while Rnn stays untouched (and vice versa)."""
    cfg = default()
    cfg.mwf.passthrough = False
    m = Mwf(cfg)
    N = cfg.frame_size
    rng = np.random.default_rng(4)
    x = rng.standard_normal((N, cfg.n_mics))

    eps_I = cfg.mwf.eps_reg * np.eye(cfg.n_mics, dtype=np.complex128)
    m.step(x, is_speech=True)
    # Rss moves, Rnn doesn't.
    assert not np.allclose(m.Rss[0], eps_I, atol=1e-12)
    np.testing.assert_allclose(m.Rnn[0], eps_I, atol=1e-12)

    m.step(x, is_speech=False)
    assert not np.allclose(m.Rnn[0], eps_I, atol=1e-12)
