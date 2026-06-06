"""Tests for :mod:`backend.core.geometry`.

Mirrors ``acoustic_simulator/tests/test_geometry.m``. The acceptance
contract is unchanged: every numeric output (positions, distances,
delays, gains, grazing angle) must match the MATLAB reference to
double-precision tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.config import default
from backend.core import Geometry, build_geometry, build_mic_array, spherical_offset


ATOL = 1e-9


# --------------------------------------------------------------------------- #
# Positions + shapes
# --------------------------------------------------------------------------- #
def test_positions_and_shapes() -> None:
    cfg = default()
    geo = build_geometry(cfg)
    assert isinstance(geo, Geometry)
    assert geo.pos_mics.shape == (cfg.n_mics, 3)
    np.testing.assert_allclose(geo.pos_human, [0.0, 0.0, cfg.mouth_height], atol=ATOL)
    assert geo.pos_img_src[2] == pytest.approx(-cfg.mouth_height, abs=ATOL)
    assert geo.delays.shape == (cfg.n_mics,)
    assert geo.drone_agl == pytest.approx(geo.pos_drone[2], abs=ATOL)


def test_delays_are_nonneg_and_min_zero() -> None:
    cfg = default()
    geo = build_geometry(cfg)
    assert geo.delays.min() >= 0
    assert geo.delays.min() == 0


def test_drone_distance_matches_slant() -> None:
    cfg = default()
    geo = build_geometry(cfg)
    d = float(np.linalg.norm(geo.pos_drone - geo.pos_human))
    assert d == pytest.approx(cfg.slant_dist, abs=1e-6)


def test_env_position_comes_from_cfg() -> None:
    """Env source sits at the spherical position described by cfg.env.*.

    Direct port of ``test_env_position_comes_from_cfg`` in the MATLAB
    suite — uses azimuth=90° (so the offset lands on +y) to make the
    expected Cartesian coordinates explicit.
    """
    cfg = default()
    cfg.env.distance_from_mouth = 6.0
    cfg.env.azimuth_deg = 90.0
    cfg.env.elevation_deg = 0.0
    geo = build_geometry(cfg)
    expected = np.array([0.0, 6.0, cfg.mouth_height])
    np.testing.assert_allclose(geo.pos_env, expected, atol=ATOL)
    # Legacy alias.
    np.testing.assert_allclose(geo.pos_env_noise, expected, atol=ATOL)


def test_mic_spacing_matches_cfg_linear() -> None:
    cfg = default()
    geo = build_geometry(cfg)
    d12 = float(np.linalg.norm(geo.pos_mics[0] - geo.pos_mics[1]))
    assert d12 == pytest.approx(cfg.mic_spacing, abs=ATOL)


# --------------------------------------------------------------------------- #
# Per-source delay / gain contracts
# --------------------------------------------------------------------------- #
def test_per_source_delays_exist() -> None:
    cfg = default()
    geo = build_geometry(cfg)
    for name in ("delays_speech", "delays_drone", "delays_env"):
        arr = getattr(geo, name)
        assert arr.shape == (cfg.n_mics,), name
        assert arr.dtype.kind == "i"
        assert arr.min() == 0


def test_per_source_frac_delays_exist_and_match_round() -> None:
    cfg = default()
    geo = build_geometry(cfg)
    for name in ("frac_delays_speech", "frac_delays_drone", "frac_delays_env"):
        arr = getattr(geo, name)
        assert arr.shape == (cfg.n_mics,), name
        assert arr.dtype.kind == "f"
        assert arr.min() >= 0
        assert arr.min() == pytest.approx(0.0, abs=1e-12)
    # Integer arrays must equal round-to-int of the fractional arrays.
    np.testing.assert_array_equal(
        np.rint(geo.frac_delays_speech).astype(np.int64), geo.delays_speech
    )
    np.testing.assert_array_equal(
        np.rint(geo.frac_delays_drone).astype(np.int64), geo.delays_drone
    )
    np.testing.assert_array_equal(
        np.rint(geo.frac_delays_env).astype(np.int64), geo.delays_env
    )


def test_per_source_gains_obey_inverse_distance() -> None:
    """Gain[m] == d_ref / max(d[m], d_ref) per source. Exact equality."""
    cfg = default()
    geo = build_geometry(cfg)
    # Speech gain must follow the 1/r law clamped at d_ref to machine ε.
    expected = cfg.distance_ref / np.maximum(geo.dist_speech, cfg.distance_ref)
    np.testing.assert_allclose(geo.gains_speech, expected, atol=1e-12)
    # No gain ever exceeds 1.
    assert geo.gains_speech.max() <= 1 + 1e-12

    # Ref-mic gain matches the 1/r law evaluated at the ref mic's actual
    # distance — NOT against 1/slant_dist, because for wider arrays the
    # ref mic does not sit exactly at slant_dist from the speaker.
    ref = cfg.mwf.ref_mic - 1  # MATLAB → Python 0-based index
    g_ref = cfg.distance_ref / max(float(geo.dist_speech[ref]), cfg.distance_ref)
    assert geo.gains_speech[ref] == pytest.approx(g_ref, abs=1e-12)

    # Drone is co-located with the array centroid → at least one mic
    # clamps to 1.0 (its distance is exactly d_ref before the inner mic
    # tilt shrinks it further, depending on geometry).
    assert geo.gains_drone.max() == pytest.approx(1.0, abs=1e-9)

    # Env source is 8 m away → all gains < 0.5.
    assert geo.gains_env.max() < 0.5


# --------------------------------------------------------------------------- #
# Circular geometry
# --------------------------------------------------------------------------- #
def test_circular_geometry_chord_spacing() -> None:
    cfg = default()
    cfg.mic_geometry = "circular"
    cfg.n_mics = 4
    geo = build_geometry(cfg)
    assert geo.pos_mics.shape == (4, 3)
    d12 = float(np.linalg.norm(geo.pos_mics[0] - geo.pos_mics[1]))
    assert d12 == pytest.approx(cfg.mic_spacing, abs=1e-6)


def test_circular_geometry_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError):
        build_mic_array((0.0, 0.0, 0.0), 4, 0.1, "diagonal")


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def test_spherical_offset_cardinal_directions() -> None:
    """+x along az=0 / el=0, +y along az=90 / el=0, +z along el=90."""
    np.testing.assert_allclose(spherical_offset(1.0, 0.0, 0.0), [1.0, 0.0, 0.0], atol=ATOL)
    np.testing.assert_allclose(spherical_offset(1.0, 90.0, 0.0), [0.0, 1.0, 0.0], atol=ATOL)
    np.testing.assert_allclose(spherical_offset(2.5, 0.0, 90.0), [0.0, 0.0, 2.5], atol=ATOL)
    # 45/45 — sanity that distance is preserved.
    p = spherical_offset(2.0, 45.0, 30.0)
    assert float(np.linalg.norm(p)) == pytest.approx(2.0, abs=1e-12)


def test_build_mic_array_linear_centred_on_origin() -> None:
    """For n=5 / spacing=0.2 / centre=(0,0,0) the mic xs are -0.4..0.4."""
    pos = build_mic_array((0.0, 0.0, 0.0), 5, 0.2, "linear")
    np.testing.assert_allclose(
        pos[:, 0], np.array([-0.4, -0.2, 0.0, 0.2, 0.4]), atol=ATOL
    )
    np.testing.assert_allclose(pos[:, 1], np.zeros(5), atol=ATOL)
    np.testing.assert_allclose(pos[:, 2], np.zeros(5), atol=ATOL)


def test_legacy_aliases_are_views_not_copies() -> None:
    """``geo.delays`` and ``geo.pos_env_noise`` must be the same array
    object as the canonical fields — keeps memory usage predictable for
    callers that follow the MATLAB struct field names."""
    geo = build_geometry(default())
    assert geo.delays is geo.delays_speech
    assert geo.pos_env_noise is geo.pos_env
