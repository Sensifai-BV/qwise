"""Tests for :mod:`backend.config.default`.

Mirrors ``acoustic_simulator/tests/test_config.m`` plus checks for the
UI sidebar contract: exactly the set of fields the project owner asked
to be UI-editable must be tagged ``ui``-visible, and the canonical
section order must hold.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.config import (
    Config,
    DroneCfg,
    EnvCfg,
    MixerCfg,
    MwfCfg,
    RecordCfg,
    UiCfg,
    VadCfg,
    default,
    ui_sidebar_schema,
)


# --------------------------------------------------------------------- #
# Field-existence + default-value checks (mirrors test_config.m)
# --------------------------------------------------------------------- #
def test_default_returns_a_fully_populated_config() -> None:
    cfg = default()
    assert isinstance(cfg, Config)
    # spot-check defaults straight from default.m
    assert cfg.fs == 16000
    assert cfg.frame_size == 1024
    assert cfg.c == 343.0
    assert cfg.n_mics == 3
    assert cfg.mic_spacing == 0.10
    assert cfg.mic_geometry == "linear"
    assert cfg.human_height == 1.70
    assert cfg.slant_dist == 2.50
    assert cfg.elev_deg == 30.0
    assert cfg.distance_ref == 1.0


def test_nested_section_types() -> None:
    """Every nested section must be its dedicated submodel — not a dict
    or plain class — so live edits go through Pydantic validation."""
    cfg = default()
    assert isinstance(cfg.drone, DroneCfg)
    assert isinstance(cfg.env, EnvCfg)
    assert isinstance(cfg.mixer, MixerCfg)
    assert isinstance(cfg.vad, VadCfg)
    assert isinstance(cfg.mwf, MwfCfg)
    assert isinstance(cfg.record, RecordCfg)
    assert isinstance(cfg.ui, UiCfg)


def test_mouth_height_is_derived_from_human_height() -> None:
    cfg = default()
    assert cfg.mouth_height == pytest.approx(0.88 * cfg.human_height)
    cfg.human_height = 1.80
    assert cfg.mouth_height == pytest.approx(0.88 * 1.80)


def test_waveform_span_defaults_to_frame_size() -> None:
    """Mirrors MATLAB ``cfg.ui.waveform_span = cfg.frame_size``."""
    cfg = default()
    assert cfg.ui.waveform_span == cfg.frame_size


def test_mixer_defaults_are_sane() -> None:
    cfg = default()
    assert cfg.mixer.mode == "physical"
    assert cfg.mixer.composite == "mic1"


def test_mwf_method_default_is_gev() -> None:
    """Mirrors the bug-fix where the MATLAB default was once 'gav'."""
    cfg = default()
    assert cfg.mwf.method == "gev"
    assert cfg.mwf.passthrough is False


def test_env_section_defaults() -> None:
    cfg = default()
    assert cfg.env.distance_from_mouth == 8.0
    assert cfg.env.azimuth_deg == 135.0
    assert cfg.env.elevation_deg == 0.0


def test_drone_section_defaults() -> None:
    cfg = default()
    assert cfg.drone.azimuth_deg == 0.0
    assert cfg.drone_rpm == 8000.0
    assert cfg.drone_blades == 3


# --------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------- #
def test_validation_rejects_zero_mics() -> None:
    with pytest.raises(ValidationError):
        Config(n_mics=0)


def test_validation_rejects_bad_geometry() -> None:
    with pytest.raises(ValidationError):
        Config(mic_geometry="diagonal")  # type: ignore[arg-type]


def test_validation_rejects_unknown_mwf_method() -> None:
    cfg = default()
    with pytest.raises(ValidationError):
        cfg.mwf.method = "gav"   # historical typo we explicitly disallow


def test_validation_rejects_extra_fields() -> None:
    """``extra='forbid'`` — typos must raise at construction time."""
    with pytest.raises(ValidationError):
        Config(mic_model="MX2H3")  # type: ignore[call-arg]


def test_validation_assignment_revalidates() -> None:
    cfg = default()
    with pytest.raises(ValidationError):
        cfg.n_mics = -1
    # Out-of-range Field constraints fire on the nested model too.
    with pytest.raises(ValidationError):
        cfg.env.azimuth_deg = 9999


# --------------------------------------------------------------------- #
# UI sidebar contract
# --------------------------------------------------------------------- #
EXPECTED_UI_KEYS: set[str] = {
    "n_mics",
    "mic_spacing",
    "mic_geometry",
    "human_height",
    "slant_dist",
    "elev_deg",
    "env.distance_from_mouth",
    "env.azimuth_deg",
    "env.elevation_deg",
    "drone_rpm",
    "drone_blades",
    "ground_R",
    "alpha_air_dB",
    "distance_ref",
    "speech_gain_init",
    "drone_gain_init",
    "env_gain_init",
    "mwf.method",
}


def test_ui_sidebar_exposes_exact_field_set() -> None:
    """The accordion may only contain the keys the project owner approved.

    If this fails the diff tells you which field crept in or fell out —
    don't bypass without reviewing with the project owner.
    """
    entries = ui_sidebar_schema(default())
    keys = {e["key"] for e in entries}
    assert keys == EXPECTED_UI_KEYS, (
        f"unexpected:{keys - EXPECTED_UI_KEYS}, "
        f"missing:{EXPECTED_UI_KEYS - keys}"
    )


def test_ui_sidebar_sections_are_canonical_and_ordered() -> None:
    entries = ui_sidebar_schema(default())
    sections = [e["section"] for e in entries]
    canonical_order = [
        "Microphones",
        "Scene",
        "Drone source",
        "Environment",
        "Acoustics",
        "Gains",
        "MWF",
    ]
    # Each section appears, in canonical order, with no interleaving.
    seen: list[str] = []
    for s in sections:
        if not seen or seen[-1] != s:
            seen.append(s)
    assert seen == canonical_order


def test_ui_sidebar_entries_carry_values_and_widget_hints() -> None:
    cfg = default()
    entries = {e["key"]: e for e in ui_sidebar_schema(cfg)}
    # Sliders carry min/max/step.
    n_mics = entries["n_mics"]
    assert n_mics["widget"] == "slider"
    assert n_mics["value"] == cfg.n_mics
    assert {"min", "max", "step"}.issubset(n_mics)
    # Selects carry options.
    method = entries["mwf.method"]
    assert method["widget"] == "select"
    assert method["options"] == ["gev", "mwf", "mvdr"]
    assert method["value"] == cfg.mwf.method


def test_config_serialises_to_dict_for_the_api() -> None:
    """The FastAPI endpoint will return ``cfg.model_dump()`` — verify it
    round-trips through JSON without losing the derived mouth_height."""
    cfg = default()
    dumped = cfg.model_dump()
    assert dumped["fs"] == 16000
    assert dumped["mouth_height"] == pytest.approx(0.88 * cfg.human_height)
    # Round-trip back into a Config (excluding the computed field).
    dumped.pop("mouth_height")
    Config.model_validate(dumped)
