"""Default configuration — Python port of ``config/default.m``.

The MATLAB file is the single source of truth for runtime parameters
(see ``/Users/javad/Projects/qwise/simulator/acoustic_simulator/config/default.m``).
This module mirrors it 1:1 as a Pydantic v2 model so:

* every numeric routine in :mod:`backend.core`, :mod:`backend.vad` and
  :mod:`backend.mwf` reads the same fields with the same units/types as
  the MATLAB code, and
* the FastAPI sidebar can serialise the model to JSON, render an
  accordion, and POST live edits back into the running pipeline.

Field hints for the sidebar live in ``Field(json_schema_extra={"ui": …})``.
Only fields tagged that way appear in :func:`ui_sidebar_schema`; the rest
stay as backend-only knobs.

The subset exposed to the UI (per the project owner's request) is:

* Microphones      — ``n_mics``, ``mic_spacing``, ``mic_geometry``
* Scene            — ``human_height``, ``slant_dist``, ``elev_deg``
* Environment      — ``env.distance_from_mouth``, ``env.azimuth_deg``,
                     ``env.elevation_deg``
* Drone source     — ``drone_rpm``, ``drone_blades``
* Acoustics        — ``ground_R``, ``alpha_air_dB``, ``distance_ref``
* Gains            — ``speech_gain_init``, ``drone_gain_init``,
                     ``env_gain_init``
* MWF              — ``mwf.method``

``mouth_height`` is a *computed* field (0.88 × ``human_height``), so the
sidebar's Human-height slider implicitly drives it; it ships in
serialised JSON but has no dedicated control.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _ui(
    section: str,
    widget: Literal["slider", "select", "number", "toggle"],
    label: str,
    **extra: object,
) -> dict:
    """Build the ``json_schema_extra`` payload for a UI-visible field.

    Storing the metadata under a stable ``"ui"`` namespace keeps it
    out of the way of Pydantic's own JSON-schema keywords and makes the
    sidebar walker trivially detect which fields to render.
    """
    return {"ui": {"section": section, "widget": widget, "label": label, **extra}}


_MODEL_CFG = ConfigDict(
    extra="forbid",            # reject typos / unknown keys (MATLAB has no extras)
    validate_assignment=True,  # live UI edits must re-validate
    str_strip_whitespace=True,
)


# --------------------------------------------------------------------------- #
# Nested sections
# --------------------------------------------------------------------------- #
class DroneCfg(BaseModel):
    """Maps to MATLAB ``cfg.drone`` (orientation/azimuth of the drone source).

    Distance + elevation come from the top-level ``slant_dist`` /
    ``elev_deg`` fields, matching ``build_geometry.m``.
    """

    model_config = _MODEL_CFG

    azimuth_deg: float = Field(
        0.0,
        ge=-180.0,
        le=180.0,
        description="0° puts the drone along +x from the speaker's mouth.",
    )


class EnvCfg(BaseModel):
    """Environmental noise source. Maps to MATLAB ``cfg.env``."""

    model_config = _MODEL_CFG

    distance_from_mouth: float = Field(
        8.0,
        gt=0,
        description="Slant distance from the speaker's mouth, metres.",
        json_schema_extra=_ui(
            "Environment", "slider", "Env distance (m)",
            min=0.5, max=30.0, step=0.1,
        ),
    )
    azimuth_deg: float = Field(
        135.0,
        ge=-180.0,
        le=180.0,
        description="Azimuth of the env source relative to the mouth.",
        json_schema_extra=_ui(
            "Environment", "slider", "Env azimuth (°)",
            min=-180, max=180, step=1,
        ),
    )
    elevation_deg: float = Field(
        0.0,
        ge=-90.0,
        le=90.0,
        description="Elevation of the env source relative to the mouth.",
        json_schema_extra=_ui(
            "Environment", "slider", "Env elevation (°)",
            min=-90, max=90, step=1,
        ),
    )


class MixerCfg(BaseModel):
    """Mixer wiring. Single physical model in the current MATLAB port.

    ``composite`` controls how the N-channel mic block is reduced to one
    mono stream for the VAD (mic1 = reference mic, sum, or mean).
    """

    model_config = _MODEL_CFG

    mode: Literal["physical"] = "physical"
    composite: Literal["mic1", "sum", "mean"] = "mic1"


class VadCfg(BaseModel):
    """Voice activity detection settings. Maps to ``cfg.vad``."""

    model_config = _MODEL_CFG

    backend: Literal["auto", "qwise", "energy"] = "qwise"
    # MATLAB used ``vad/qwise_vad.onnx``; the Python project keeps every
    # ONNX checkpoint under ``models/`` so the layout reads consistently
    # ("models hold ML assets", "wavs hold audio loops").
    onnx_path: str = "models/qwise_vad.onnx"
    qwise_frame: int = Field(512, ge=1)
    qwise_threshold: float = Field(0.50, ge=0.0, le=1.0)
    energy_threshold: float = -45.0  # dBFS
    sfm_threshold: float = Field(0.45, ge=0.0, le=1.0)
    hang_frames: int = Field(8, ge=0)
    smoothing: float = Field(0.30, ge=0.0, le=1.0)


class MwfCfg(BaseModel):
    """Multi-channel Wiener filter settings. Maps to ``cfg.mwf``."""

    model_config = _MODEL_CFG

    enabled: bool = True
    method: Literal["gev", "mwf", "mvdr"] = Field(
        "gev",
        description="Beamformer kernel. GEV is the project default.",
        json_schema_extra=_ui(
            "MWF", "select", "MWF method",
            options=["gev", "mwf", "mvdr"],
        ),
    )
    # Batch geometry (matches the reference Python pipeline).
    n_fft: int = Field(1024, ge=8)
    hop: int = Field(256, ge=1)
    # Streaming geometry.
    stft_win: int = Field(512, ge=8)
    stft_hop: int = Field(256, ge=1)
    ref_mic: int = Field(1, ge=1)
    mu: float = Field(1.0, ge=0.0)
    eps_reg: float = Field(1e-10, ge=0.0)
    diag_load_ratio: float = Field(1e-4, ge=0.0)
    alpha_nn: float = Field(0.92, ge=0.0, le=1.0)
    alpha_ss: float = Field(0.88, ge=0.0, le=1.0)
    postfilter: bool = True
    gain_floor: float = Field(0.08, ge=0.0, le=1.0)
    noise_floor_alpha: float = Field(0.98, ge=0.0, le=1.0)
    pf_smooth_kernel: int = Field(3, ge=1)
    mask_threshold: float = Field(0.01, ge=0.0)
    mask_context: int = Field(3, ge=0)
    passthrough: bool = False


class RecordCfg(BaseModel):
    """Recording session paths. Maps to ``cfg.record``."""

    model_config = _MODEL_CFG

    dir: str = "recordings"
    prefix: str = "qwise"
    multi_subdir: str = "multi"


class UiCfg(BaseModel):
    """Visualization settings. Maps to ``cfg.ui``.

    Most of these are MATLAB-figure quirks; the web frontend will read
    only ``spec_ncols`` and ``vad_hist_sec``. ``waveform_span`` is left
    optional and gets filled with ``frame_size`` by ``Config``'s
    validator (mirrors MATLAB's ``cfg.ui.waveform_span = cfg.frame_size``).
    """

    model_config = _MODEL_CFG

    spec_ncols: int = Field(90, ge=1)
    vad_hist_sec: float = Field(8.0, gt=0)
    waveform_span: int | None = Field(default=None, ge=1)
    # Kept for parity with MATLAB even though the web app doesn't use it.
    fig_position: tuple[int, int, int, int] = (50, 40, 1560, 900)


# --------------------------------------------------------------------------- #
# Top-level config
# --------------------------------------------------------------------------- #
class Config(BaseModel):
    """Top-level Q-WiSE runtime configuration.

    Field names, defaults and units match
    :file:`acoustic_simulator/config/default.m`. Where the MATLAB
    project nested values under structs (``cfg.env.*`` etc.) the Python
    port uses dedicated Pydantic submodels so live edits stay typed.
    """

    model_config = _MODEL_CFG

    # ---------------- Audio / framing -----------------------------------
    fs: int = Field(16000, ge=1, description="Sample rate, Hz.")
    frame_size: int = Field(1024, ge=1, description="Block size, samples.")
    loop_sec: float = Field(120.0, gt=0, description="Pre-loaded noise loop length, seconds.")
    c: float = Field(343.0, gt=0, description="Speed of sound, m/s.")

    # ---------------- Microphone array ----------------------------------
    n_mics: int = Field(
        3,
        ge=1,
        description="Virtual mic array size.",
        json_schema_extra=_ui(
            "Microphones", "slider", "Mic count",
            min=1, max=8, step=1,
        ),
    )
    mic_spacing: float = Field(
        0.10,
        gt=0,
        description="Adjacent-mic spacing, metres.",
        json_schema_extra=_ui(
            "Microphones", "slider", "Mic spacing (m)",
            min=0.02, max=0.50, step=0.01,
        ),
    )
    mic_geometry: Literal["linear", "circular"] = Field(
        "linear",
        description="Array geometry; array is centred on the drone.",
        json_schema_extra=_ui(
            "Microphones", "select", "Geometry",
            options=["linear", "circular"],
        ),
    )

    # ---------------- Scene geometry ------------------------------------
    human_height: float = Field(
        1.70,
        gt=0,
        description="Speaker total height, metres.",
        json_schema_extra=_ui(
            "Scene", "slider", "Human height (m)",
            min=1.00, max=2.20, step=0.01,
        ),
    )
    slant_dist: float = Field(
        2.50,
        gt=0,
        description="Slant distance from the mouth to the drone, metres.",
        json_schema_extra=_ui(
            "Scene", "slider", "Slant distance (m)",
            min=0.50, max=10.00, step=0.05,
        ),
    )
    elev_deg: float = Field(
        30.0,
        ge=-90.0,
        le=90.0,
        description="Elevation of the drone above the mouth, degrees.",
        json_schema_extra=_ui(
            "Scene", "slider", "Drone elevation (°)",
            min=-30, max=90, step=1,
        ),
    )

    # ---------------- Drone source (orientation) ------------------------
    drone: DroneCfg = Field(default_factory=DroneCfg)

    # ---------------- Environmental noise source ------------------------
    env: EnvCfg = Field(default_factory=EnvCfg)

    # ---------------- Drone-rotor + acoustic constants ------------------
    drone_rpm: float = Field(
        8000.0,
        gt=0,
        description="Drone rotor RPM (drives the blade-pass frequency).",
        json_schema_extra=_ui(
            "Drone source", "slider", "Drone RPM",
            min=2000, max=15000, step=100,
        ),
    )
    drone_blades: int = Field(
        3,
        ge=1,
        description="Number of rotor blades.",
        json_schema_extra=_ui(
            "Drone source", "slider", "Drone blades",
            min=2, max=6, step=1,
        ),
    )
    ground_R: float = Field(
        0.90,
        ge=0.0,
        le=1.0,
        description="Ground reflection coefficient (asphalt, viz only).",
        json_schema_extra=_ui(
            "Acoustics", "slider", "Asphalt reflection R",
            min=0.0, max=1.0, step=0.01,
        ),
    )
    alpha_air_dB: float = Field(
        0.004,
        ge=0.0,
        description="Air absorption, dB per metre per kHz (viz only).",
        json_schema_extra=_ui(
            "Acoustics", "slider", "Air absorption (dB/m/kHz)",
            min=0.0, max=0.05, step=0.001,
        ),
    )
    distance_ref: float = Field(
        1.0,
        gt=0,
        description=(
            "1/r-law reference distance. Gains are clamped so d<d_ref → "
            "unity (no boost)."
        ),
        json_schema_extra=_ui(
            "Acoustics", "slider", "1/r reference distance (m)",
            min=0.1, max=5.0, step=0.05,
        ),
    )

    # ---------------- Noise-source WAV loops ----------------------------
    # The Python project ships the same loops as the MATLAB project under
    # ``wavs/``; tests + AudioIO resolve paths relative to the repo root.
    drone_wav_path: str = "wavs/drone_fan.wav"
    env_wav_path: str = "wavs/env_ambient.wav"

    # ---------------- Initial gains (pre-mixer) -------------------------
    speech_gain_init: float = Field(
        1.00,
        ge=0.0,
        description="Live-mic speech level into the mixer.",
        json_schema_extra=_ui(
            "Gains", "slider", "Speech gain",
            min=0.0, max=2.0, step=0.01,
        ),
    )
    drone_gain_init: float = Field(
        0.03,
        ge=0.0,
        description="Drone-fan loop level into the mixer.",
        json_schema_extra=_ui(
            "Gains", "slider", "Drone gain",
            min=0.0, max=1.0, step=0.01,
        ),
    )
    env_gain_init: float = Field(
        0.01,
        ge=0.0,
        description="Env-ambient loop level into the mixer.",
        json_schema_extra=_ui(
            "Gains", "slider", "Env gain",
            min=0.0, max=1.0, step=0.01,
        ),
    )

    # ---------------- Pipeline submodels --------------------------------
    mixer: MixerCfg = Field(default_factory=MixerCfg)
    vad: VadCfg = Field(default_factory=VadCfg)
    mwf: MwfCfg = Field(default_factory=MwfCfg)
    record: RecordCfg = Field(default_factory=RecordCfg)
    ui: UiCfg = Field(default_factory=UiCfg)

    # ---------------- Derived / computed values -------------------------
    @computed_field  # type: ignore[prop-decorator]
    @property
    def mouth_height(self) -> float:
        """Mouth height above ground — port of ``cfg.mouth_height``.

        Always ``0.88 × human_height``. Recomputed implicitly whenever
        ``human_height`` changes, so the UI slider drives it for free.
        """
        return 0.88 * self.human_height

    @model_validator(mode="after")
    def _default_waveform_span(self) -> Config:
        """Mirror ``cfg.ui.waveform_span = cfg.frame_size`` when unset."""
        if self.ui.waveform_span is None:
            # Bypass ``validate_assignment`` for this single derivation —
            # the value is itself an in-range int by construction.
            object.__setattr__(self.ui, "waveform_span", self.frame_size)
        return self


# --------------------------------------------------------------------------- #
# Public factories
# --------------------------------------------------------------------------- #
def default() -> Config:
    """Return a fully-populated :class:`Config` mirroring ``default.m``."""
    return Config()


def ui_sidebar_schema(cfg: Config | None = None) -> list[dict]:
    """Flatten the config into the entries the frontend renders.

    Each returned dict has:
        * ``key``     — dotted path (e.g. ``"env.distance_from_mouth"``)
        * ``value``   — current value from ``cfg``
        * ``section`` — accordion group label
        * ``widget``  — render hint (``slider`` / ``select`` / ...)
        * ``label``   — human-readable label
        * (widget-specific) ``min`` / ``max`` / ``step`` / ``options``

    The list is grouped by the canonical section order so the accordion
    is deterministic.
    """
    cfg = cfg or default()
    out: list[dict] = []

    def walk(prefix: str, model: BaseModel) -> None:
        for fname, field in model.__class__.model_fields.items():
            full = f"{prefix}.{fname}" if prefix else fname
            value = getattr(model, fname)
            if isinstance(value, BaseModel):
                walk(full, value)
                continue
            extra = field.json_schema_extra or {}
            ui_meta = extra.get("ui") if isinstance(extra, dict) else None
            if not ui_meta:
                continue
            out.append({"key": full, "value": value, **ui_meta})

    walk("", cfg)

    canonical = [
        "Microphones",
        "Scene",
        "Drone source",
        "Environment",
        "Acoustics",
        "Gains",
        "MWF",
    ]
    rank = {name: i for i, name in enumerate(canonical)}
    out.sort(key=lambda e: (rank.get(e["section"], 999),))
    return out


__all__ = [
    "Config",
    "DroneCfg",
    "EnvCfg",
    "MixerCfg",
    "MwfCfg",
    "RecordCfg",
    "UiCfg",
    "VadCfg",
    "default",
    "ui_sidebar_schema",
]
