"""Scene geometry — port of ``core/build_geometry.m``.

Returns a :class:`Geometry` value object (a frozen-ish dataclass holding
NumPy arrays) used by the SourceMixer to synthesise N physically-realistic
microphone signals (mixer_1.m-style):

* **Human speech** — point source at mouth height in front of the drone.
* **Drone body**   — point source at the array centre (rotor / fan noise).
* **Environment**  — independent point source at a configurable
  distance / azimuth / elevation from the speaker, so the mixer can
  give it its own TDOA + 1/r gain on every microphone.

For each source ``s`` and mic ``m`` the function computes:

    d_s(m)              = ||pos_mic[m] - pos_src||
    frac_delay_s(m)     = (d_s(m) - min(d_s)) / c * fs    # float samples
    delay_s(m)          = round(frac_delay_s(m))          # int samples
    gain_s(m)           = d_ref / max(d_s(m), d_ref)

Fractional delays are what :mod:`backend.core.source_mixer` uses
(fractional taps via linear interpolation). The integer ``delays_*``
arrays are kept for legacy utilities (print_geometry, expand_channels,
draw_scene).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..config import Config

# --------------------------------------------------------------------------- #
# Type aliases
# --------------------------------------------------------------------------- #
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


# --------------------------------------------------------------------------- #
# Geometry value object
# --------------------------------------------------------------------------- #
@dataclass
class Geometry:
    """Output of :func:`build_geometry`. Mirrors the MATLAB ``geo`` struct.

    All arrays are 1-D length-N (per-mic) unless noted; positions are
    Cartesian 3-vectors and ``pos_mics`` is ``(N, 3)``.

    Notes
    -----
    * ``delays_*`` arrays are *integer* sample counts (rounded from the
      fractional delays). Both representations are referenced so the
      closest mic has delay ``0`` for that source.
    * ``frac_delays_*`` are float-valued sample counts — the SourceMixer
      interpolates between them.
    * ``pos_env_noise`` and ``delays`` are MATLAB-era legacy aliases
      exposed as ``@property`` so callers that follow the old struct
      shape (print_geometry, expand_channels) keep working.
    """

    # --- positions ---
    pos_human: FloatArray              # (3,)
    pos_img_src: FloatArray            # (3,) — ground image, viz only
    pos_drone: FloatArray              # (3,)
    pos_mics: FloatArray               # (N, 3)
    pos_env: FloatArray                # (3,)

    # --- per-source per-mic distances ---
    dist_speech: FloatArray            # (N,)
    dist_img: FloatArray               # (N,) — viz only
    dist_drone: FloatArray             # (N,)
    dist_env: FloatArray               # (N,)

    # --- per-source TDOA (integer samples, min == 0) ---
    delays_speech: IntArray
    delays_drone: IntArray
    delays_env: IntArray

    # --- per-source TDOA (fractional samples, min == 0) ---
    frac_delays_speech: FloatArray
    frac_delays_drone: FloatArray
    frac_delays_env: FloatArray

    # --- per-source 1/r gain (clamped at ``distance_ref``) ---
    gains_speech: FloatArray
    gains_drone: FloatArray
    gains_env: FloatArray

    # --- derived scalars ---
    grazing_deg: FloatArray            # (N,) — ground-reflection grazing angle
    distance_ref: float
    drone_agl: float

    # ----------------------------------------------------------------- #
    # Legacy aliases (kept so print_geometry / expand_channels / draw_scene
    # ports keep working unchanged).
    # ----------------------------------------------------------------- #
    @property
    def delays(self) -> IntArray:
        """Speech-centric integer TDOA — alias of :attr:`delays_speech`."""
        return self.delays_speech

    @property
    def pos_env_noise(self) -> FloatArray:
        """MATLAB alias for :attr:`pos_env`."""
        return self.pos_env

    # ----------------------------------------------------------------- #
    # Convenience
    # ----------------------------------------------------------------- #
    @property
    def n_mics(self) -> int:
        return int(self.pos_mics.shape[0])

    def summary(self) -> str:
        """Human-readable one-liner per mic — port of ``print_geometry.m``."""
        lines = ["=== Geometry (outdoor / asphalt) ==="]
        lines.append(
            f"  Mouth     : [{self.pos_human[0]: .3f} "
            f"{self.pos_human[1]: .3f} {self.pos_human[2]: .3f}] m"
        )
        lines.append(
            f"  Image src : [{self.pos_img_src[0]: .3f} "
            f"{self.pos_img_src[1]: .3f} {self.pos_img_src[2]: .3f}] m"
        )
        lines.append(
            f"  Drone     : [{self.pos_drone[0]: .3f} "
            f"{self.pos_drone[1]: .3f} {self.pos_drone[2]: .3f}] m  "
            f"AGL={self.drone_agl:.2f} m"
        )
        for m in range(self.n_mics):
            p = self.pos_mics[m]
            lines.append(
                f"  Mic{m + 1} pos=[{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}]  "
                f"d={self.dist_speech[m]:.3f} m  "
                f"TDOA={int(self.delays_speech[m])} smp"
            )
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Geometry helpers (kept module-public; SourceMixer / tests reuse them)
# --------------------------------------------------------------------------- #
def spherical_offset(
    distance: float, azimuth_deg: float, elevation_deg: float
) -> FloatArray:
    """Cartesian offset for a spherical (range, az, el) target.

    Port of MATLAB ``spherical_offset_``::

        off = distance * [cos(el)*cos(az), cos(el)*sin(az), sin(el)]
    """
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    return distance * np.array(
        [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)],
        dtype=np.float64,
    )


def build_mic_array(
    center: FloatArray | tuple[float, float, float],
    n: int,
    spacing: float,
    geometry: str,
) -> FloatArray:
    """Build a linear or circular mic array centred on ``center``.

    Port of MATLAB ``build_mic_array_``.

    Parameters
    ----------
    center
        Cartesian centre of the array, broadcast onto every mic.
    n
        Number of microphones (``>= 1``).
    spacing
        Adjacent-mic spacing in metres. For ``'linear'`` this is the ULA
        spacing; for ``'circular'`` it is the chord between adjacent
        mics (radius is derived so the chord matches).
    geometry
        ``'linear'`` (ULA along x-axis) or ``'circular'`` (horizontal
        ring in the xy-plane).
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    c = np.asarray(center, dtype=np.float64).reshape(3)
    g = geometry.lower()
    if g == "linear":
        # ULA along the x-axis, centred on the drone.
        idx = np.arange(n, dtype=np.float64) - (n - 1) / 2.0
        offsets = np.column_stack(
            [idx * spacing, np.zeros(n), np.zeros(n)]
        )
    elif g == "circular":
        # Equal-chord circular array; chord ≈ spacing → radius = spacing / (2 sin(pi/n)).
        # ``max(n, 2)`` matches the MATLAB code's safety for n == 1.
        theta = np.arange(n, dtype=np.float64) * (2.0 * np.pi / max(n, 2))
        radius = spacing / (2.0 * np.sin(np.pi / max(n, 2)))
        offsets = np.column_stack(
            [radius * np.cos(theta), radius * np.sin(theta), np.zeros(n)]
        )
    else:
        raise ValueError(
            f"Unsupported mic geometry {geometry!r} (use 'linear' or 'circular')."
        )
    return c[None, :] + offsets


def _samp_delays(
    d: FloatArray, c: float, fs: int
) -> tuple[IntArray, FloatArray]:
    """Per-mic sample delay, referenced to the closest mic (min == 0).

    Port of MATLAB ``samp_delays_``. Returns ``(integer_samples, fractional_samples)``.
    """
    tau_abs_frac = (d / c) * fs
    tau_frac = tau_abs_frac - np.min(tau_abs_frac)
    tau_int = np.rint(tau_frac).astype(np.int64)
    return tau_int, tau_frac


# --------------------------------------------------------------------------- #
# Top-level
# --------------------------------------------------------------------------- #
def build_geometry(cfg: Config) -> Geometry:
    """Compute scene geometry, per-source TDOAs and 1/r gains from ``cfg``.

    Mirrors ``core/build_geometry.m``. The output is the single value
    object the SourceMixer (and downstream MWF / VAD wiring) consumes
    between blocks; live UI edits rebuild it.
    """
    c = float(cfg.c)
    mh = float(cfg.mouth_height)
    d_ref = float(cfg.distance_ref)
    fs = int(cfg.fs)
    n_mics = int(cfg.n_mics)

    # --- Source positions ---------------------------------------------------
    pos_human = np.array([0.0, 0.0, mh], dtype=np.float64)
    pos_img_src = np.array([0.0, 0.0, -mh], dtype=np.float64)  # ground image, viz only

    pos_drone = pos_human + spherical_offset(
        cfg.slant_dist, cfg.drone.azimuth_deg, cfg.elev_deg
    )
    pos_mics = build_mic_array(pos_drone, n_mics, cfg.mic_spacing, cfg.mic_geometry)
    pos_env = pos_human + spherical_offset(
        cfg.env.distance_from_mouth,
        cfg.env.azimuth_deg,
        cfg.env.elevation_deg,
    )

    # --- Per-source per-mic distances --------------------------------------
    d_speech = np.linalg.norm(pos_mics - pos_human[None, :], axis=1)
    d_img = np.linalg.norm(pos_mics - pos_img_src[None, :], axis=1)
    d_drone = np.linalg.norm(pos_mics - pos_drone[None, :], axis=1)
    d_env = np.linalg.norm(pos_mics - pos_env[None, :], axis=1)

    # --- Per-source TDOA (fractional + integer) ----------------------------
    delays_speech, frac_delays_speech = _samp_delays(d_speech, c, fs)
    delays_drone, frac_delays_drone = _samp_delays(d_drone, c, fs)
    delays_env, frac_delays_env = _samp_delays(d_env, c, fs)

    # --- Per-source 1/r gain (clamped at d_ref) ----------------------------
    gains_speech = d_ref / np.maximum(d_speech, d_ref)
    gains_drone = d_ref / np.maximum(d_drone, d_ref)
    gains_env = d_ref / np.maximum(d_env, d_ref)

    # --- Grazing angle for the ground reflection (viz only) ----------------
    # mh / d_img is always in [0, 1] for mics above the ground.
    grazing_deg = np.rad2deg(np.arcsin(np.clip(mh / d_img, 0.0, 1.0)))

    return Geometry(
        pos_human=pos_human,
        pos_img_src=pos_img_src,
        pos_drone=pos_drone,
        pos_mics=pos_mics,
        pos_env=pos_env,
        dist_speech=d_speech,
        dist_img=d_img,
        dist_drone=d_drone,
        dist_env=d_env,
        delays_speech=delays_speech,
        delays_drone=delays_drone,
        delays_env=delays_env,
        frac_delays_speech=frac_delays_speech,
        frac_delays_drone=frac_delays_drone,
        frac_delays_env=frac_delays_env,
        gains_speech=gains_speech,
        gains_drone=gains_drone,
        gains_env=gains_env,
        grazing_deg=grazing_deg,
        distance_ref=d_ref,
        drone_agl=float(pos_drone[2]),
    )


__all__ = [
    "Geometry",
    "build_geometry",
    "build_mic_array",
    "spherical_offset",
]
