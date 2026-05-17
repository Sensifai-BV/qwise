"""Physical multi-source → N-microphone array simulator.

Port of ``core/SourceMixer.m``.

Every microphone receives every source (speech, drone, environment).
For each source ``s`` and mic ``m``::

    mic_m[n] = sum over s in {speech, drone, env}
                  gain_s(m) * src_s[n - frac_delay_s(m)]

Fractional sample delays are applied by linear interpolation across a
per-source history ring buffer so blocks stitch together without
clicking.

This wiring matches what the rest of the Q-WiSE pipeline expects:

* the VAD consumes a single mono signal (the composite — typically the
  reference mic), and
* the MWF consumes the full N-channel block returned by :meth:`mix`.

The ``composite`` reduction is controlled by ``cfg.mixer.composite``
(``'mic1' | 'sum' | 'mean'``).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..config import Config
from .geometry import Geometry

FloatArray = NDArray[np.float64]


class NMicChangeError(ValueError):
    """:meth:`SourceMixer.update_geometry` got a geo with a different mic count.

    Mirrors the MATLAB error ID ``SourceMixer:update_geometry:NMicChange``.
    Changing the mic count at runtime would invalidate downstream buffer
    shapes (history rings, MWF covariance buffers, recording sessions),
    so we raise rather than silently re-allocate.
    """


class SourceMixer:
    """Streaming N-mic mixer with per-source TDOA + 1/r gain."""

    # Public — kept on ``self`` for parity with the MATLAB instance.
    cfg: Config
    geo: Geometry
    n_mics: int
    composite_kind: str
    mode: str = "physical"     # retained for UI / status display
    hist_len: int

    def __init__(self, cfg: Config, geo: Geometry) -> None:
        self.cfg = cfg
        self.geo = geo
        self.n_mics = int(cfg.n_mics)

        # ``cfg.mixer.composite`` is guaranteed to exist (Pydantic
        # default), but defensively coerce to lower-case once so the
        # hot loop in :meth:`composite` doesn't have to.
        self.composite_kind = (cfg.mixer.composite or "mic1").lower()
        self._ref_mic = self._resolve_ref_mic()

        # Cache per-mic delays and gains as 1-D ``float64`` arrays so the
        # block-loop avoids re-validating Pydantic models per call.
        self._frac_delays_speech = self._fetch_frac_delays(geo, "speech")
        self._frac_delays_drone = self._fetch_frac_delays(geo, "drone")
        self._frac_delays_env = self._fetch_frac_delays(geo, "env")
        self._gains_speech = np.asarray(geo.gains_speech, dtype=np.float64).reshape(-1)
        self._gains_drone = np.asarray(geo.gains_drone, dtype=np.float64).reshape(-1)
        self._gains_env = np.asarray(geo.gains_env, dtype=np.float64).reshape(-1)

        # Size the history ring so the linear-interp right tap is always
        # in range. ``ceil(tau_max) + 1`` matches the MATLAB constructor.
        tau_max = float(max(
            self._frac_delays_speech.max(),
            self._frac_delays_drone.max(),
            self._frac_delays_env.max(),
            0.0,
        ))
        self.hist_len = int(np.ceil(tau_max)) + 1

        self._hist_speech = np.zeros(self.hist_len, dtype=np.float64)
        self._hist_drone = np.zeros(self.hist_len, dtype=np.float64)
        self._hist_env = np.zeros(self.hist_len, dtype=np.float64)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def mix(
        self, speech: ArrayLike, drone: ArrayLike, env: ArrayLike
    ) -> FloatArray:
        """Run one block through the physical multi-source mixer.

        Inputs are coerced to 1-D ``float64`` arrays and zero-padded to
        the longest of the three. Output is ``[N, n_mics]`` where every
        mic carries speech + drone + env with its own TDOA + 1/r gain.
        """
        s, d, e, n = self._coerce_sources(speech, drone, env)

        # Build [history; current] tapes once per source. The current
        # block sits at indices ``[hist_len, hist_len + n)`` (0-based);
        # query indices ``q`` below are computed accordingly.
        sp_tape = np.concatenate([self._hist_speech, s])
        dr_tape = np.concatenate([self._hist_drone, d])
        en_tape = np.concatenate([self._hist_env, e])

        out = np.zeros((n, self.n_mics), dtype=np.float64)
        n_idx = np.arange(n, dtype=np.float64)
        base = float(self.hist_len)
        for m in range(self.n_mics):
            qs = base + n_idx - self._frac_delays_speech[m]
            qd = base + n_idx - self._frac_delays_drone[m]
            qe = base + n_idx - self._frac_delays_env[m]

            ts = self._frac_tap(sp_tape, qs)
            td = self._frac_tap(dr_tape, qd)
            te = self._frac_tap(en_tape, qe)

            out[:, m] = (
                self._gains_speech[m] * ts
                + self._gains_drone[m] * td
                + self._gains_env[m] * te
            )

        # Advance the history rings.
        self._hist_speech = self._push(self._hist_speech, s)
        self._hist_drone = self._push(self._hist_drone, d)
        self._hist_env = self._push(self._hist_env, e)
        return out

    def composite(self, mic: FloatArray) -> FloatArray:
        """Reduce the N-channel mix to one mono signal for the VAD.

        * ``'mic1'`` (default) — returns the reference-mic channel.
        * ``'sum'``            — sums across mics.
        * ``'mean'``           — averages across mics.

        The empty-array branch mirrors the MATLAB safety net for callers
        that pass through a frame before the pipeline has produced any
        data yet.
        """
        if mic is None or mic.size == 0:
            return np.zeros(self.cfg.frame_size, dtype=np.float64)
        if self.composite_kind == "mean":
            return mic.mean(axis=1)
        if self.composite_kind == "sum":
            return mic.sum(axis=1)
        # 'mic1' or anything else falls through to the ref-mic channel.
        ref0 = min(self._ref_mic, mic.shape[1]) - 1   # 1-indexed → 0-indexed
        return mic[:, ref0].copy()

    def reset(self) -> None:
        """Zero every per-source history ring."""
        self._hist_speech.fill(0.0)
        self._hist_drone.fill(0.0)
        self._hist_env.fill(0.0)

    def ref_mic(self) -> int:
        """1-indexed reference-mic number (matches MATLAB convention)."""
        return self._ref_mic

    def update_geometry(self, new_geo: Geometry) -> None:
        """Swap in a new :class:`Geometry` at runtime.

        Re-caches per-source fractional delays and 1/r gains and resizes
        the per-source history rings, preserving the tail so the next
        ``mix()`` call does not click on the boundary.

        Used by the live UI scene controls (human height, slant
        distance, azimuths, ...). The mic count must NOT change at
        runtime because that would invalidate downstream buffer shapes;
        an attempt to do so raises :class:`NMicChangeError`.
        """
        new_n = int(np.asarray(new_geo.gains_speech).shape[0])
        if new_n != self.n_mics:
            raise NMicChangeError(
                f"Cannot change n_mics at runtime (have {self.n_mics}, got {new_n})."
            )

        self.geo = new_geo
        self._frac_delays_speech = self._fetch_frac_delays(new_geo, "speech")
        self._frac_delays_drone = self._fetch_frac_delays(new_geo, "drone")
        self._frac_delays_env = self._fetch_frac_delays(new_geo, "env")
        self._gains_speech = np.asarray(new_geo.gains_speech, dtype=np.float64).reshape(-1)
        self._gains_drone = np.asarray(new_geo.gains_drone, dtype=np.float64).reshape(-1)
        self._gains_env = np.asarray(new_geo.gains_env, dtype=np.float64).reshape(-1)

        tau_max = float(max(
            self._frac_delays_speech.max(),
            self._frac_delays_drone.max(),
            self._frac_delays_env.max(),
            0.0,
        ))
        new_hist_len = max(1, int(np.ceil(tau_max)) + 1)

        if new_hist_len > self.hist_len:
            pad = new_hist_len - self.hist_len
            zeros = np.zeros(pad, dtype=np.float64)
            self._hist_speech = np.concatenate([zeros, self._hist_speech])
            self._hist_drone = np.concatenate([zeros, self._hist_drone])
            self._hist_env = np.concatenate([zeros, self._hist_env])
        elif new_hist_len < self.hist_len:
            self._hist_speech = self._hist_speech[-new_hist_len:].copy()
            self._hist_drone = self._hist_drone[-new_hist_len:].copy()
            self._hist_env = self._hist_env[-new_hist_len:].copy()
        self.hist_len = new_hist_len

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _coerce_sources(
        speech: ArrayLike, drone: ArrayLike, env: ArrayLike
    ) -> tuple[FloatArray, FloatArray, FloatArray, int]:
        """Coerce inputs to ``float64`` 1-D arrays of the same length."""
        s = np.asarray(speech, dtype=np.float64).reshape(-1)
        d = np.asarray(drone, dtype=np.float64).reshape(-1)
        e = np.asarray(env, dtype=np.float64).reshape(-1)
        n = max(s.size, d.size, e.size)
        if s.size < n:
            s = np.concatenate([s, np.zeros(n - s.size, dtype=np.float64)])
        if d.size < n:
            d = np.concatenate([d, np.zeros(n - d.size, dtype=np.float64)])
        if e.size < n:
            e = np.concatenate([e, np.zeros(n - e.size, dtype=np.float64)])
        return s, d, e, n

    @staticmethod
    def _frac_tap(tape: FloatArray, q: FloatArray) -> FloatArray:
        """Linear-interpolated tap into a ``[history; current]`` tape.

        ``q`` is an array of 0-based fractional query indices. Anything
        that falls outside the tape reads as zero (silence), so a
        too-large delay at startup produces a clean fade-in instead of
        an out-of-bounds error. Note: this is the Python (0-indexed)
        equivalent of the MATLAB ``frac_tap_`` helper.
        """
        L = tape.shape[0]
        q0 = np.floor(q).astype(np.int64)
        qf = q - q0
        in_range = (q0 >= 0) & (q0 + 1 < L)
        idx_a = np.clip(q0, 0, L - 1)
        idx_b = np.clip(q0 + 1, 0, L - 1)
        interp = (1.0 - qf) * tape[idx_a] + qf * tape[idx_b]
        return np.where(in_range, interp, 0.0)

    def _push(self, h: FloatArray, src: FloatArray) -> FloatArray:
        """Slide the history ring left, appending ``src`` on the right."""
        n = src.shape[0]
        if n >= self.hist_len:
            return src[-self.hist_len:].copy()
        return np.concatenate([h[n:], src])

    @staticmethod
    def _fetch_frac_delays(geo: Geometry, which: str) -> FloatArray:
        """Read the per-source fractional delays from ``geo``.

        Falls back to the integer ``delays_*`` field for backward
        compatibility with older :class:`Geometry` value objects (and
        with any future MATLAB->Python round-trip).
        """
        name = f"frac_delays_{which}"
        if hasattr(geo, name) and getattr(geo, name) is not None:
            return np.asarray(getattr(geo, name), dtype=np.float64).reshape(-1)
        return np.asarray(getattr(geo, f"delays_{which}"), dtype=np.float64).reshape(-1)

    def _resolve_ref_mic(self) -> int:
        """Read ``cfg.mwf.ref_mic`` and clamp to ``[1, n_mics]``."""
        ref = int(self.cfg.mwf.ref_mic)
        return max(1, min(ref, self.n_mics))


__all__ = ["SourceMixer", "NMicChangeError"]
