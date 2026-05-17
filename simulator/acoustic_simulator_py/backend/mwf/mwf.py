"""Top-level Multi-channel Wiener Filter — port of ``mwf/mwf.m``.

The class exposes two complementary entry points:

* :meth:`Mwf.process` — full batch pipeline (align VAD → speech/noise
  masks → covariance estimation → beamformer weights → optional Wiener
  post-filter → iSTFT). Mirrors the reference Q-WiSE Python pipeline.
* :meth:`Mwf.step`    — streaming wrapper used by the FastAPI WebSocket
  loop. Maintains per-bin EMA estimates of ``Phi_ss`` and ``Phi_nn``
  and recomputes the beamformer per hop. Pass-through when
  ``cfg.mwf.passthrough`` is true.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .align_vad import align_vad
from .apply_beamformer import apply_beamformer
from .build_speech_mask import build_speech_mask
from .estimate_covariance import enforce_psd, estimate_covariance
from .stft import _periodic_hann, istft, stft
from .weights import (
    compute_gev_weights,
    compute_mvdr_weights,
    compute_mwf_weights,
)
from .wiener_postfilter import wiener_postfilter

if TYPE_CHECKING:
    from ..config import Config

log = logging.getLogger(__name__)

_VALID_METHODS = frozenset({"gev", "mwf", "mvdr"})


class BadMethodError(ValueError):
    """Raised when ``cfg.mwf.method`` is not in ``{gev, mwf, mvdr}``.

    Mirrors the MATLAB ``mwf:bad_method`` error ID.
    """


class Mwf:
    """Q-WiSE Multi-channel Wiener Filter (batch + streaming).

    Construction mirrors the MATLAB constructor: cache every ``cfg.mwf.*``
    field once, build the streaming Hann window, allocate the per-bin
    covariance EMAs and the overlap-add buffers, and validate ``method``.
    """

    cfg: "Config"

    def __init__(self, cfg: "Config") -> None:
        self.cfg = cfg
        m = cfg.mwf

        self.method = m.method.lower()
        self._validate_method(self.method)
        self.ref_mic = int(m.ref_mic)
        self.mu = float(m.mu)
        self.eps_reg = float(m.eps_reg)
        self.diag_load_ratio = float(m.diag_load_ratio)
        self.postfilter = bool(m.postfilter)
        self.gain_floor = float(m.gain_floor)
        self.noise_floor_alpha = float(m.noise_floor_alpha)
        self.pf_smooth_kernel = int(m.pf_smooth_kernel)
        self.mask_threshold = float(m.mask_threshold)
        self.mask_context = int(m.mask_context)
        self.n_fft = int(m.n_fft)
        self.hop = int(m.hop)
        self.Lwin = int(m.stft_win)
        self.Hstep = int(m.stft_hop)
        self.nbin = self.Lwin // 2 + 1
        self.n_mics = int(cfg.n_mics)
        self.alpha_nn = float(m.alpha_nn)
        self.alpha_ss = float(m.alpha_ss)
        self.passthrough = bool(m.passthrough)

        # Streaming state.
        self.win = np.sqrt(_periodic_hann(self.Lwin))
        self.Rnn = self._make_initial_covariance()
        self.Rss = self._make_initial_covariance()
        self._in_buf = np.zeros((self.Lwin, self.n_mics), dtype=np.float64)
        self._ola_buf = np.zeros(self.Lwin, dtype=np.float64)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Zero the analysis ring + OLA tail and re-seed the covariances."""
        self._in_buf.fill(0.0)
        self._ola_buf.fill(0.0)
        self.Rnn = self._make_initial_covariance()
        self.Rss = self._make_initial_covariance()

    # ---- Batch -------------------------------------------------------- #
    def process(
        self,
        vad_audio: ArrayLike,
        mic_signals: ArrayLike | Iterable[ArrayLike],
    ) -> NDArray[np.float64]:
        """Run the full batch MWF pipeline on a complete recording.

        Parameters
        ----------
        vad_audio
            VAD-extracted speech audio, ``[L_vad]``.
        mic_signals
            Either a ``[L_mic, n_mics]`` matrix or a list/tuple of
            ``n_mics`` 1-D arrays. If channel lengths differ the matrix
            is truncated to the shortest.
        """
        mic_mat = self._pack_mic_matrix(mic_signals)
        n_samples, n_ch = mic_mat.shape
        if n_ch < 2:
            warnings.warn(
                "MWF expects at least 2 channels for spatial filtering.",
                RuntimeWarning,
                stacklevel=2,
            )

        # ---- Align VAD-extracted speech to the reference mic ----------
        ref = mic_mat[:, self.ref_mic - 1]
        _, aligned_vad = align_vad(vad_audio, ref)

        # ---- Multi-channel STFT ---------------------------------------
        stft_list = [stft(mic_mat[:, m], self.n_fft, self.hop) for m in range(n_ch)]
        n_freq, n_frames = stft_list[0].shape
        X_multi = np.stack(stft_list, axis=0)   # [n_ch, n_freq, n_frames]

        # ---- Build speech / noise frame masks -------------------------
        speech_mask = build_speech_mask(
            aligned_vad, n_frames, self.n_fft, self.hop,
            self.mask_threshold, self.mask_context,
        )
        noise_mask = ~speech_mask

        # Safety nets — same fall-throughs as the MATLAB reference.
        if speech_mask.sum() < 5:
            speech_mask = build_speech_mask(
                aligned_vad, n_frames, self.n_fft, self.hop,
                0.003, self.mask_context,
            )
            noise_mask = ~speech_mask
        if noise_mask.sum() < 10:
            n15 = max(10, n_frames // 7)
            noise_mask = np.zeros(n_frames, dtype=bool)
            noise_mask[: min(n15, n_frames)] = True
            tail_start = max(0, n_frames - n15)
            noise_mask[tail_start:] = True
            speech_mask = ~noise_mask

        # ---- Covariances ----------------------------------------------
        Phi_nn = estimate_covariance(X_multi, noise_mask, self.eps_reg)
        Phi_yy = estimate_covariance(X_multi, speech_mask, self.eps_reg)
        Phi_ss = Phi_yy - Phi_nn
        Phi_ss = enforce_psd(Phi_ss, n_ch, self.eps_reg)

        # ---- Beamformer weights + apply --------------------------------
        W = self._compute_weights(Phi_ss, Phi_nn)
        Y = apply_beamformer(W, X_multi)

        if self.postfilter:
            Y = wiener_postfilter(
                Y, speech_mask,
                self.noise_floor_alpha, self.gain_floor,
                self.eps_reg, self.pf_smooth_kernel,
            )

        # ---- iSTFT + normalise ----------------------------------------
        y = istft(Y, self.n_fft, self.hop)
        y = np.real(y)
        if y.size > n_samples:
            y = y[:n_samples]
        peak = float(np.max(np.abs(y))) if y.size else 0.0
        if peak > 0:
            y = y / peak * 0.9
        return y

    # ---- Streaming ---------------------------------------------------- #
    def step(
        self, x: ArrayLike, is_speech: bool
    ) -> NDArray[np.float64]:
        """Process one ``[N, n_mics]`` mic block, return an ``[N]`` mono out."""
        xa = np.asarray(x, dtype=np.float64)
        if xa.ndim == 1:
            xa = xa[:, None]
        N = xa.shape[0]

        if self.passthrough:
            return xa[:, self.ref_mic - 1].copy()

        y = np.zeros(N, dtype=np.float64)
        L = self.Lwin
        H = self.Hstep
        cursor = 0
        while cursor + H <= N:
            seg = xa[cursor : cursor + H, :]
            # Slide the analysis ring left, append the new hop.
            self._in_buf = np.concatenate(
                [self._in_buf[H:, :], seg], axis=0
            )

            if np.any(self._in_buf):
                buf_win = self._in_buf * self.win[:, None]
                Xf_full = np.fft.fft(buf_win, n=L, axis=0)
                Xf = Xf_full[: self.nbin, :]              # [nbin, n_mics]

                self._update_covariances(Xf, bool(is_speech))
                W = self._compute_weights(self.Rss, self.Rnn)
                # Y_f = sum_m conj(W[k, m]) * Xf[k, m]   -> [nbin]
                Yf = np.sum(np.conj(W) * Xf, axis=1)

                # Mirror to the full conjugate-symmetric spectrum and ifft.
                Yfull = np.concatenate([Yf, np.conj(Yf[-2:0:-1])])
                yblk = np.real(np.fft.ifft(Yfull, n=L)) * self.win

                self._ola_buf = self._ola_buf + yblk
                out_h = self._ola_buf[:H].copy()
                self._ola_buf = np.concatenate(
                    [self._ola_buf[H:], np.zeros(H, dtype=np.float64)]
                )
            else:
                out_h = np.zeros(H, dtype=np.float64)

            y[cursor : cursor + H] = out_h
            cursor += H

        # Tail bytes that don't make a full hop pass through the ref mic.
        if cursor < N:
            y[cursor:] = xa[cursor:, self.ref_mic - 1]
        return y

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _make_initial_covariance(self) -> NDArray[np.complex128]:
        I0 = self.eps_reg * np.eye(self.n_mics, dtype=np.complex128)
        return np.broadcast_to(
            I0[None, :, :], (self.nbin, self.n_mics, self.n_mics)
        ).copy()

    def _update_covariances(
        self, Xf: NDArray[np.complex128], is_speech: bool
    ) -> None:
        """EMA-update either Rss (speech frames) or Rnn (noise frames).

        Vectorised across freq bins: ``R[k] = outer(Xf[k], conj(Xf[k]))``.
        """
        a = self.alpha_ss if is_speech else self.alpha_nn
        R = Xf[:, :, None] * np.conj(Xf[:, None, :])    # [nbin, n_mics, n_mics]
        if is_speech:
            self.Rss = a * self.Rss + (1.0 - a) * R
        else:
            self.Rnn = a * self.Rnn + (1.0 - a) * R

    def _compute_weights(
        self,
        Phi_ss: NDArray[np.complex128],
        Phi_nn: NDArray[np.complex128],
    ) -> NDArray[np.complex128]:
        if self.method == "mvdr":
            return compute_mvdr_weights(
                Phi_ss, Phi_nn, self.ref_mic, self.eps_reg, self.diag_load_ratio
            )
        if self.method == "gev":
            return compute_gev_weights(
                Phi_ss, Phi_nn, self.ref_mic, self.eps_reg, self.diag_load_ratio
            )
        return compute_mwf_weights(
            Phi_ss, Phi_nn, self.ref_mic, self.mu, self.eps_reg, self.diag_load_ratio
        )

    @staticmethod
    def _validate_method(method: str) -> None:
        if method not in _VALID_METHODS:
            raise BadMethodError(
                f"cfg.mwf.method must be one of: gev | mwf | mvdr (got {method!r})."
            )

    @staticmethod
    def _pack_mic_matrix(
        mic_signals: ArrayLike | Iterable[ArrayLike],
    ) -> NDArray[np.float64]:
        """Coerce input to ``[n_samples, n_mics]`` ``float64`` matrix."""
        if isinstance(mic_signals, np.ndarray):
            m = mic_signals
            if m.ndim != 2:
                raise ValueError(
                    "mic_signals must be 2-D or an iterable of 1-D arrays."
                )
            if m.shape[0] < m.shape[1]:
                m = m.T  # treat [n_ch, L] orientation defensively
            return np.asarray(m, dtype=np.float64)
        # Iterable of 1-D arrays (list / tuple of per-mic captures).
        if not isinstance(mic_signals, (list, tuple)):
            mic_signals = list(mic_signals)
        arrs = [np.asarray(a, dtype=np.float64).reshape(-1) for a in mic_signals]
        if not arrs:
            raise ValueError("mic_signals is empty.")
        L = min(a.size for a in arrs)
        return np.column_stack([a[:L] for a in arrs])


__all__ = ["Mwf", "BadMethodError"]
