"""Decision-directed single-channel Wiener post-filter.

Port of ``mwf/mwf_wiener_postfilter.m``. Seeds the noise PSD from
noise-flagged frames, EMA-updates it on every noise frame, builds an
a-priori SNR per (t, f), applies the Wiener gain ``SNR / (SNR + 1)``
clamped at ``gain_floor``, and finally smooths the gain across
frequency with a length-``smooth_kernel`` moving average.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def wiener_postfilter(
    Y: ArrayLike,
    speech_mask: ArrayLike,
    noise_floor_alpha: float = 0.98,
    gain_floor: float = 0.08,
    eps_reg: float = 1e-10,
    smooth_kernel: int = 3,
) -> NDArray[np.complex128]:
    """Return gain-adjusted complex STFT, same shape as ``Y``."""
    Ya = np.asarray(Y, dtype=np.complex128)
    if Ya.ndim != 2:
        raise ValueError("Y must be a 2-D STFT matrix [n_freq, n_frames]")
    n_freq, n_frames = Ya.shape
    mag = np.abs(Ya)
    phase = np.angle(Ya)

    mask = np.asarray(speech_mask, dtype=bool).reshape(-1)
    if mask.size > n_frames:
        mask = mask[:n_frames]
    elif mask.size < n_frames:
        mask = np.concatenate([mask, np.zeros(n_frames - mask.size, dtype=bool)])

    noise_idx = np.flatnonzero(~mask)
    if noise_idx.size > 0:
        noise_psd = (mag[:, noise_idx] ** 2).mean(axis=1)
    else:
        noise_psd = 0.1 * (mag ** 2).mean(axis=1)
    noise_psd = np.maximum(noise_psd, eps_reg)
    noise_est = noise_psd.copy()

    G = np.ones((n_freq, n_frames), dtype=np.float64)
    for t in range(n_frames):
        P = mag[:, t] ** 2
        if not mask[t]:
            noise_est = (
                float(noise_floor_alpha) * noise_est
                + (1.0 - float(noise_floor_alpha)) * P
            )
        snr = np.maximum(P - noise_est, 0.0) / (noise_est + float(eps_reg))
        gain = snr / (snr + 1.0)
        G[:, t] = np.maximum(gain, float(gain_floor))

    if int(smooth_kernel) >= 3:
        kern = np.ones(int(smooth_kernel)) / float(smooth_kernel)
        for t in range(n_frames):
            G[:, t] = np.convolve(G[:, t], kern, mode="same")

    return mag * G * np.exp(1j * phase)


__all__ = ["wiener_postfilter"]
