"""Audio I/O — port of ``core/AudioIO.m`` adapted for the web demo.

The MATLAB AudioIO does three things: (1) read the host microphone, (2)
play back to the speakers, (3) handle the looped noise sources +
session recording. In the FastAPI / WebSocket port the *browser* owns
the mic and the speakers — only (3) survives. We keep:

* Looped drone / env / optional speech-WAV sources, exactly the buffer
  feed the SourceMixer expects.
* The :meth:`AudioIO.rec_start_session` / :meth:`rec_session_write` /
  :meth:`rec_stop_session` triple used by the SimulatorUI's playback
  panel — one folder per session, one WAV per track (mic in N files,
  vad + mwf in one file each).

Hardware-only members (``reader``, ``writer``, ``n_hw_ch``, ``play``,
``release``) are deliberately gone; the MATLAB ``mono`` / ``multi``
``rec_start`` modes also disappear because the live GUI only ever uses
``session`` mode now.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import soundfile as sf
from numpy.typing import ArrayLike, NDArray

from .loops import REPO_ROOT, load_wav_loop, loop_chunk

if TYPE_CHECKING:
    from ..config import Config

log = logging.getLogger(__name__)

# Default folder for runtime data (recordings + uploads). HF Docker
# Spaces set ``QWISE_DATA_DIR=/data`` in the Dockerfile; locally we
# fall back to the project root so the developer can find sessions
# without spelunking under /var.
DEFAULT_DATA_DIR = Path(os.getenv("QWISE_DATA_DIR", str(REPO_ROOT)))


class AudioIO:
    """Looped-source + session-recording sink. Browser handles HW I/O."""

    cfg: "Config"
    drone_wav: NDArray[np.float64]
    env_wav: NDArray[np.float64]
    speech_wav: NDArray[np.float64] | None
    speech_wav_path: str

    def __init__(self, cfg: "Config", data_dir: Path | str | None = None) -> None:
        self.cfg = cfg
        self.data_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR

        # --- Looped sources --------------------------------------------
        self.drone_wav = load_wav_loop(cfg.drone_wav_path, cfg.fs, cfg.loop_sec)
        self.env_wav = load_wav_loop(cfg.env_wav_path, cfg.fs, cfg.loop_sec)
        self.speech_wav = None
        self.speech_wav_path = ""

        # 0-based pointers (Python convention; MATLAB sibling is 1-based).
        self._drone_ptr: int = 0
        self._env_ptr: int = 0
        self._speech_ptr: int = 0

        # --- Recording session state -----------------------------------
        self._sess_active: bool = False
        self._sess_dir: str = ""
        # name → dict(buf=ndarray, len=int, cap=int, n_ch=int)
        self._sess_tracks: dict[str, dict] = {}

        log.info(
            "[Q-WiSE] AudioIO ready (data_dir=%s, n_mics=%d, fs=%d, loop_sec=%.1f)",
            self.data_dir,
            cfg.n_mics,
            cfg.fs,
            cfg.loop_sec,
        )

    # ================================================================== #
    # Looped sources
    # ================================================================== #
    def next_drone_chunk(self, n: int) -> NDArray[np.float64]:
        """Next ``n`` samples of the drone-fan loop."""
        c = loop_chunk(self.drone_wav, self._drone_ptr, n)
        self._drone_ptr = (self._drone_ptr + int(n)) % self.drone_wav.shape[0]
        return c

    def next_env_chunk(self, n: int) -> NDArray[np.float64]:
        """Next ``n`` samples of the env-ambient loop."""
        c = loop_chunk(self.env_wav, self._env_ptr, n)
        self._env_ptr = (self._env_ptr + int(n)) % self.env_wav.shape[0]
        return c

    # ---- Clean-speech WAV source ------------------------------------- #
    def load_speech_wav(self, path: str | Path) -> None:
        """Load a clean-speech WAV (any sr / channel count), peak-normalise
        to ~0.9, cache as a looped buffer. Mirrors ``load_speech_wav`` in
        the MATLAB AudioIO."""
        p = Path(path)
        data, fs_orig = sf.read(str(p), dtype="float64")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if fs_orig != self.cfg.fs:
            from scipy import signal as sig
            data = sig.resample(
                data, int(round(len(data) * self.cfg.fs / fs_orig))
            )
        peak = float(np.max(np.abs(data)))
        if peak > 0:
            data = data / peak * 0.9
        self.speech_wav = data.astype(np.float64)
        self.speech_wav_path = str(p)
        self._speech_ptr = 0
        log.info(
            "[Q-WiSE] Loaded clean-speech WAV: %s (%.2f s, %d Hz)",
            p, data.size / self.cfg.fs, self.cfg.fs,
        )

    def clear_speech_wav(self) -> None:
        """Drop the cached speech loop (revert to live mic input)."""
        self.speech_wav = None
        self.speech_wav_path = ""
        self._speech_ptr = 0

    def has_speech_wav(self) -> bool:
        return self.speech_wav is not None and self.speech_wav.size > 0

    def next_speech_chunk(self, n: int) -> NDArray[np.float64]:
        """Next ``n`` samples of the clean-speech loop, or silence if no WAV."""
        if self.speech_wav is None or self.speech_wav.size == 0:
            return np.zeros(int(n), dtype=np.float64)
        c = loop_chunk(self.speech_wav, self._speech_ptr, n)
        self._speech_ptr = (self._speech_ptr + int(n)) % self.speech_wav.shape[0]
        return c

    # ================================================================== #
    # Session recording — single mode, multiple named tracks
    # ================================================================== #
    def rec_start_session(self, path: str | Path | None = None) -> str:
        """Open a recording session folder.

        Returns the absolute folder path. Subsequent ``rec_session_write``
        calls add named tracks (in memory); ``rec_stop_session`` flushes
        every track to ``<session>/<name>.wav`` (single-channel) or
        ``<session>/<name>01.wav … <name>NN.wav`` (multi-channel).
        """
        if self._sess_active:
            log.warning(
                "[Q-WiSE] rec_start_session called while a session is active."
            )
            return self._sess_dir

        if path is None:
            path = self._default_session_path()
        target = Path(path).resolve()
        target.mkdir(parents=True, exist_ok=True)

        self._sess_dir = str(target)
        self._sess_tracks = {}
        self._sess_active = True
        log.info("[Q-WiSE] Recording session started -> %s", self._sess_dir)
        return self._sess_dir

    def rec_session_write(self, name: str, x: ArrayLike) -> None:
        """Append samples to a named track inside the active session.

        ``x`` is either a 1-D array (mono track) or a 2-D array
        ``[samples, channels]`` (multi-channel track). The track is
        lazily allocated on the first write; subsequent writes must
        match the original channel count.
        """
        if not self._sess_active:
            return
        if not name.isidentifier():
            log.warning(
                "[Q-WiSE] track name %r is not a valid identifier — skipping.",
                name,
            )
            return

        arr = np.asarray(x, dtype=np.float64)
        if arr.size == 0:
            return
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        elif arr.ndim != 2:
            log.warning("[Q-WiSE] track %r: expected 1-D or 2-D, got %dD.",
                        name, arr.ndim)
            return

        # Clip to [-1, 1] like the MATLAB sibling.
        np.clip(arr, -1.0, 1.0, out=arr)
        nx, n_ch = arr.shape

        track = self._sess_tracks.get(name)
        if track is None:
            cap0 = max(30 * self.cfg.fs, nx)
            track = {
                "buf": np.zeros((cap0, n_ch), dtype=np.float64),
                "len": 0,
                "cap": cap0,
                "n_ch": n_ch,
            }
            self._sess_tracks[name] = track
        elif track["n_ch"] != n_ch:
            log.warning(
                "[Q-WiSE] track %r: %d channels, expected %d.",
                name, n_ch, track["n_ch"],
            )
            return

        need = track["len"] + nx
        if need > track["cap"]:
            new_cap = max(need, 2 * track["cap"])
            tmp = np.zeros((new_cap, track["n_ch"]), dtype=np.float64)
            tmp[: track["len"], :] = track["buf"][: track["len"], :]
            track["buf"] = tmp
            track["cap"] = new_cap
        track["buf"][track["len"] : track["len"] + nx, :] = arr
        track["len"] = need

    def rec_stop_session(self) -> str:
        """Flush every track to disk and close the session.

        Returns the session folder, or ``""`` if nothing was written.
        """
        if not self._sess_active:
            return ""

        self._sess_active = False
        dst = self._sess_dir
        written: list[str] = []
        try:
            for name, track in self._sess_tracks.items():
                y = track["buf"][: track["len"], :]
                if y.size == 0:
                    continue
                if track["n_ch"] == 1:
                    fn = Path(dst) / f"{name}.wav"
                    sf.write(str(fn), y.reshape(-1), self.cfg.fs, subtype="PCM_16")
                    written.append(str(fn))
                else:
                    for m in range(track["n_ch"]):
                        fn = Path(dst) / f"{name}{m + 1:02d}.wav"
                        sf.write(
                            str(fn), y[:, m], self.cfg.fs, subtype="PCM_16"
                        )
                        written.append(str(fn))
        except Exception as ex:                 # pragma: no cover
            log.warning("[Q-WiSE] session flush failed: %s", ex)
            self._reset_session_state()
            return ""

        if not written:
            log.info(
                "[Q-WiSE] Session stopped with no samples — nothing written."
            )
            self._reset_session_state()
            return ""

        log.info(
            "[Q-WiSE] Session stopped -> %s (%d files)", dst, len(written)
        )
        self._reset_session_state()
        return dst

    # ---- Introspection used by the FastAPI playback endpoint -------- #
    def is_recording(self) -> bool:
        return self._sess_active

    def session_dir(self) -> str:
        """Active session folder, or ``""`` when idle."""
        return self._sess_dir if self._sess_active else ""

    def session_tracks(self) -> list[dict]:
        """Per-track summary ``{name, n_ch, samples}`` for the active session."""
        return [
            {"name": n, "n_ch": t["n_ch"], "samples": t["len"]}
            for n, t in self._sess_tracks.items()
        ]

    def list_recordings(self) -> list[dict]:
        """Scan ``<data_dir>/<cfg.record.dir>`` for session folders.

        Returns a newest-first list of ``{dir, name, mtime, files}`` —
        the same shape Task 18's playback grid will consume.
        """
        root = self._records_root()
        if not root.is_dir():
            return []
        out: list[dict] = []
        for entry in root.iterdir():
            if not entry.is_dir():
                continue
            wavs = sorted(p.name for p in entry.glob("*.wav"))
            if not wavs:
                continue
            stat = entry.stat()
            out.append({
                "dir": str(entry),
                "name": entry.name,
                "mtime": stat.st_mtime,
                "files": wavs,
            })
        out.sort(key=lambda d: d["mtime"], reverse=True)
        return out

    # ================================================================== #
    # Internals
    # ================================================================== #
    def _reset_session_state(self) -> None:
        self._sess_tracks = {}
        self._sess_dir = ""

    def _records_root(self) -> Path:
        d = Path(self.cfg.record.dir)
        return d if d.is_absolute() else self.data_dir / d

    def _default_session_path(self) -> Path:
        # Millisecond resolution so back-to-back sessions never collide
        # on the same folder name. ``strftime`` only goes down to ``%f``
        # (microseconds); we truncate the last three digits to keep the
        # path short.
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        sub = self.cfg.record.multi_subdir or "multi"
        name = f"{self.cfg.record.prefix}_{sub}_{stamp}"
        return self._records_root() / name


__all__ = ["AudioIO", "DEFAULT_DATA_DIR"]
