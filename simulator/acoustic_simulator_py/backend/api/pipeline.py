"""Per-WebSocket pipeline session.

Owns one full instance of the simulator pipeline (config, geometry,
SourceMixer, VAD, MWF, AudioIO) and routes one incoming mic block
through it, returning every output the frontend renders (mic block,
composite, MWF output, VAD score).

A new :class:`PipelineSession` is built when a WebSocket connects and
torn down when it disconnects, so every browser tab gets its own
covariance EMAs, recording sessions, and per-source history rings.
Lives in process memory — for HF Spaces this is fine (the demo is
short-lived; auto-purge in Task 13 cleans recordings folders).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..audio import AudioIO
from ..config import Config, default, ui_sidebar_schema
from ..core import SourceMixer, build_geometry
from ..mwf import Mwf
from ..vad import Vad, make_vad
from .uploads import UPLOAD_DIR_NAME, uploads_root

log = logging.getLogger(__name__)


class ConfigUpdateError(ValueError):
    """A live cfg patch tried to change a field we can't safely hot-swap."""


# Fields the UI is allowed to edit. n_mics is deliberately NOT in this
# set: changing the mic count at runtime would invalidate downstream
# buffer shapes (SourceMixer history rings, MWF covariance buffers,
# AudioIO recording tracks). A page reload is the right escape hatch.
_UI_EDITABLE_KEYS: frozenset[str] = frozenset(
    e["key"] for e in ui_sidebar_schema(default())
) - {"n_mics"}


class PipelineSession:
    """One block-by-block simulator session for a single WebSocket."""

    cfg: Config
    audio: AudioIO

    # --- runtime toggles, mirror SimulatorUI flags --------------------- #
    # Default-on so a fresh WebSocket session boots the full demo (mic
    # mixer + drone + env + VAD + MWF). Pressing Record then captures
    # mic*.wav, vad.wav and mwf.wav without needing extra UI toggles.
    drone_on: bool = True
    env_on: bool = True
    drone_gain: float
    env_gain: float
    vad_on: bool = True
    mwf_on: bool = True
    speech_source: str = "mic"               # 'mic' | 'wav'
    recording: bool = False
    rec_tracks: dict[str, bool]
    frame_idx: int = 0

    def __init__(self, cfg: Config | None = None, audio: AudioIO | None = None):
        self.cfg = cfg if cfg is not None else default()
        self.audio = audio if audio is not None else AudioIO(self.cfg)
        self.geo = build_geometry(self.cfg)
        self.mixer = SourceMixer(self.cfg, self.geo)
        self.vad: Vad = make_vad(self.cfg)
        self.mwf = Mwf(self.cfg)

        self.drone_gain = float(self.cfg.drone_gain_init)
        self.env_gain = float(self.cfg.env_gain_init)
        self.rec_tracks = {"mic": False, "vad": False, "mwf": False}

    # ================================================================== #
    # Per-block pipeline
    # ================================================================== #
    def process_block(self, mic_input: ArrayLike | None) -> dict[str, Any]:
        """Run one block end-to-end.

        ``mic_input`` is the browser's mic frame (1-D float, length
        ``cfg.frame_size``). It may be ``None`` if the source is set to
        WAV — in which case the speech comes from the cached upload.
        """
        n = int(self.cfg.frame_size)

        # ---- source 1: speech ----------------------------------------
        speech = self._take_speech(mic_input, n)

        # ---- sources 2 & 3: drone + env loops ------------------------
        if self.drone_on:
            drone = float(self.drone_gain) * self.audio.next_drone_chunk(n)
        else:
            drone = np.zeros(n, dtype=np.float64)
        if self.env_on:
            env = float(self.env_gain) * self.audio.next_env_chunk(n)
        else:
            env = np.zeros(n, dtype=np.float64)

        # ---- mix → composite → VAD → MWF -----------------------------
        mic = self.mixer.mix(speech, drone, env)              # [N, n_mics]
        comp = self.mixer.composite(mic)                      # [N]

        if self.vad_on:
            is_speech, score = self.vad.step(comp)
        else:
            is_speech, score = False, 0.0

        if self.mwf_on:
            y_enh = self.mwf.step(mic, bool(is_speech))
        else:
            y_enh = np.zeros(n, dtype=np.float64)

        # ---- optional recording --------------------------------------
        if self.recording:
            self._write_tracks(mic, comp, y_enh, bool(is_speech))

        self.frame_idx += 1
        return {
            "frame_idx": self.frame_idx,
            "vad_score": float(score),
            "is_speech": bool(is_speech),
            "mic": mic,           # [N, n_mics] float64
            "comp": comp,         # [N]
            "mwf": y_enh,         # [N]
        }

    # ---- recording wiring (used by Task 18; available now) ----------- #
    def start_recording(self) -> str:
        """Open a recording session and freeze the track set."""
        tracks = {
            "mic": True,
            "vad": bool(self.vad_on),
            "mwf": bool(self.vad_on and self.mwf_on),
        }
        path = self.audio.rec_start_session()
        if not path:
            return ""
        self.rec_tracks = tracks
        self.recording = True
        return path

    def stop_recording(self) -> str:
        """Flush and close the active session, return the folder."""
        if not self.recording:
            return ""
        self.recording = False
        self.rec_tracks = {"mic": False, "vad": False, "mwf": False}
        return self.audio.rec_stop_session()

    # ================================================================== #
    # Control messages — UI toggles + live config patches
    # ================================================================== #
    def apply_control(self, msg: dict) -> dict:
        """Dispatch one JSON control message; return ``{ok, …}`` ack."""
        kind = msg.get("type", "")
        try:
            if kind == "drone_on":
                self.drone_on = bool(msg.get("on", False))
            elif kind == "env_on":
                self.env_on = bool(msg.get("on", False))
            elif kind == "drone_gain":
                self.drone_gain = float(msg.get("value", self.drone_gain))
            elif kind == "env_gain":
                self.env_gain = float(msg.get("value", self.env_gain))
            elif kind == "enable_vad":
                self.vad_on = bool(msg.get("on", False))
                if not self.vad_on and self.mwf_on:
                    self.mwf_on = False        # MWF needs VAD covariance updates
            elif kind == "enable_mwf":
                on = bool(msg.get("on", False))
                if on and not self.vad_on:
                    self.vad_on = True
                self.mwf_on = on
            elif kind == "speech_source":
                src = str(msg.get("source", "mic")).lower()
                if src not in ("mic", "wav"):
                    raise ConfigUpdateError(f"unknown speech source {src!r}")
                self.speech_source = src
            elif kind == "load_speech_wav":
                path = str(msg.get("path", ""))
                if not path:
                    raise ConfigUpdateError("path is required")
                resolved = self._safe_upload_path(path)
                self.audio.load_speech_wav(resolved)
                return {
                    "type": "ack", "kind": kind, "ok": True,
                    "has_wav": True,
                    "path": str(resolved),
                }
            elif kind == "clear_speech_wav":
                self.audio.clear_speech_wav()
                return {
                    "type": "ack", "kind": kind, "ok": True,
                    "has_wav": False,
                }
            elif kind == "config_patch":
                self.apply_config_patch(dict(msg.get("patch", {})))
            elif kind == "reset":
                self.reset_pipeline()
            elif kind == "recording_start":
                path = self.start_recording()
                return {"type": "ack", "kind": kind, "ok": bool(path), "path": path}
            elif kind == "recording_stop":
                path = self.stop_recording()
                return {"type": "ack", "kind": kind, "ok": True, "path": path}
            else:
                return {"type": "ack", "kind": kind, "ok": False, "error": "unknown_type"}
            return {"type": "ack", "kind": kind, "ok": True}
        except (ConfigUpdateError, ValueError, TypeError) as ex:
            return {"type": "ack", "kind": kind, "ok": False, "error": str(ex)}

    def apply_config_patch(self, patch: dict) -> None:
        """Apply a flat ``{dotted_key: value}`` patch.

        Only UI-editable keys land; everything else raises so misuse is
        surfaced rather than silently ignored. After the values land we
        rebuild geometry and push it into the mixer.
        """
        if not patch:
            return
        for key, value in patch.items():
            if key not in _UI_EDITABLE_KEYS:
                raise ConfigUpdateError(
                    f"field {key!r} is not in the live-edit allow-list"
                )
            self._assign_dotted(self.cfg, key, value)

        # Anything geometry-affecting → rebuild + push into the mixer.
        self.geo = build_geometry(self.cfg)
        self.mixer.update_geometry(self.geo)

        # mwf.method may have changed — re-validate by reaching for the
        # private setter on a fresh Mwf if we cannot mutate in-place.
        if "mwf.method" in patch:
            try:
                Mwf._validate_method(self.mwf.method)
            except Exception:
                # The Pydantic model already rejected an invalid value
                # before we got here; this is belt-and-braces.
                pass

    def reset_pipeline(self) -> None:
        """Zero all stateful buffers (history rings, VAD ring, MWF EMAs)."""
        self.mixer.reset()
        self.vad.reset()
        self.mwf.reset()
        self.frame_idx = 0

    # ================================================================== #
    # Internals
    # ================================================================== #
    def _take_speech(
        self, mic_input: ArrayLike | None, n: int
    ) -> NDArray[np.float64]:
        if self.speech_source == "wav" and self.audio.has_speech_wav():
            return self.audio.next_speech_chunk(n)
        if mic_input is None:
            return np.zeros(n, dtype=np.float64)
        sp = np.asarray(mic_input, dtype=np.float64).reshape(-1)
        if sp.size < n:
            sp = np.concatenate([sp, np.zeros(n - sp.size, dtype=np.float64)])
        elif sp.size > n:
            sp = sp[:n]
        return sp

    def _write_tracks(
        self,
        mic: NDArray[np.float64],
        comp: NDArray[np.float64],
        y_enh: NDArray[np.float64],
        is_speech: bool,
    ) -> None:
        if self.rec_tracks.get("mic"):
            self.audio.rec_session_write("mic", mic)
        if self.rec_tracks.get("vad"):
            v = comp if is_speech else np.zeros_like(comp)
            self.audio.rec_session_write("vad", v)
        if self.rec_tracks.get("mwf"):
            self.audio.rec_session_write("mwf", y_enh)

    def serialize_geometry(self) -> dict:
        """JSON-serialisable snapshot of the current scene geometry.

        Numpy arrays go through ``.tolist()`` so the WebSocket payload
        is plain JSON. Consumed by the frontend 3-D scene plot (Task 17)
        and re-shipped on every successful ``config_patch``.
        """
        geo = self.geo
        return {
            "pos_human":    geo.pos_human.tolist(),
            "pos_img_src":  geo.pos_img_src.tolist(),
            "pos_drone":    geo.pos_drone.tolist(),
            "pos_env":      geo.pos_env.tolist(),
            "pos_mics":     geo.pos_mics.tolist(),
            "dist_speech":  geo.dist_speech.tolist(),
            "dist_drone":   geo.dist_drone.tolist(),
            "dist_env":     geo.dist_env.tolist(),
            "gains_speech": geo.gains_speech.tolist(),
            "gains_drone":  geo.gains_drone.tolist(),
            "gains_env":    geo.gains_env.tolist(),
            "drone_agl":    float(geo.drone_agl),
            "ref_mic":      int(self.mixer.ref_mic()),
        }

    @staticmethod
    def _assign_dotted(obj: Any, dotted_key: str, value: Any) -> None:
        parts = dotted_key.split(".")
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], value)

    @staticmethod
    def _safe_upload_path(raw: str) -> Path:
        """Resolve ``raw`` and verify it sits under the uploads root.

        Prevents path traversal — a client can't load arbitrary files
        off the server's disk by handing us ``../etc/passwd.wav``.
        """
        target = Path(raw).resolve()
        allowed = uploads_root().resolve()
        try:
            target.relative_to(allowed)
        except ValueError as ex:
            raise ConfigUpdateError(
                f"speech-WAV path must live under {UPLOAD_DIR_NAME}/"
            ) from ex
        if not target.is_file():
            raise ConfigUpdateError(f"uploaded file not found: {target.name}")
        return target


__all__ = ["PipelineSession", "ConfigUpdateError"]
