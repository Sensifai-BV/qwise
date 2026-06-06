"""Tests for :mod:`backend.audio`.

Covers the loop helpers (``load_wav_loop`` + ``loop_chunk``) and the
``AudioIO`` session-recording API. The hardware mic / speaker code is
intentionally absent in the web port, so there is no equivalent of
``test_audio_device_*`` from the MATLAB suite.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from backend.audio import AudioIO, load_wav_loop, loop_chunk
from backend.config import default


# --------------------------------------------------------------------- #
# loop_chunk
# --------------------------------------------------------------------- #
def test_loop_chunk_wraps_modulo() -> None:
    wav = np.arange(10, dtype=np.float64)
    # Single hop inside the buffer.
    np.testing.assert_array_equal(loop_chunk(wav, 0, 4), [0, 1, 2, 3])
    np.testing.assert_array_equal(loop_chunk(wav, 7, 5), [7, 8, 9, 0, 1])
    # Multiple wraps.
    np.testing.assert_array_equal(loop_chunk(wav, 8, 22)[:12], [8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])


def test_loop_chunk_empty_buffer_returns_silence() -> None:
    out = loop_chunk(np.zeros(0, dtype=np.float64), 0, 16)
    assert out.shape == (16,)
    assert np.all(out == 0)


# --------------------------------------------------------------------- #
# load_wav_loop
# --------------------------------------------------------------------- #
def test_load_wav_loop_normalises_to_unit_rms() -> None:
    cfg = default()
    y = load_wav_loop(cfg.drone_wav_path, cfg.fs, 1.0)   # 1 s loop
    assert y.shape == (cfg.fs,)
    assert float(np.sqrt(np.mean(y * y))) == pytest.approx(1.0, abs=1e-6)


def test_load_wav_loop_missing_falls_back_to_pink_noise(tmp_path) -> None:
    cfg = default()
    missing = tmp_path / "this_does_not_exist.wav"
    y = load_wav_loop(missing, cfg.fs, 0.5)
    assert y.shape == (int(0.5 * cfg.fs),)
    # Pink-noise placeholder is also normalised to unit RMS.
    assert float(np.sqrt(np.mean(y * y))) == pytest.approx(1.0, abs=1e-6)


def test_load_wav_loop_resamples_when_fs_mismatches(tmp_path) -> None:
    """A 8 kHz, 0.5 s tone loaded at fs=16 kHz / 1 s must yield 16 000
    samples and round-trip the original frequency."""
    src_fs = 8000
    f0 = 440
    duration = 0.5
    t = np.arange(int(src_fs * duration)) / src_fs
    tone = 0.5 * np.sin(2 * np.pi * f0 * t)
    path = tmp_path / "tone8k.wav"
    sf.write(str(path), tone, src_fs, subtype="PCM_16")

    target_fs = 16_000
    y = load_wav_loop(path, target_fs, 1.0)
    assert y.shape == (target_fs,)
    # Locate the dominant frequency in the spectrum — should be ~440 Hz.
    spec = np.abs(np.fft.rfft(y))
    f_axis = np.fft.rfftfreq(len(y), d=1 / target_fs)
    peak_freq = float(f_axis[int(np.argmax(spec))])
    assert abs(peak_freq - f0) < 5.0    # ±5 Hz tolerance


# --------------------------------------------------------------------- #
# AudioIO — looped sources
# --------------------------------------------------------------------- #
def test_audioio_loads_bundled_loops() -> None:
    cfg = default()
    aio = AudioIO(cfg)
    n = cfg.frame_size
    drone = aio.next_drone_chunk(n)
    env = aio.next_env_chunk(n)
    assert drone.shape == (n,)
    assert env.shape == (n,)
    assert np.all(np.isfinite(drone))
    assert np.all(np.isfinite(env))
    # The full loop is unit-RMS by construction. Per-frame RMS can be
    # arbitrary (env_ambient.wav is near-silence at its boundary, for
    # example), so we only check the whole-loop normalisation contract.
    assert float(np.sqrt(np.mean(aio.drone_wav ** 2))) == pytest.approx(1.0, abs=1e-6)
    assert float(np.sqrt(np.mean(aio.env_wav ** 2))) == pytest.approx(1.0, abs=1e-6)


def test_audioio_pointer_wraps_back_to_start() -> None:
    cfg = default()
    aio = AudioIO(cfg)
    n = cfg.frame_size
    head = aio.next_drone_chunk(n).copy()
    # Walk the pointer all the way around the loop.
    loop_len = aio.drone_wav.shape[0]
    full = loop_len - n
    while full > 0:
        step = min(full, 4 * n)
        aio.next_drone_chunk(step)
        full -= step
    # Next chunk should match the first chunk byte-for-byte.
    again = aio.next_drone_chunk(n)
    np.testing.assert_allclose(again, head, atol=0.0)


# --------------------------------------------------------------------- #
# AudioIO — clean-speech WAV
# --------------------------------------------------------------------- #
def test_audioio_speech_wav_round_trip(tmp_path) -> None:
    cfg = default()
    aio = AudioIO(cfg)
    assert aio.has_speech_wav() is False

    fs = cfg.fs
    t = np.arange(fs) / fs
    voiced = 0.7 * np.sin(2 * np.pi * 220 * t)
    path = tmp_path / "speech.wav"
    sf.write(str(path), voiced, fs, subtype="PCM_16")

    aio.load_speech_wav(path)
    assert aio.has_speech_wav()
    assert aio.speech_wav_path.endswith("speech.wav")
    # Peak-normalised to ~0.9 by load_speech_wav.
    assert float(np.max(np.abs(aio.speech_wav))) == pytest.approx(0.9, abs=1e-2)

    # Chunks consume from the same buffer at fs.
    n = cfg.frame_size
    c1 = aio.next_speech_chunk(n)
    c2 = aio.next_speech_chunk(n)
    assert c1.shape == (n,) and c2.shape == (n,)
    assert not np.allclose(c1, 0.0)

    aio.clear_speech_wav()
    assert aio.has_speech_wav() is False
    np.testing.assert_array_equal(aio.next_speech_chunk(n), np.zeros(n))


def test_audioio_speech_wav_missing_file_raises() -> None:
    cfg = default()
    aio = AudioIO(cfg)
    with pytest.raises(Exception):
        aio.load_speech_wav("/tmp/__definitely_missing__.wav")


# --------------------------------------------------------------------- #
# Session recording
# --------------------------------------------------------------------- #
def _stage_aio(tmp_path: Path) -> AudioIO:
    cfg = default()
    return AudioIO(cfg, data_dir=tmp_path)


def test_session_writes_one_wav_per_track(tmp_path) -> None:
    cfg = default()
    aio = _stage_aio(tmp_path)

    sess = aio.rec_start_session()
    assert aio.is_recording()
    assert aio.session_dir() == sess

    n = cfg.frame_size
    # Multi-channel "mic" + mono "vad" + mono "mwf" — what the GUI writes.
    mic_block = 0.5 * np.random.default_rng(0).standard_normal((n, cfg.n_mics))
    vad_block = 0.3 * np.random.default_rng(1).standard_normal(n)
    mwf_block = 0.2 * np.random.default_rng(2).standard_normal(n)

    for _ in range(3):
        aio.rec_session_write("mic", mic_block)
        aio.rec_session_write("vad", vad_block)
        aio.rec_session_write("mwf", mwf_block)

    tracks = {t["name"]: t for t in aio.session_tracks()}
    assert tracks["mic"]["n_ch"] == cfg.n_mics
    assert tracks["vad"]["n_ch"] == 1
    assert tracks["mwf"]["n_ch"] == 1
    assert tracks["mic"]["samples"] == 3 * n

    dst = aio.rec_stop_session()
    assert dst == sess
    assert aio.is_recording() is False
    assert aio.session_dir() == ""

    files = sorted(p.name for p in Path(dst).glob("*.wav"))
    # One per channel for mic, one each for vad / mwf.
    assert files == sorted(
        [f"mic{m + 1:02d}.wav" for m in range(cfg.n_mics)]
        + ["vad.wav", "mwf.wav"]
    )

    # Round-trip read back: mic01 has the first column of the staged block.
    mic1, fs_read = sf.read(str(Path(dst) / "mic01.wav"))
    assert fs_read == cfg.fs
    assert mic1.shape == (3 * n,)
    # PCM_16 quantisation tolerance.
    np.testing.assert_allclose(
        mic1[:n], mic_block[:, 0], atol=1.0 / 32768, err_msg="mic01 round-trip lost data"
    )


def test_session_with_no_writes_yields_empty_string(tmp_path) -> None:
    aio = _stage_aio(tmp_path)
    aio.rec_start_session()
    out = aio.rec_stop_session()
    assert out == ""              # nothing written → empty return
    assert aio.is_recording() is False


def test_session_rejects_bad_track_name(tmp_path) -> None:
    cfg = default()
    aio = _stage_aio(tmp_path)
    aio.rec_start_session()
    # Track names must be valid identifiers (used as file stems).
    aio.rec_session_write("not a name", np.zeros(cfg.frame_size))
    assert aio.session_tracks() == []
    aio.rec_stop_session()


def test_double_start_warns_but_returns_active_dir(tmp_path, caplog) -> None:
    aio = _stage_aio(tmp_path)
    first = aio.rec_start_session()
    second = aio.rec_start_session()
    assert second == first
    aio.rec_stop_session()


# --------------------------------------------------------------------- #
# list_recordings
# --------------------------------------------------------------------- #
def test_list_recordings_sorts_newest_first(tmp_path) -> None:
    import time
    cfg = default()
    aio = _stage_aio(tmp_path)

    # Record + stop three sessions back-to-back; the third is newest.
    paths = []
    for i in range(3):
        aio.rec_start_session()
        aio.rec_session_write("mic", np.random.default_rng(i).standard_normal((cfg.frame_size, cfg.n_mics)))
        paths.append(aio.rec_stop_session())
        time.sleep(0.01)            # ensure mtime ordering is monotone

    listing = aio.list_recordings()
    assert len(listing) == 3
    names = [d["name"] for d in listing]
    assert names[0] == Path(paths[-1]).name      # newest first
    assert names[-1] == Path(paths[0]).name
    # Every entry exposes the wav file list — Task 18 will render this directly.
    for entry in listing:
        assert all(name.endswith(".wav") for name in entry["files"])
