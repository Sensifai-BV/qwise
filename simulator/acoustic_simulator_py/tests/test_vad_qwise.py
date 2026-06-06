"""Tests for :class:`backend.vad.QwiseVad`.

These tests ship with the project model (``models/qwise_vad.onnx``).
They're skipped on environments where ``onnxruntime`` or the model file
is missing, so the rest of the suite stays green in those installs.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.config import default
from backend.vad import CONTEXT_SIZE, STATE_SHAPE, QwiseVad


@pytest.fixture
def cfg():
    return default()


@pytest.fixture
def vad(cfg) -> QwiseVad:
    v = QwiseVad(cfg)
    if not v.ready:
        pytest.skip("onnxruntime or the VAD ONNX model is unavailable")
    return v


# --------------------------------------------------------------------- #
# Loading / introspection
# --------------------------------------------------------------------- #
def test_loads_onnx_session(vad) -> None:
    """The constructor sets ``ready`` and the input/output name lists."""
    assert vad.ready
    assert vad._session is not None
    assert vad._in_names, "ORT inputs were not discovered"
    assert vad._out_names, "ORT outputs were not discovered"


def test_initial_state_and_context_shapes(vad) -> None:
    assert vad._state.shape == STATE_SHAPE
    assert vad._state.dtype == np.float32
    assert vad._ctx.shape == (CONTEXT_SIZE,)
    assert vad._ctx.dtype == np.float32


# --------------------------------------------------------------------- #
# Score behaviour: silence vs voiced
# --------------------------------------------------------------------- #
def test_silence_score_is_low(cfg, vad) -> None:
    """Pure silence at the host frame size keeps the score below the
    Q-WiSE threshold for every block in a short run."""
    n = cfg.frame_size
    silence = np.zeros(n, dtype=np.float32)
    seen_speech = False
    for _ in range(6):
        is_speech, score = vad.step(silence)
        if is_speech:
            seen_speech = True
        assert score < 0.5, f"silence score {score:.3f} crossed 0.5"
    assert not seen_speech, "silence was reported as speech"


def test_voiced_tone_eventually_speech(cfg, vad) -> None:
    """A voiced-tone proxy reaches the speech threshold within a few
    frames. The exact rise time depends on the LSTM warm-up; we accept
    anywhere in the first 8 blocks."""
    n = cfg.frame_size
    t = np.arange(n) / cfg.fs
    voiced = (
        0.40 * np.sin(2 * np.pi * 220 * t)
        + 0.20 * np.sin(2 * np.pi * 440 * t)
        + 0.10 * np.sin(2 * np.pi * 880 * t)
    ).astype(np.float32)
    max_score = 0.0
    for _ in range(8):
        is_speech, score = vad.step(voiced)
        max_score = max(max_score, score)
        if is_speech:
            return
    pytest.fail(f"voiced proxy never crossed threshold (max score={max_score:.3f})")


# --------------------------------------------------------------------- #
# State + context threading
# --------------------------------------------------------------------- #
def test_step_threads_audio_context_between_calls(cfg, vad) -> None:
    """After processing a chunk, ``_ctx`` must equal the trailing 64
    samples of the last processed sub-block. This is the upstream
    model's key contract; if it breaks, every other block returns ~0."""
    n = cfg.vad.qwise_frame  # 512
    rng = np.random.default_rng(4)
    chunk = rng.standard_normal(n).astype(np.float32) * 0.1
    vad.step(chunk)
    np.testing.assert_allclose(vad._ctx, chunk[-CONTEXT_SIZE:], atol=1e-6)


def test_reset_zeros_state_and_context(cfg, vad) -> None:
    n = cfg.frame_size
    t = np.arange(n) / cfg.fs
    voiced = 0.30 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    for _ in range(2):
        vad.step(voiced)
    assert np.any(vad._ctx != 0)  # context populated
    vad.reset()
    assert np.all(vad._state == 0)
    assert np.all(vad._ctx == 0)


# --------------------------------------------------------------------- #
# Fallback path
# --------------------------------------------------------------------- #
def test_not_ready_returns_silence(cfg) -> None:
    """Construct a VAD that points at a missing model path — it should
    refuse to crash, stay ``ready=False``, and return ``(False, 0.0)``
    for any input."""
    cfg2 = default()
    cfg2.vad.onnx_path = "models/__definitely_missing__.onnx"
    v = QwiseVad(cfg2)
    assert v.ready is False
    n = cfg2.frame_size
    is_speech, score = v.step(np.random.default_rng(5).standard_normal(n))
    assert is_speech is False
    assert score == 0.0
