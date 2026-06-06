"""Tests for the frontend skeleton — static assets reachable, HTML
contains the layout regions Task 15+ expects to hydrate.

The actual UX is verified by hand in a browser; this suite just guards
against the FastAPI ``StaticFiles`` mount / index route getting unwired.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.api.app import app

client = TestClient(app)


# --------------------------------------------------------------------- #
# / — serves index.html
# --------------------------------------------------------------------- #
def test_index_html_serves_layout() -> None:
    r = client.get("/")
    assert r.status_code == 200
    html = r.text
    # Top-level layout containers later tasks need to hydrate.
    for hook in (
        'id="app"', 'class="topbar"', 'class="canvas"', 'class="plot-grid"',
        'id="sidebar"', 'id="sidebar-body"',
        'id="mic-rows"', 'id="card-vad-trace"',
        'id="btn-capture"', 'id="btn-hamburger"', 'id="btn-record"',
        'id="quota-info"', 'id="playback-grid"',
    ):
        assert hook in html, f"index.html missing hook: {hook}"


def test_index_html_links_to_static_assets() -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert '/static/styles.css' in r.text
    assert '/static/app.js' in r.text


# --------------------------------------------------------------------- #
# /static/* — CSS + JS payloads
# --------------------------------------------------------------------- #
def test_styles_css_is_served() -> None:
    r = client.get("/static/styles.css")
    assert r.status_code == 200
    assert "text/css" in r.headers.get("content-type", "")
    # Cheap content-shape check so a stripped-down rebuild doesn't slip past.
    body = r.text
    assert "--accent" in body
    assert "#469e00" in body
    assert ".plot-grid" in body


def test_app_js_is_served() -> None:
    r = client.get("/static/app.js")
    assert r.status_code == 200
    body = r.text
    # Should pull both config + quota at boot (Task 15 / 13 wiring).
    assert "/api/config/default" in body
    assert "/api/quota" in body
    # Hamburger toggle is the layout's load-bearing interaction.
    assert "dataset.sidebar" in body
    assert "btn-hamburger" in body
    # Task 15 import — accordion comes in as a separate module.
    assert "sidebar.js" in body
    assert "setupSidebar" in body


# --------------------------------------------------------------------- #
# /static/sidebar.js — Task 15 accordion module
# --------------------------------------------------------------------- #
def test_sidebar_js_is_served() -> None:
    r = client.get("/static/sidebar.js")
    assert r.status_code == 200
    body = r.text
    # Must reach the schema endpoint and dispatch the change event the
    # WebSocket wiring (Task 16) expects to forward as ``config_patch``.
    assert "/api/config/schema" in body
    assert "qwise:config-change" in body
    # n_mics needs a page reload — surfaced in the UI as a hint.
    assert "n_mics" in body
    # Both widget types are constructed.
    assert "buildSlider" in body
    assert "buildSelect" in body
    # localStorage persistence key for accordion open/closed state.
    assert "qwise.sidebar.sections" in body


def test_styles_carry_accordion_rules() -> None:
    """Task 14 reserved the CSS for the accordion; Task 15 must keep it
    intact so the live render lands styled."""
    body = client.get("/static/styles.css").text
    assert ".accordion-section" in body
    assert ".accordion-head" in body
    assert ".accordion-body" in body
    assert ".field-row" in body
    assert ".field-hint" in body
    # Task 16: WAV filename badge.
    assert ".wav-name" in body


# --------------------------------------------------------------------- #
# Task 16: audio + upload modules
# --------------------------------------------------------------------- #
def test_audio_js_is_served() -> None:
    r = client.get("/static/audio.js")
    assert r.status_code == 200
    body = r.text
    assert "/ws/stream" in body
    assert "getUserMedia" in body
    assert "ScriptProcessor" in body or "AudioContext" in body
    assert "qwise:ws-open" in body
    assert "qwise:ws-audio" in body
    # Public exports the rest of the app pulls in.
    for sym in ("openWebSocket", "closeWebSocket", "startMic",
                "stopMic", "sendControl", "isOpen"):
        assert f"export " in body and sym in body, f"missing {sym}"


def test_upload_js_is_served() -> None:
    r = client.get("/static/upload.js")
    assert r.status_code == 200
    body = r.text
    assert "/api/speech-wav" in body
    assert "load_speech_wav" in body
    assert "speech_source" in body
    assert "setupSpeechUpload" in body


def test_app_js_wires_capture_and_upload_modules() -> None:
    body = client.get("/static/app.js").text
    assert "audio.js" in body
    assert "upload.js" in body
    assert "startMic" in body
    assert "stopMic" in body
    assert "setupSpeechUpload" in body
    # Sidebar config-change events get forwarded as config_patch.
    assert "config_patch" in body
    # Task 17 wiring: plots module + per-frame ingest hooks.
    assert "plots.js" in body
    assert "initPlots" in body
    assert "pushAudio" in body
    assert "pushFrameMeta" in body


# --------------------------------------------------------------------- #
# Task 17: plots + FFT modules
# --------------------------------------------------------------------- #
def test_fft_js_is_served() -> None:
    r = client.get("/static/fft.js")
    assert r.status_code == 200
    body = r.text
    assert "fftMagnitudeDb" in body
    assert "periodicHann" in body
    # Iterative radix-2 — guards against the algorithm being dropped.
    assert "Cooley" in body or "butterflies" in body.lower()
    assert "export" in body


def test_plots_js_is_served() -> None:
    r = client.get("/static/plots.js")
    assert r.status_code == 200
    body = r.text
    # Six plot regions hydrated by this module.
    for hook in ("card-scene", "card-spec-noisy", "card-spec-mwf",
                 "mic-rows", "card-mwf-wave", "card-vad-trace"):
        assert hook in body, f"plots.js missing {hook}"
    # Public symbols app.js imports.
    for sym in ("initPlots", "pushAudio", "pushFrameMeta", "applyGeometry"):
        assert f"export " in body and sym in body
    # FFT module is the only dependency.
    assert "/static/fft.js" in body
    # Plotly is loaded externally — the module gracefully degrades.
    assert "Plotly" in body


def test_index_html_loads_plotly_cdn() -> None:
    body = client.get("/").text
    assert "cdn.plot.ly" in body
    assert "plotly" in body.lower()


# --------------------------------------------------------------------- #
# Task 18: recordings module + playback grid
# --------------------------------------------------------------------- #
def test_recordings_js_is_served() -> None:
    r = client.get("/static/recordings.js")
    assert r.status_code == 200
    body = r.text
    # Hooks into the WS via control messages.
    assert "recording_start" in body
    assert "recording_stop"  in body
    # REST endpoints it touches (uses encodeURIComponent → no literal slashes).
    assert "/api/recordings" in body
    assert "/api/quota" in body
    # Public bootstrap symbol the entry point calls.
    assert "setupRecordingUi" in body
    # Shared <audio> element behaviour: only one plays at a time.
    assert "activeChip" in body or "stopPlayer" in body


def test_app_js_wires_recordings_module() -> None:
    body = client.get("/static/app.js").text
    assert "recordings.js" in body
    assert "setupRecordingUi" in body


def test_styles_carry_playback_chip_rules() -> None:
    body = client.get("/static/styles.css").text
    assert ".play-chip" in body
    assert ".bottom-grid[data-empty" in body
    assert ".quota-info[data-state" in body


# --------------------------------------------------------------------- #
# Task 19: session.js — cookie + localStorage cache
# --------------------------------------------------------------------- #
def test_session_js_is_served() -> None:
    r = client.get("/static/session.js")
    assert r.status_code == 200
    body = r.text
    # Every exported entry point used elsewhere in the bundle.
    for sym in ("setupSessionTracking", "getSessionId", "markRecording",
                "getRecordings", "clearRecordings", "getRecordingsCount",
                "pruneRecordings"):
        assert f"export " in body and sym in body, f"session.js missing {sym}"
    # Storage + cookie contract.
    assert "qwise_sid" in body
    assert "qwise.session.v1" in body
    assert "SameSite=Lax" in body
    # 3-hour TTL mirrors the server's cleanup window.
    assert "3 * 3600 * 1000" in body or "10800000" in body
    # UUID v4 generation works without crypto.randomUUID too.
    assert "randomUUID" in body
    assert "getRandomValues" in body


def test_app_js_wires_session_tracking() -> None:
    body = client.get("/static/app.js").text
    assert "session.js" in body
    assert "setupSessionTracking" in body


def test_recordings_js_marks_after_stop() -> None:
    """The recording_stop ack path must persist the latest session to
    the client cache exactly once."""
    body = client.get("/static/recordings.js").text
    assert "markRecording" in body
    assert "pendingMark" in body
    # Empty-state hint surfaces the cached count to dev tools.
    assert "data-cached" in body or "dataset.cached" in body
