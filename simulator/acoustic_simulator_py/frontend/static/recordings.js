/* =========================================================================
   Q-WiSE recording UI — Task 18

   Drives the Record button + the bottom playback grid:

     * Record button toggles a session via WebSocket controls
       (`recording_start` / `recording_stop`).
     * On a successful `recording_start` ack we flip the button to red
       and update the quota hint.
     * On `recording_stop` ack we refresh the playback grid by hitting
       `GET /api/recordings?limit=1` for the newest session.
     * Each WAV in the latest session becomes a Play/Stop chip; a
       shared <audio> element plays one file at a time.

   Public surface:
     setupRecordingUi()    — call once during boot.

   Listens for these events (dispatched by audio.js / app.js):
     qwise:ws-message    — recording_start / recording_stop acks
     qwise:capture-toggled  — re-enable Record when capture goes live
   ========================================================================= */

import { isOpen, sendControl } from "/static/audio.js";
import { getRecordingsCount, markRecording } from "/static/session.js";


const RECORD_BTN = "#btn-record";
const RECORD_HINT = "#record-hint";
const PLAYBACK_HOST = "#playback-grid";
const QUOTA_HOST = "#quota-info";

let captureOn = false;
let recording = false;
let player = null;        // shared HTMLAudioElement
let activeChip = null;    // togglebutton currently playing
let pendingMark = false;  // true between recording_stop ack and the
                          // grid refresh that follows — so we cache the
                          // newest session exactly once per stop.


/* =====================================================================
   Bootstrap
   ===================================================================== */
export function setupRecordingUi() {
  const btn = document.querySelector(RECORD_BTN);
  if (!btn) return;

  btn.addEventListener("click", onRecordClick);
  document.addEventListener("qwise:capture-toggled", onCaptureToggled);
  document.addEventListener("qwise:ws-message", onWsMessage);

  // Initial playback grid + quota render — defer until DOM settled.
  refreshPlaybackGrid().catch(noop);
  refreshQuota().catch(noop);
}


/* =====================================================================
   Record button — WS recording_start / recording_stop
   ===================================================================== */
function onCaptureToggled(ev) {
  captureOn = !!ev.detail?.capturing;
  syncRecordButton();
}


function onRecordClick() {
  const btn = document.querySelector(RECORD_BTN);
  if (!btn || btn.disabled) return;
  if (!isOpen()) {
    setHint("Start Capture first…");
    return;
  }
  if (recording) {
    sendControl({ type: "recording_stop" });
  } else {
    sendControl({ type: "recording_start" });
  }
}


function onWsMessage(ev) {
  const msg = ev.detail;
  if (!msg || msg.type !== "ack") return;
  if (msg.kind === "recording_start") {
    if (msg.ok) {
      recording = true;
      paintRecording(true);
      setHint(`recording → ${shortPath(msg.path)}`);
    } else if (msg.error === "rate_limited") {
      recording = false;
      paintRecording(false);
      const reset = formatResetTime(msg.quota?.next_reset);
      setHint(`quota reached — try again at ${reset}`);
    } else {
      recording = false;
      paintRecording(false);
      setHint(`could not start: ${msg.error || "unknown error"}`);
    }
    if (msg.quota) updateQuotaPill(msg.quota);
  } else if (msg.kind === "recording_stop") {
    recording = false;
    paintRecording(false);
    setHint("Press Record to capture a session");
    // The latest session just hit disk — refresh the grid and persist
    // its filenames to the client-side cache (Task 19).
    pendingMark = msg.ok === true;
    refreshPlaybackGrid().catch(noop);
    refreshQuota().catch(noop);
  }
}


function syncRecordButton() {
  const btn = document.querySelector(RECORD_BTN);
  if (!btn) return;
  btn.disabled = !captureOn;
  setHint(
    captureOn
      ? (recording ? "recording…" : "Press Record to capture a session")
      : "Start Capture first…"
  );
  if (!captureOn) paintRecording(false);
}


function paintRecording(on) {
  const btn = document.querySelector(RECORD_BTN);
  if (!btn) return;
  btn.setAttribute("aria-pressed", on ? "true" : "false");
  const label = btn.querySelector(".btn-label");
  if (label) label.textContent = on ? "● REC" : "Record";
}


function setHint(text) {
  const el = document.querySelector(RECORD_HINT);
  if (el) el.textContent = text;
}


/* =====================================================================
   Quota pill
   ===================================================================== */
async function refreshQuota() {
  try {
    const r = await fetch("/api/quota");
    if (!r.ok) return;
    updateQuotaPill(await r.json());
  } catch (err) {
    // Silent — the pill is non-critical UX.
  }
}


function updateQuotaPill(quota) {
  const el = document.querySelector(QUOTA_HOST);
  if (!el || !quota) return;
  const remaining = Math.max(0, quota.remaining ?? 0);
  const limit = quota.limit ?? "?";
  const baseline = `quota: ${remaining} / ${limit} recordings left`;
  if (quota.allowed === false) {
    const reset = formatResetTime(quota.next_reset);
    el.textContent = `${baseline} — resets at ${reset}`;
    el.dataset.state = "exhausted";
  } else {
    el.textContent = baseline;
    el.dataset.state = "ok";
  }
}


/* =====================================================================
   Playback grid — latest session only (Task 18 brief)
   ===================================================================== */
async function refreshPlaybackGrid() {
  const host = document.querySelector(PLAYBACK_HOST);
  if (!host) return;

  let listing;
  try {
    const r = await fetch("/api/recordings?limit=1");
    if (!r.ok) {
      host.innerHTML = "";
      return;
    }
    listing = await r.json();
  } catch (err) {
    return;
  }

  // Tear down any active playback before we replace the chips.
  stopPlayer();
  host.innerHTML = "";

  if (!listing.length) {
    host.dataset.empty = "true";
    host.dataset.cached = String(getRecordingsCount());
    return;
  }
  delete host.dataset.empty;

  const session = listing[0];
  for (const file of session.files) {
    host.appendChild(buildChip(session.name, file));
  }

  // Persist exactly once per stop ack: a regular page refresh that
  // hits this code path leaves the cache untouched.
  if (pendingMark) {
    markRecording(session.name, session.files);
    pendingMark = false;
  }
  host.dataset.cached = String(getRecordingsCount());
}


function buildChip(session, file) {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "play-chip";
  btn.dataset.state = "idle";
  btn.dataset.session = session;
  btn.dataset.file = file;
  btn.title = file;
  btn.innerHTML = `<span class="play-icon">▶</span><span class="play-name">${escapeHtml(shortLabel(file))}</span>`;
  btn.addEventListener("click", () => onChipClick(btn));
  return btn;
}


function onChipClick(btn) {
  if (activeChip === btn) {
    stopPlayer();
    return;
  }
  stopPlayer();
  const url = `/api/recordings/${encodeURIComponent(btn.dataset.session)}`
            + `/files/${encodeURIComponent(btn.dataset.file)}`;
  player = new Audio(url);
  player.addEventListener("ended", () => resetChip(btn));
  player.addEventListener("error", () => {
    resetChip(btn);
    console.warn("Q-WiSE: audio playback failed for", url);
  });
  player.play().then(() => {
    activeChip = btn;
    btn.dataset.state = "playing";
    btn.querySelector(".play-icon").textContent = "■";
  }).catch(err => {
    resetChip(btn);
    console.warn("Q-WiSE: audio.play() rejected:", err);
  });
}


function stopPlayer() {
  if (player) {
    try { player.pause(); player.currentTime = 0; } catch {}
    player = null;
  }
  if (activeChip) {
    resetChip(activeChip);
    activeChip = null;
  }
}


function resetChip(btn) {
  if (!btn) return;
  btn.dataset.state = "idle";
  const icon = btn.querySelector(".play-icon");
  if (icon) icon.textContent = "▶";
  if (activeChip === btn) activeChip = null;
}


/* =====================================================================
   Helpers
   ===================================================================== */
function shortLabel(filename) {
  // mic03.wav -> M3 ; mwf.wav -> MWF ; vad.wav -> VAD
  const stem = filename.replace(/\.wav$/i, "");
  const m = stem.match(/^mic0*(\d+)$/i);
  if (m) return `M${m[1]}`;
  return stem.toUpperCase();
}


function shortPath(path) {
  if (!path) return "";
  const parts = path.split("/");
  return parts.slice(-2).join("/");
}


function formatResetTime(epochSec) {
  if (!epochSec) return "—";
  try {
    const d = new Date(epochSec * 1000);
    return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  } catch {
    return "—";
  }
}


function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}


function noop() {}
