/* =========================================================================
   Q-WiSE Acoustic Simulator — frontend entry point

   Coordinates the small per-task modules. Each module owns one
   concern; this file is the glue:

     * Layout interactions: hamburger, source toggle, capture toggle.
     * Bootstrap fetches:   /api/config/default sizes the mic placeholder
                            rows; /api/quota populates the bottom hint.
     * Sidebar accordion (Task 15) — render from /api/config/schema.
     * Audio + WebSocket   (Task 16) — opens /ws/stream, ships mic
                            blocks, forwards `qwise:config-change`
                            CustomEvents as `config_patch` controls.
     * Plotly plots        (Task 17).
     * Recording UI        (Task 18).
     * Session tracking    (Task 19).
   ========================================================================= */

import { setupSidebar } from "/static/sidebar.js";
import {
  closeWebSocket,
  helloCfg,
  isOpen as wsIsOpen,
  openWebSocket,
  sendControl,
  startMic,
  stopMic,
} from "/static/audio.js";
import { setupSpeechUpload } from "/static/upload.js";
import {
  applyGeometry,
  initEmptyPlots,
  initPlots,
  pushAudio,
  pushFrameMeta,
} from "/static/plots.js";
import { setupRecordingUi } from "/static/recordings.js";
import { setupSessionTracking } from "/static/session.js";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

/* ---------- sidebar visibility (hamburger) -------------------------- */
function setupSidebarToggle() {
  const app = $("#app");
  const btn = $("#btn-hamburger");
  const KEY = "qwise.sidebar";
  const saved = localStorage.getItem(KEY);
  if (saved === "closed" || saved === "open") {
    app.dataset.sidebar = saved;
  }
  btn.setAttribute("aria-expanded", app.dataset.sidebar === "open");
  btn.addEventListener("click", () => {
    const next = app.dataset.sidebar === "open" ? "closed" : "open";
    app.dataset.sidebar = next;
    btn.setAttribute("aria-expanded", next === "open");
    localStorage.setItem(KEY, next);
  });
}

/* ---------- source toggle (Mic / WAV) -------------------------------- */
function setupSourceToggle() {
  const group = $("#source-toggle");
  const browse = $("#btn-browse-wav");
  group.addEventListener("click", (ev) => {
    const item = ev.target.closest(".seg-item");
    if (!item) return;
    for (const el of $$(".seg-item", group)) {
      const active = el === item;
      el.classList.toggle("is-active", active);
      el.setAttribute("aria-checked", active ? "true" : "false");
    }
    const source = item.dataset.source;
    browse.disabled = source !== "wav";

    // Tell the active pipeline session which source to read from.
    if (wsIsOpen()) {
      sendControl({ type: "speech_source", source });
    }
    document.dispatchEvent(
      new CustomEvent("qwise:source-changed", { detail: { source } })
    );
  });
}

/* ---------- Start Capture — opens the WS + mic ---------------------- */
function setupCaptureToggle() {
  const btn = $("#btn-capture");
  const pill = $("#status-pill");
  const label = $(".status-label", pill);
  const recBtn = $("#btn-record");
  const recHint = $("#record-hint");
  let busy = false;

  async function go(next) {
    if (busy) return;
    busy = true;
    btn.disabled = true;
    try {
      if (next) {
        await openWebSocket();
        await startMic();
      } else {
        stopMic();
        closeWebSocket();
      }
      btn.setAttribute("aria-pressed", String(next));
      $(".btn-label", btn).textContent = next ? "Stop Capture" : "Start Capture";
      pill.dataset.state = next ? "capturing" : "idle";
      label.textContent = next ? "Capturing" : "Idle";

      recBtn.disabled = !next;
      recHint.textContent = next
        ? "Press Record to capture a session"
        : "Start Capture first…";

      document.dispatchEvent(
        new CustomEvent("qwise:capture-toggled", { detail: { capturing: next } })
      );
    } catch (err) {
      console.warn("Q-WiSE: capture toggle failed:", err);
      stopMic();
      closeWebSocket();
      btn.setAttribute("aria-pressed", "false");
      $(".btn-label", btn).textContent = "Start Capture";
      pill.dataset.state = "error";
      label.textContent = "Mic blocked";
      alert(`Could not start capture: ${err.message ?? err}`);
    } finally {
      busy = false;
      btn.disabled = false;
    }
  }

  btn.addEventListener("click", () => {
    const pressed = btn.getAttribute("aria-pressed") === "true";
    go(!pressed);
  });

  // When the WS receives a `hello`, dump it into the page-wide cfg
  // signal so the sidebar + mic-row code can refresh against the
  // server-resolved values (handles the rare case where the cfg
  // changed between page load and the WS opening).
  document.addEventListener("qwise:ws-open", (ev) => {
    document.dispatchEvent(
      new CustomEvent("qwise:cfg-loaded", { detail: ev.detail?.config ?? {} })
    );
  });
}

/* ---------- mic placeholder rows (sized from /api/config/default) ----- */
function renderMicRows(nMics, refMic) {
  const host = $("#mic-rows");
  host.innerHTML = "";
  for (let m = 1; m <= nMics; m++) {
    const row = document.createElement("div");
    row.className = "mic-row";
    const tag = document.createElement("span");
    tag.className = "mic-tag" + (m === refMic ? " is-ref" : "");
    tag.textContent = m === refMic ? `Mic ${m} (ref)` : `Mic ${m}`;
    const wave = document.createElement("div");
    wave.className = "mic-wave";
    row.appendChild(tag);
    row.appendChild(wave);
    host.appendChild(row);
  }
  $("#mic-array-title").textContent = `Physical ${nMics}-Mic Array`;
}

async function bootstrap() {
  let cfg = null;
  try {
    cfg = await (await fetch("/api/config/default")).json();
    renderMicRows(cfg.n_mics, cfg.mwf.ref_mic);
    document.dispatchEvent(new CustomEvent("qwise:cfg-loaded", { detail: cfg }));
  } catch (err) {
    // Backend not reachable — leave the placeholder text and keep going.
    console.warn("Q-WiSE: /api/config/default unreachable:", err);
  }

  try {
    const q = await (await fetch("/api/quota")).json();
    $("#quota-info").textContent = `quota: ${q.remaining} / ${q.limit} recordings left`;
  } catch (err) {
    console.warn("Q-WiSE: /api/quota unreachable:", err);
  }

  // Render real (empty) Plotly cards so the user sees actual plot
  // frames before pressing Start Capture. The first WS `hello` redraws
  // these against the live geometry + config.
  if (cfg) bootstrapEmptyPlots(cfg).catch(noop);
}


/* ---------- empty-plot bootstrap (renders before any WS data) -------- */
async function bootstrapEmptyPlots(cfg) {
  let geom;
  try {
    geom = await (await fetch("/api/geometry")).json();
  } catch (err) {
    console.warn("Q-WiSE: /api/geometry unreachable:", err);
    return;
  }
  // Plotly loads async — poll briefly so we don't miss its arrival.
  for (let i = 0; i < 50 && typeof window.Plotly === "undefined"; i++) {
    await new Promise(r => setTimeout(r, 60));
  }
  if (typeof window.Plotly === "undefined") return;
  initEmptyPlots(cfg, geom);
}


function noop() {}

/* ---------- forward sidebar edits over the WebSocket ---------------- */
function setupConfigForwarder() {
  document.addEventListener("qwise:config-change", (ev) => {
    const { key, value } = ev.detail || {};
    if (!key) return;
    // n_mics needs a page reload (backend rejects the patch) — skip it.
    if (key === "n_mics") return;
    if (wsIsOpen()) {
      sendControl({ type: "config_patch", patch: { [key]: value } });
    }
  });
}


/* ---------- Plotly wiring (Task 17) --------------------------------- */
function setupPlots() {
  // First `hello` from the server — build the plots against its cfg + geometry.
  document.addEventListener("qwise:ws-open", (ev) => {
    try { initPlots(ev.detail); }
    catch (err) { console.warn("Q-WiSE: initPlots failed:", err); }
  });

  // Every binary frame carries one block of audio (mwf | comp | mics…).
  document.addEventListener("qwise:ws-audio", (ev) => {
    pushAudio(ev.detail?.buffer);
  });

  // The text JSON sibling of each audio block carries vad_score / flags.
  document.addEventListener("qwise:ws-message", (ev) => {
    const msg = ev.detail;
    if (!msg) return;
    if (msg.type === "frame") {
      pushFrameMeta(msg);
    } else if (msg.type === "ack" && msg.ok && msg.geometry) {
      applyGeometry(msg.geometry);
    }
  });
}


/* ---------- entry point --------------------------------------------- */
function main() {
  setupSessionTracking();  // anonymous session-id + recordings cache (Task 19)
  setupSidebarToggle();    // hamburger
  setupSourceToggle();
  setupCaptureToggle();
  bootstrap();
  setupSidebar();          // accordion render (Task 15 module)
  setupSpeechUpload();     // WAV picker + POST (Task 16 module)
  setupConfigForwarder();  // bridge sidebar → WS config_patch
  setupPlots();            // Plotly bindings (Task 17 module)
  setupRecordingUi();      // Record button + playback grid (Task 18)
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", main);
} else {
  main();
}
