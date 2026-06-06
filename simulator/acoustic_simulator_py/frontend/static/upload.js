/* =========================================================================
   Q-WiSE speech-WAV upload — Task 16

   Wires the "WAV…" button next to the source toggle: opens a hidden
   <input type="file">, POSTs the selected file to `/api/speech-wav`,
   shows the loaded filename + duration, and instructs the active
   WebSocket session to load the saved path via two control messages
   (`load_speech_wav` then `speech_source: 'wav'`).
   ========================================================================= */

import { sendControl, isOpen } from "/static/audio.js";


export function setupSpeechUpload() {
  const btn = document.querySelector("#btn-browse-wav");
  if (!btn) return;
  const label = ensureLabel(btn);

  btn.addEventListener("click", async () => {
    let file;
    try {
      file = await pickFile();
    } catch (err) {
      return;   // user cancelled
    }

    const oldText = label.textContent;
    label.textContent = `Uploading ${file.name}…`;
    label.dataset.state = "loading";

    try {
      const meta = await uploadFile(file);
      label.textContent = `WAV: ${meta.name} (${meta.duration.toFixed(1)} s)`;
      label.dataset.state = "loaded";

      // Forward to the active WS pipeline. Both messages are best-effort
      // — if the WS hasn't been opened yet (Capture not started), the
      // user just needs to flip Source → WAV after starting Capture.
      if (isOpen()) {
        sendControl({ type: "load_speech_wav", path: meta.path });
        sendControl({ type: "speech_source", source: "wav" });
      }

      // Tell the segmented control to switch to WAV in the UI.
      const wavSeg = document.querySelector(
        '#source-toggle .seg-item[data-source="wav"]'
      );
      wavSeg?.click();
    } catch (err) {
      label.textContent = oldText || "";
      label.dataset.state = "error";
      console.warn("Q-WiSE: WAV upload failed:", err);
      alert(`WAV upload failed: ${err.message ?? err}`);
    }
  });
}


function pickFile() {
  return new Promise((resolve, reject) => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "audio/wav, audio/mpeg, audio/flac, audio/ogg, audio/x-m4a";
    input.addEventListener("change", () => {
      if (input.files && input.files[0]) resolve(input.files[0]);
      else reject(new Error("no file selected"));
    }, { once: true });
    input.addEventListener("cancel", () => reject(new Error("cancelled")));
    input.click();
  });
}


async function uploadFile(file) {
  const fd = new FormData();
  fd.append("file", file);
  const r = await fetch("/api/speech-wav", { method: "POST", body: fd });
  if (!r.ok) {
    const txt = await r.text().catch(() => "");
    throw new Error(`HTTP ${r.status}${txt ? ` — ${txt}` : ""}`);
  }
  return r.json();
}


function ensureLabel(anchor) {
  let lbl = document.querySelector("#wav-name");
  if (lbl) return lbl;
  lbl = document.createElement("span");
  lbl.id = "wav-name";
  lbl.className = "wav-name";
  lbl.textContent = "";
  anchor.insertAdjacentElement("afterend", lbl);
  return lbl;
}
