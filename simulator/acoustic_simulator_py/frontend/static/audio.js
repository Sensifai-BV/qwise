/* =========================================================================
   Q-WiSE audio + WebSocket client — Task 16

   * Opens `/ws/stream`, buffers the server's `hello` payload (Task 12).
   * Captures mic audio via getUserMedia + AudioContext at cfg.fs.
     A ScriptProcessorNode harvests frames at `cfg.frame_size` samples
     and pipes them straight to the WebSocket as float32 ArrayBuffers.
   * Re-broadcasts every server message as a DOM CustomEvent so the
     plot module (Task 17) and the playback / status UI (Task 18 / 19)
     can subscribe without touching the WS object directly.

   Public surface:
     openWebSocket(), closeWebSocket(), isOpen()
     startMic(), stopMic()
     sendControl(payload)        // any JSON control object

   Events emitted on `document`:
     "qwise:ws-open"             { detail: hello payload }
     "qwise:ws-close"            { detail: {} }
     "qwise:ws-error"            { detail: { error } }
     "qwise:ws-message"          { detail: <parsed JSON> }   — every text msg
     "qwise:ws-audio"            { detail: { buffer } }      — every binary msg
   ========================================================================= */

let ws = null;
let helloPayload = null;
let audioCtx = null;
let micStream = null;
let processor = null;
let micSource = null;
let pendingTail = new Float32Array(0);    // resample / rebatch carry-over


export function isOpen() {
  return ws !== null && ws.readyState === WebSocket.OPEN;
}

export function helloCfg() {
  return helloPayload;
}


/* ---------- WebSocket ---------------------------------------------- */
export function openWebSocket() {
  if (ws) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const url = `${proto}://${location.host}/ws/stream`;
    ws = new WebSocket(url);
    ws.binaryType = "arraybuffer";

    ws.addEventListener("open", () => {
      // The `open` event resolves the promise; the actual session
      // start happens once we've received `hello` from the server.
      resolve();
    });
    ws.addEventListener("error", (ev) => {
      document.dispatchEvent(
        new CustomEvent("qwise:ws-error", { detail: { error: String(ev) } })
      );
      reject(ev);
    });
    ws.addEventListener("close", () => {
      ws = null;
      helloPayload = null;
      document.dispatchEvent(new CustomEvent("qwise:ws-close"));
    });
    ws.addEventListener("message", onMessage);
  });
}


export function closeWebSocket() {
  if (ws) {
    ws.close();
    ws = null;
  }
}


function onMessage(ev) {
  if (typeof ev.data === "string") {
    let parsed;
    try { parsed = JSON.parse(ev.data); }
    catch (e) { console.warn("Q-WiSE: bad JSON from WS:", e); return; }

    if (parsed?.type === "hello") {
      helloPayload = parsed;
      document.dispatchEvent(
        new CustomEvent("qwise:ws-open", { detail: parsed })
      );
    }
    document.dispatchEvent(
      new CustomEvent("qwise:ws-message", { detail: parsed })
    );
  } else {
    document.dispatchEvent(
      new CustomEvent("qwise:ws-audio", { detail: { buffer: ev.data } })
    );
  }
}


export function sendControl(payload) {
  if (!isOpen()) return false;
  try {
    ws.send(JSON.stringify(payload));
    return true;
  } catch (err) {
    console.warn("Q-WiSE: sendControl failed:", err);
    return false;
  }
}


/* ---------- Microphone capture ------------------------------------- */
export async function startMic() {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("Browser does not expose getUserMedia");
  }
  const fs = helloPayload?.fs ?? 16000;
  const frameSize = helloPayload?.frame_size ?? 1024;

  // Try to open an AudioContext at the target SR; if the browser
  // ignores the request (Safari historically did) we live with the
  // device rate and resample with a cheap linear interpolation.
  audioCtx = new (window.AudioContext || window.webkitAudioContext)({
    sampleRate: fs,
  });
  micStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
    },
  });
  micSource = audioCtx.createMediaStreamSource(micStream);

  // Pick a buffer size whose post-resample length is roughly frameSize;
  // ScriptProcessor accepts {256, 512, ..., 16384} so we round to the
  // nearest power-of-two.
  const ratio = audioCtx.sampleRate / fs;
  const procSize = nearestPow2(Math.max(256, Math.round(frameSize * ratio)));

  processor = audioCtx.createScriptProcessor(procSize, 1, 1);
  processor.onaudioprocess = (ev) => {
    const inBuf = ev.inputBuffer.getChannelData(0);
    const out = ratio === 1
      ? new Float32Array(inBuf)          // copy — buffer reused by AC
      : resampleLinear(inBuf, 1 / ratio);
    streamFrames(out, frameSize);
  };

  micSource.connect(processor);
  processor.connect(audioCtx.destination);    // ScriptProcessor needs this
}


export function stopMic() {
  pendingTail = new Float32Array(0);
  if (processor) {
    try { processor.disconnect(); } catch {}
    processor.onaudioprocess = null;
    processor = null;
  }
  if (micSource) {
    try { micSource.disconnect(); } catch {}
    micSource = null;
  }
  if (micStream) {
    for (const t of micStream.getTracks()) t.stop();
    micStream = null;
  }
  if (audioCtx) {
    audioCtx.close().catch(() => {});
    audioCtx = null;
  }
}


/* ---------- Frame batcher (input rate → frameSize-sized WS frames) - */
function streamFrames(newSamples, frameSize) {
  const total = pendingTail.length + newSamples.length;
  const combined = new Float32Array(total);
  combined.set(pendingTail, 0);
  combined.set(newSamples, pendingTail.length);

  let pos = 0;
  while (combined.length - pos >= frameSize) {
    const frame = combined.slice(pos, pos + frameSize);
    if (isOpen()) ws.send(frame.buffer);
    pos += frameSize;
  }
  pendingTail = combined.slice(pos);
}


/* ---------- Cheap utilities --------------------------------------- */
function nearestPow2(n) {
  const allowed = [256, 512, 1024, 2048, 4096, 8192, 16384];
  let best = allowed[0];
  let bestDiff = Infinity;
  for (const v of allowed) {
    const d = Math.abs(v - n);
    if (d < bestDiff) { best = v; bestDiff = d; }
  }
  return best;
}


function resampleLinear(input, factor) {
  // Resample `input` so its output length is `Math.floor(input.length * factor)`
  // — factor < 1 downsamples (input fs > target fs), factor > 1 upsamples.
  const outLen = Math.floor(input.length * factor);
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const srcIdx = i / factor;
    const i0 = Math.floor(srcIdx);
    const frac = srcIdx - i0;
    const a = input[i0] || 0;
    const b = input[i0 + 1] !== undefined ? input[i0 + 1] : a;
    out[i] = a * (1 - frac) + b * frac;
  }
  return out;
}
