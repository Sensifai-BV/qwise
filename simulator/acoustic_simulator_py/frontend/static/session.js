/* =========================================================================
   Q-WiSE client-side session tracking — Task 19

   Two responsibilities:

     1. Anonymous session id — UUID v4 stored in the `qwise_sid` cookie
        AND mirrored in `localStorage`. Persists across reloads + tabs
        on the same origin. The id is purely client-side affinity; the
        backend rate limiter still keys on the caller's IP.

     2. Recordings catalogue — `{name, ts, files}` entries written on
        every successful `recording_stop`. Pruned to a 3-hour TTL that
        mirrors the server's auto-purge so a refresh after that window
        doesn't keep claiming sessions that disappeared on the server.

   Public surface:
     setupSessionTracking()  — call once at boot.
     getSessionId()          — UUID v4 string.
     markRecording(name, files)
     getRecordings()         — pruned snapshot.
     clearRecordings()       — wipe the cache.
     getRecordingsCount()    — convenience for hint UI.

   Storage shape (localStorage key `qwise.session.v1`):
     { session_id: "<uuid>", recordings: [{name, ts, files: [string]}] }

   The cookie + the localStorage key are both safe to clear by hand —
   the page rebuilds them on next load.
   ========================================================================= */

const COOKIE_NAME = "qwise_sid";
const STORAGE_KEY = "qwise.session.v1";
const TTL_MS      = 3 * 3600 * 1000;   // 3 hours; matches the server.
const CACHE_CAP   = 50;                 // hard ceiling on stored entries.


/* =====================================================================
   Boot
   ===================================================================== */
export function setupSessionTracking() {
  getSessionId();           // make sure the cookie + storage are seeded
  pruneRecordings();        // drop anything past the TTL on startup
  return readStorage();
}


/* =====================================================================
   Session id
   ===================================================================== */
export function getSessionId() {
  let sid = readCookie(COOKIE_NAME) || readStorage().session_id || null;
  if (!sid) {
    sid = generateUuid();
  }
  // Refresh the cookie expiry on every boot so an active user keeps it.
  setCookie(COOKIE_NAME, sid, 365);
  writeStorage({ ...readStorage(), session_id: sid });
  return sid;
}


/* =====================================================================
   Recordings catalogue
   ===================================================================== */
export function markRecording(name, files) {
  if (!name) return;
  const data = readStorage();
  const list = Array.isArray(data.recordings) ? data.recordings : [];
  // Remove a previous entry with the same name (server can re-list
  // a session multiple times across reloads) before prepending.
  const deduped = list.filter((e) => e.name !== name);
  deduped.unshift({
    name,
    ts: Date.now(),
    files: Array.isArray(files) ? files.slice(0, 32) : [],
  });
  data.recordings = deduped.slice(0, CACHE_CAP);
  writeStorage(data);
  return pruneRecordings();
}


export function getRecordings() {
  return pruneRecordings();
}


export function getRecordingsCount() {
  return getRecordings().length;
}


export function clearRecordings() {
  const data = readStorage();
  data.recordings = [];
  writeStorage(data);
}


export function pruneRecordings() {
  const data = readStorage();
  const list = Array.isArray(data.recordings) ? data.recordings : [];
  const cutoff = Date.now() - TTL_MS;
  const fresh = list.filter((e) => Number(e.ts || 0) >= cutoff);
  if (fresh.length !== list.length) {
    data.recordings = fresh;
    writeStorage(data);
  }
  return fresh;
}


/* =====================================================================
   Storage + cookie helpers
   ===================================================================== */
function readStorage() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) || {}) : {};
  } catch {
    return {};
  }
}


function writeStorage(data) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(data || {}));
  } catch {
    /* localStorage may be disabled (Safari private mode) — silently
       degrade; the rest of the app still works without persistence. */
  }
}


function readCookie(name) {
  if (typeof document === "undefined") return null;
  const re = new RegExp("(?:^|;\\s*)" + name + "=([^;]+)");
  const m = document.cookie.match(re);
  return m ? decodeURIComponent(m[1]) : null;
}


function setCookie(name, value, days) {
  if (typeof document === "undefined") return;
  const exp = new Date(Date.now() + days * 86_400_000).toUTCString();
  document.cookie =
    `${name}=${encodeURIComponent(value)}; expires=${exp}; path=/; SameSite=Lax`;
}


/* =====================================================================
   UUID v4 — prefer crypto.randomUUID, fall back to getRandomValues.
   ===================================================================== */
function generateUuid() {
  const c = (typeof window !== "undefined" && window.crypto)
        || (typeof globalThis !== "undefined" && globalThis.crypto);
  if (c && typeof c.randomUUID === "function") {
    return c.randomUUID();
  }
  const bytes = new Uint8Array(16);
  if (c && c.getRandomValues) {
    c.getRandomValues(bytes);
  } else {
    for (let i = 0; i < 16; i++) bytes[i] = Math.floor(Math.random() * 256);
  }
  bytes[6] = (bytes[6] & 0x0f) | 0x40;     // version 4
  bytes[8] = (bytes[8] & 0x3f) | 0x80;     // RFC variant
  const hex = Array.from(bytes, (b) => b.toString(16).padStart(2, "0"));
  return (
    hex.slice(0, 4).join("") + "-" +
    hex.slice(4, 6).join("") + "-" +
    hex.slice(6, 8).join("") + "-" +
    hex.slice(8, 10).join("") + "-" +
    hex.slice(10, 16).join("")
  );
}
