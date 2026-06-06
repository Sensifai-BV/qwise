/* =========================================================================
   Q-WiSE sidebar accordion — Task 15

   Hydrates the right-hand sidebar from /api/config/schema. Each entry
   becomes a slider (numeric) or select (enum) inside a collapsible
   section card. A change on any widget fires a CustomEvent so Task 16
   can forward it to the WebSocket as a `config_patch` control message.

   The backend's allow-list explicitly rejects ``n_mics`` patches — we
   still render the slider so the user sees the current mic count, but
   we tag the row with a hint that a page reload is required.
   ========================================================================= */

const ACCORDION_KEY = "qwise.sidebar.sections";
const N_MICS_HINT =
  "Reload the page to apply a different mic count.";

const CANONICAL_SECTIONS = [
  "Microphones",
  "Scene",
  "Drone source",
  "Environment",
  "Acoustics",
  "Gains",
  "MWF",
];


/* ---------- public entry point -------------------------------------- */
export async function setupSidebar() {
  const host = document.querySelector("#sidebar-body");
  if (!host) return;

  let schema;
  try {
    const r = await fetch("/api/config/schema");
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    schema = (await r.json()).sidebar;
  } catch (err) {
    host.innerHTML =
      '<p class="sidebar-loading">Config schema unreachable — ' +
      "is the backend up?</p>";
    console.warn("Q-WiSE: /api/config/schema failed:", err);
    return;
  }

  const sections = groupBySection(schema);
  renderAccordion(host, sections);
}


/* ---------- group + sort by canonical section order ---------------- */
function groupBySection(entries) {
  const buckets = new Map();
  for (const e of entries) {
    if (!buckets.has(e.section)) buckets.set(e.section, []);
    buckets.get(e.section).push(e);
  }
  const rank = new Map(CANONICAL_SECTIONS.map((n, i) => [n, i]));
  return new Map(
    [...buckets.entries()].sort((a, b) => {
      const ra = rank.get(a[0]) ?? 999;
      const rb = rank.get(b[0]) ?? 999;
      return ra - rb;
    })
  );
}


/* ---------- render -------------------------------------------------- */
function renderAccordion(host, sections) {
  host.innerHTML = "";
  const wrap = document.createElement("div");
  wrap.className = "accordion";

  const openState = readOpenState();
  let firstSection = true;
  for (const [name, fields] of sections) {
    const isOpen = openState[name] ?? firstSection;   // default-open the first
    wrap.appendChild(buildSection(name, fields, isOpen));
    firstSection = false;
  }
  host.appendChild(wrap);
}


function buildSection(name, fields, isOpen) {
  const sec = document.createElement("section");
  sec.className = "accordion-section";
  sec.dataset.section = name;
  sec.dataset.open = String(isOpen);

  const head = document.createElement("div");
  head.className = "accordion-head";
  head.tabIndex = 0;
  head.setAttribute("role", "button");
  head.setAttribute("aria-expanded", String(isOpen));

  const bar = document.createElement("span");
  bar.className = "acc-bar";
  const title = document.createElement("span");
  title.className = "acc-title";
  title.textContent = name;
  const toggle = document.createElement("span");
  toggle.className = "acc-toggle";
  toggle.setAttribute("aria-hidden", "true");
  toggle.textContent = isOpen ? "−" : "+";

  head.appendChild(bar);
  head.appendChild(title);
  head.appendChild(toggle);

  const body = document.createElement("div");
  body.className = "accordion-body";
  for (const field of fields) {
    body.appendChild(buildFieldRow(field));
  }

  head.addEventListener("click", () => toggleSection(sec));
  head.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" || ev.key === " ") {
      ev.preventDefault();
      toggleSection(sec);
    }
  });

  sec.appendChild(head);
  sec.appendChild(body);
  return sec;
}


function toggleSection(sec) {
  const next = sec.dataset.open !== "true";
  sec.dataset.open = String(next);
  sec.querySelector(".accordion-head").setAttribute(
    "aria-expanded", String(next)
  );
  sec.querySelector(".acc-toggle").textContent = next ? "−" : "+";
  writeOpenState(sec.dataset.section, next);
}


/* ---------- field rows --------------------------------------------- */
function buildFieldRow(field) {
  const row = document.createElement("div");
  row.className = "field-row";
  row.dataset.key = field.key;

  const top = document.createElement("div");
  top.className = "field-top";
  const label = document.createElement("span");
  label.className = "field-label";
  label.textContent = field.label;
  const valueLabel = document.createElement("span");
  valueLabel.className = "field-value";
  valueLabel.textContent = formatValue(field, field.value);
  top.appendChild(label);
  top.appendChild(valueLabel);
  row.appendChild(top);

  const input = field.widget === "select"
    ? buildSelect(field)
    : buildSlider(field);
  row.appendChild(input);

  // n_mics requires a full reconstruction of the simulator session
  // (per-source history rings, MWF covariance buffers, AudioIO mic
  // tracks). The backend rejects live patches; surface that here.
  if (field.key === "n_mics") {
    const hint = document.createElement("span");
    hint.className = "field-hint";
    hint.textContent = N_MICS_HINT;
    row.appendChild(hint);
    row.dataset.requiresReload = "true";
  }

  const dispatchChange = (rawValue) => {
    const value = parseInputValue(field, rawValue);
    valueLabel.textContent = formatValue(field, value);
    document.dispatchEvent(new CustomEvent("qwise:config-change", {
      detail: { key: field.key, value, widget: field.widget },
    }));
  };

  // `input` fires while dragging, `change` fires on commit; both
  // forward the same payload so Task 16 can throttle whichever side it
  // prefers when wiring the WebSocket.
  input.addEventListener("input", () => dispatchChange(input.value));
  input.addEventListener("change", () => dispatchChange(input.value));

  return row;
}


function buildSlider(field) {
  const input = document.createElement("input");
  input.type = "range";
  input.min = field.min ?? 0;
  input.max = field.max ?? 100;
  input.step = field.step ?? 1;
  input.value = String(field.value);
  input.setAttribute("aria-label", field.label);
  return input;
}


function buildSelect(field) {
  const input = document.createElement("select");
  input.setAttribute("aria-label", field.label);
  for (const opt of field.options || []) {
    const o = document.createElement("option");
    o.value = String(opt);
    o.textContent = String(opt);
    if (String(opt) === String(field.value)) o.selected = true;
    input.appendChild(o);
  }
  return input;
}


/* ---------- value formatting --------------------------------------- */
function formatValue(field, value) {
  if (field.widget === "select") return String(value);
  if (typeof value === "number") {
    if (Number.isInteger(value)) return String(value);
    const step = Number(field.step ?? 0);
    const decimals = step > 0 && step < 1 ? Math.max(
      2, -Math.floor(Math.log10(step))
    ) : 2;
    return value.toFixed(decimals);
  }
  return String(value);
}


function parseInputValue(field, raw) {
  if (field.widget === "select") return raw;
  const n = Number(raw);
  return Number.isFinite(n) ? n : raw;
}


/* ---------- localStorage persistence of open/closed state ---------- */
function readOpenState() {
  try {
    const raw = localStorage.getItem(ACCORDION_KEY);
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
}


function writeOpenState(section, open) {
  try {
    const cur = readOpenState();
    cur[section] = open;
    localStorage.setItem(ACCORDION_KEY, JSON.stringify(cur));
  } catch {
    /* swallow quota / privacy errors silently */
  }
}
