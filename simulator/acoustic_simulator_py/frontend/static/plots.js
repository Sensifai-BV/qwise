/* =========================================================================
   Q-WiSE Plotly plot manager — Task 17

   Owns six plots:
     #card-scene      → 3-D scene (drone, mics, human, env)
     #card-spec-noisy → noisy composite spectrogram (rolling)
     #card-spec-mwf   → MWF-enhanced spectrogram (rolling)
     #card-mic-array  → N per-mic waveform rows (overwrite per frame)
     #card-mwf-wave   → MWF output waveform (overwrite per frame)
     #card-vad-trace  → VAD score line + speech-shading

   Inputs come from audio.js:
     * qwise:ws-open    { config, geometry, … }
     * qwise:ws-message { type:'frame', vad_score, is_speech, … }
     * qwise:ws-audio   { buffer: ArrayBuffer of float32 — (2+n_mics) rows × N }
     * qwise:ws-message { type:'ack', kind:'config_patch'|'reset', geometry }

   Plotly is loaded lazily via the CDN <script> tag in index.html; if it
   never lands (offline / blocked CDN) every plot falls back to its
   placeholder so the rest of the UI keeps working.
   ========================================================================= */

import { fftMagnitudeDb, FFT_MIN_DB } from "/static/fft.js";

const MIC_COLORS = ["#22c55e", "#3b82f6", "#f97316", "#e879f9",
                    "#a3e635", "#fbbf24", "#06b6d4", "#fb7185"];
const ACCENT     = "#469e00";
const TEXT_LO    = "#5f6b56";
const TEXT_MID   = "#a3b094";
const SURFACE_BG = "rgba(0,0,0,0)";

const SPEC_COLS = 90;        // matches MATLAB GUI spec_ncols
const VAD_LEN   = 200;       // rolling VAD trace length

let cfg = null;
let nMics = 0;
let frameSize = 1024;
let ready = false;

let micBufs = [];            // Float32Array views per mic row
let mwfBuf  = null;
let compBuf = null;

let specNoisy = null;        // 2D number[][]   shape: [n_bins][SPEC_COLS]
let specMwf   = null;
let specCol   = 0;           // ring write pointer

let vadScores = new Array(VAD_LEN).fill(0);
let vadFlags  = new Array(VAD_LEN).fill(false);
let vadPtr    = 0;

let lastFrameIdx = -1;
let redrawPending = false;
let specRedrawEvery = 2;     // throttle the heavy heatmaps a bit


/* ---------- bootstrap --------------------------------------------- */
export function initPlots(hello) {
  cfg = hello.config;
  nMics = hello.n_mics;
  frameSize = hello.frame_size;

  if (typeof window.Plotly === "undefined") {
    console.warn("Q-WiSE: Plotly.js not loaded — plots disabled.");
    return false;
  }

  const nFreq = (frameSize >> 1) + 1;
  specNoisy = makeFreshSpec(nFreq);
  specMwf   = makeFreshSpec(nFreq);
  specCol   = 0;

  vadScores = new Array(VAD_LEN).fill(0);
  vadFlags  = new Array(VAD_LEN).fill(false);
  vadPtr    = 0;

  buildScene(hello.geometry);
  buildSpectrogram("#card-spec-noisy", specNoisy);
  buildSpectrogram("#card-spec-mwf",   specMwf);
  buildMicWaveforms();
  buildMwfWaveform();
  buildVad();
  ready = true;
  return true;
}


/* ---------- empty bootstrap (no WebSocket yet) --------------------- */
/**
 * Render real (empty) Plotly cards from the default config + geometry.
 * Called once at page load so the user sees actual plot frames instead
 * of "Task 17 — …" placeholder strings before pressing Start Capture.
 *
 * The live `initPlots(hello)` path overwrites these on the first frame.
 *
 * @param {object} defaultCfg   Output of GET /api/config/default
 * @param {object} defaultGeom  Output of GET /api/geometry
 */
export function initEmptyPlots(defaultCfg, defaultGeom) {
  if (typeof window.Plotly === "undefined") return false;
  if (!defaultCfg || !defaultGeom) return false;

  cfg = defaultCfg;
  nMics = defaultCfg.n_mics;
  frameSize = defaultCfg.frame_size;

  const nFreq = (frameSize >> 1) + 1;
  specNoisy = makeFreshSpec(nFreq);
  specMwf   = makeFreshSpec(nFreq);
  specCol   = 0;

  vadScores = new Array(VAD_LEN).fill(0);
  vadFlags  = new Array(VAD_LEN).fill(false);
  vadPtr    = 0;

  try {
    buildScene(defaultGeom);
    buildSpectrogram("#card-spec-noisy", specNoisy);
    buildSpectrogram("#card-spec-mwf",   specMwf);
    buildMicWaveforms();
    buildMwfWaveform();
    buildVad();
  } catch (err) {
    console.warn("Q-WiSE: initEmptyPlots failed:", err);
    return false;
  }
  ready = true;
  return true;
}


/* ---------- per-frame ingest -------------------------------------- */
export function pushAudio(buffer) {
  if (!ready || !buffer) return;
  const view = new Float32Array(buffer);
  const expected = (2 + nMics) * frameSize;
  if (view.length < expected) return;

  mwfBuf  = view.subarray(0, frameSize);
  compBuf = view.subarray(frameSize, 2 * frameSize);
  micBufs = [];
  for (let m = 0; m < nMics; m++) {
    const off = (2 + m) * frameSize;
    micBufs.push(view.subarray(off, off + frameSize));
  }

  // Append one spectrogram column per buffer (noisy = composite, mwf row).
  if (compBuf) pushSpecColumn(specNoisy, compBuf);
  if (mwfBuf)  pushSpecColumn(specMwf,   mwfBuf);
  specCol = (specCol + 1) % SPEC_COLS;

  scheduleRedraw();
}


export function pushFrameMeta(meta) {
  if (!ready || !meta || meta.type !== "frame") return;
  vadScores[vadPtr] = Number(meta.vad_score) || 0;
  vadFlags[vadPtr]  = !!meta.is_speech;
  vadPtr = (vadPtr + 1) % VAD_LEN;
  lastFrameIdx = meta.frame_idx | 0;
}


export function applyGeometry(geometry) {
  if (!ready || !geometry) return;
  redrawScene(geometry);
}


/* =====================================================================
   3-D Acoustic Scene — port of MATLAB visualization/draw_scene.m + draw_drone.m
   =====================================================================

   Renders the same scene the MATLAB GUI does:

     * Asphalt ground plane (semi-transparent grey grid)
     * Human stick figure with head circle + mouth marker
     * Ground image source + dotted line to the mouth (Z-mirror trick)
     * Quadrotor drone body (cuboid mesh) with 4 arms + 4 translucent
       rotor discs and "+" cross-bars on each rotor
     * Per-mic coloured dots with M1..MN labels (matches draw_drone.m
       colour wheel)
     * Env-noise marker + dotted line to the drone
     * Direct path (dashed) + reflected path (solid green) with the
       same labels and elbow point as the MATLAB version
     * BPF / AGL annotation above the drone
   ===================================================================== */

/* Mic colour wheel mirrors draw_drone.m */
const MIC_SCENE_COLORS = [
  "#00c752",   // 0.00 0.78 0.32
  "#008cf2",   // 0.00 0.55 0.95
  "#ff8500",   // 1.00 0.52 0.00
  "#f259bf",   // 0.95 0.35 0.75
  "#9bd933",   // 0.60 0.85 0.20
];
const HUMAN_BLUE  = "#4d73e6";   // 0.30 0.45 0.90
const MOUTH_BLUE  = "#4d8cff";   // 0.30 0.55 1.00
const REFL_GREEN  = "#1ab873";   // 0.10 0.72 0.45
const REFL_DARK   = "#1a9e60";   // 0.10 0.62 0.38
const ENV_PURPLE  = "#a61ad9";   // 0.65 0.10 0.85
const ENV_PURPLE2 = "#9919cc";   // 0.60 0.10 0.80
const DRONE_BODY  = "#212121";   // 0.13 0.13 0.13
const DRONE_ARM   = "#383838";   // 0.22 0.22 0.22
const ROTOR_FACE  = "#d12424";   // 0.82 0.14 0.14
const ROTOR_EDGE  = "#8c1414";   // 0.55 0.08 0.08
const IMG_GREY    = "#808080";   // 0.50 0.50 0.50
const GROUND_FACE = "#454545";   // 0.27 0.27 0.27
const GROUND_EDGE = "#6b6b6b";   // 0.42 0.42 0.42
const BPF_RED     = "#e62e2e";   // 0.90 0.18 0.18


function buildScene(geom) {
  if (!geom) return;
  window.Plotly.newPlot(scene3DTarget(), sceneTraces(geom), sceneLayout(geom), {
    displaylogo: false, responsive: true,
  });
}


function redrawScene(geom) {
  window.Plotly.react(scene3DTarget(), sceneTraces(geom), sceneLayout(geom), {
    displaylogo: false, responsive: true,
  });
}


/* =====================================================================
   sceneTraces — build every trace, mirroring draw_scene.m + draw_drone.m
   ===================================================================== */
function sceneTraces(geom) {
  const traces = [];
  const human = geom.pos_human;
  const drone = geom.pos_drone;
  const env   = geom.pos_env;
  const mics  = geom.pos_mics || [];
  const img   = geom.pos_img_src
              || [human[0], human[1], -human[2]];     // safe fallback
  const Hh    = (cfg && cfg.human_height) ? cfg.human_height : human[2] / 0.88;
  const droneRpm    = (cfg && cfg.drone_rpm)    ? cfg.drone_rpm    : 8000;
  const droneBlades = (cfg && cfg.drone_blades) ? cfg.drone_blades : 3;
  const groundR     = (cfg && cfg.ground_R)     ? cfg.ground_R     : 0.9;
  const refMic      = (geom.ref_mic || 1) - 1;

  /* Axis bounds — same heuristic as MATLAB (union of every visible point,
     plus a 0.6 m margin, with x/y forced to share the same half-span). */
  const bounds = computeSceneBounds([human, drone, env, img, ...mics], 0.6);

  /* ---------------- Ground plane (asphalt) ---------------- */
  traces.push(makeGroundSurface(bounds));
  traces.push({
    type: "scatter3d", mode: "text",
    x: [bounds.xr[0] + 0.3], y: [bounds.yr[0] + 0.3], z: [0.02],
    text: [`Asphalt (R=${groundR.toFixed(2)})`],
    textposition: "top right",
    textfont: { color: "#7a7a7a", size: 9 },
    hoverinfo: "skip", showlegend: false,
  });

  /* ---------------- Human stick figure ---------------- */
  const xh = human[0], yh = human[1];
  traces.push({
    type: "scatter3d", mode: "lines",
    x: [xh, xh, xh],
    y: [yh, yh, yh],
    z: [0, Hh * 0.52, Hh * 0.87],
    line: { color: HUMAN_BLUE, width: 6 },
    hoverinfo: "skip", showlegend: false, name: "body",
  });
  /* head — circle in the X-Z plane (matches MATLAB) */
  const headTh = linspace(0, 2 * Math.PI, 36);
  const rh = 0.08;
  traces.push({
    type: "scatter3d", mode: "lines",
    x: headTh.map(t => xh + rh * Math.cos(t)),
    y: headTh.map(() => yh),
    z: headTh.map(t => Hh * 0.935 + rh * Math.sin(t)),
    line: { color: HUMAN_BLUE, width: 4 },
    hoverinfo: "skip", showlegend: false, name: "head",
  });
  /* mouth marker + label */
  traces.push({
    type: "scatter3d", mode: "markers+text",
    x: [xh], y: [yh], z: [human[2]],
    marker: { size: 7, color: MOUTH_BLUE, symbol: "diamond" },
    text: [`Mouth ${human[2].toFixed(2)} m`],
    textposition: "top center",
    textfont: { color: MOUTH_BLUE, size: 9 },
    hoverinfo: "skip", showlegend: false, name: "mouth",
  });
  /* height label above the head */
  traces.push({
    type: "scatter3d", mode: "text",
    x: [xh - 0.08], y: [yh], z: [Hh + 0.14],
    text: [`${(Hh * 100).toFixed(0)} cm`],
    textfont: { color: "#6680cc", size: 9 },
    hoverinfo: "skip", showlegend: false,
  });

  /* ---------------- Image source (ground image) ---------------- */
  traces.push({
    type: "scatter3d", mode: "markers",
    x: [img[0]], y: [img[1]], z: [img[2]],
    marker: { size: 5, color: IMG_GREY, symbol: "diamond-open" },
    hoverinfo: "skip", showlegend: false, name: "img-src",
  });
  traces.push({
    type: "scatter3d", mode: "lines",
    x: [xh, xh], y: [yh, yh], z: [human[2], img[2]],
    line: { color: IMG_GREY, width: 2, dash: "dot" },
    hoverinfo: "skip", showlegend: false, name: "img-link",
  });
  traces.push({
    type: "scatter3d", mode: "text",
    x: [img[0] + 0.08], y: [img[1]], z: [img[2] - 0.13],
    text: [`img z=${img[2].toFixed(2)}`],
    textfont: { color: IMG_GREY, size: 9 },
    hoverinfo: "skip", showlegend: false,
  });

  /* ---------------- Drone body + rotors + mics ---------------- */
  appendDroneTraces(traces, drone, mics, refMic);

  /* ---------------- BPF / AGL annotation ---------------- */
  const bpf = Math.round((droneRpm * droneBlades) / 60);
  const droneZTop = (mics.length ? mics[0][2] : drone[2]) + 0.075 + 0.045 + 0.008;
  traces.push({
    type: "scatter3d", mode: "text",
    x: [drone[0] + 0.11], y: [drone[1]], z: [droneZTop + 0.10],
    text: [`BPF ${bpf} Hz · AGL ${drone[2].toFixed(2)} m`],
    textfont: { color: BPF_RED, size: 9, family: "JetBrains Mono, monospace" },
    hoverinfo: "skip", showlegend: false,
  });

  /* ---------------- Env noise source ---------------- */
  traces.push({
    type: "scatter3d", mode: "markers+text",
    x: [env[0]], y: [env[1]], z: [env[2]],
    marker: {
      size: 9, color: ENV_PURPLE, symbol: "diamond",
      line: { color: ENV_PURPLE2, width: 2 },
    },
    text: ["Env Noise"],
    textposition: "top center",
    textfont: { color: ENV_PURPLE, size: 9 },
    hoverinfo: "skip", showlegend: false, name: "env",
  });

  /* ---------------- Direct path mouth → drone + distance ---------------- */
  const dDirect = euclid(human, drone);
  const midDir = midpoint(human, drone);
  traces.push({
    type: "scatter3d", mode: "lines",
    x: [xh, drone[0]], y: [yh, drone[1]], z: [human[2], drone[2]],
    line: { color: MOUTH_BLUE, width: 3, dash: "dash" },
    hoverinfo: "skip", showlegend: false, name: "direct",
  });
  traces.push({
    type: "scatter3d", mode: "text",
    x: [midDir[0] + 0.07], y: [midDir[1]], z: [midDir[2] + 0.08],
    text: [`${dDirect.toFixed(2)} m`],
    textfont: { color: MOUTH_BLUE, size: 9 },
    hoverinfo: "skip", showlegend: false,
  });

  /* ---------------- Reflected path via centre mic ---------------- */
  if (mics.length > 0) {
    const mc2 = mics[Math.ceil(mics.length / 2) - 1];
    const tsp = -img[2] / (mc2[2] - img[2]);
    const sp = [
      img[0] + tsp * (mc2[0] - img[0]),
      img[1] + tsp * (mc2[1] - img[1]),
      img[2] + tsp * (mc2[2] - img[2]),
    ];
    traces.push({
      type: "scatter3d", mode: "lines",
      x: [xh, sp[0], mc2[0]],
      y: [yh, sp[1], mc2[1]],
      z: [human[2], sp[2], mc2[2]],
      line: { color: REFL_GREEN, width: 3 },
      hoverinfo: "skip", showlegend: false, name: "reflected",
    });
    traces.push({
      type: "scatter3d", mode: "markers",
      x: [sp[0]], y: [sp[1]], z: [sp[2]],
      marker: { size: 5, color: REFL_DARK, symbol: "square" },
      hoverinfo: "skip", showlegend: false, name: "bounce-pt",
    });
    traces.push({
      type: "scatter3d", mode: "text",
      x: [sp[0] + 0.06], y: [sp[1]], z: [sp[2] + 0.07],
      text: [`R=${groundR.toFixed(2)}`],
      textfont: { color: REFL_GREEN, size: 9 },
      hoverinfo: "skip", showlegend: false,
    });
  }

  /* ---------------- Env → drone link ---------------- */
  traces.push({
    type: "scatter3d", mode: "lines",
    x: [env[0], drone[0]], y: [env[1], drone[1]], z: [env[2], drone[2]],
    line: { color: ENV_PURPLE2, width: 2, dash: "dot" },
    hoverinfo: "skip", showlegend: false, name: "env-link",
  });

  return traces;
}


/* =====================================================================
   appendDroneTraces — port of draw_drone.m
   ===================================================================== */
function appendDroneTraces(traces, pd, pm, refMic) {
  const arm_l = 0.19;
  const bw    = 0.09;
  const bh    = 0.045;
  const blift = 0.075;
  const rr    = 0.075;

  const mz  = (pm.length ? pm[0][2] : pd[2]);
  const bz0 = mz + blift;
  const bz1 = bz0 + bh;
  const rz  = bz1 + 0.008;
  const dx  = pd[0], dy = pd[1];

  /* Drone body — cuboid (8 verts, 12 triangles) */
  const vx = [dx - bw, dx + bw, dx + bw, dx - bw, dx - bw, dx + bw, dx + bw, dx - bw];
  const vy = [dy - bw, dy - bw, dy + bw, dy + bw, dy - bw, dy - bw, dy + bw, dy + bw];
  const vz = [bz0,     bz0,     bz0,     bz0,     bz1,     bz1,     bz1,     bz1];
  /* faces (each quad split into two triangles) */
  const cuboidI = [0, 0, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3];
  const cuboidJ = [1, 2, 5, 6, 1, 4, 2, 5, 3, 6, 0, 7];
  const cuboidK = [2, 3, 6, 7, 4, 5, 5, 6, 6, 7, 7, 4];
  traces.push({
    type: "mesh3d",
    x: vx, y: vy, z: vz,
    i: cuboidI, j: cuboidJ, k: cuboidK,
    color: DRONE_BODY, opacity: 0.88,
    flatshading: true,
    hoverinfo: "skip", showlegend: false, name: "drone-body",
  });

  /* 4 arms */
  const rcx = [dx + arm_l, dx - arm_l, dx,           dx          ];
  const rcy = [dy,         dy,         dy + arm_l,   dy - arm_l  ];
  for (let k = 0; k < 4; k++) {
    traces.push({
      type: "scatter3d", mode: "lines",
      x: [dx, rcx[k]], y: [dy, rcy[k]], z: [bz1, rz],
      line: { color: DRONE_ARM, width: 5 },
      hoverinfo: "skip", showlegend: false, name: `arm-${k + 1}`,
    });

    /* Rotor disc — fan triangulation around the centre */
    const N = 24;
    const th = linspace(0, 2 * Math.PI, N + 1);
    const cx = rcx[k], cy = rcy[k];
    const dxs = [cx];
    const dys = [cy];
    const dzs = [rz];
    for (let i = 0; i < N + 1; i++) {
      dxs.push(cx + rr * Math.cos(th[i]));
      dys.push(cy + rr * Math.sin(th[i]));
      dzs.push(rz);
    }
    const iI = [], iJ = [], iK = [];
    for (let i = 1; i <= N; i++) {
      iI.push(0); iJ.push(i); iK.push(i + 1);
    }
    traces.push({
      type: "mesh3d",
      x: dxs, y: dys, z: dzs, i: iI, j: iJ, k: iK,
      color: ROTOR_FACE, opacity: 0.52,
      flatshading: true,
      hoverinfo: "skip", showlegend: false, name: `rotor-${k + 1}`,
    });

    /* Rotor cross-bars */
    traces.push({
      type: "scatter3d", mode: "lines",
      x: [cx - rr, cx + rr], y: [cy, cy], z: [rz, rz],
      line: { color: ROTOR_EDGE, width: 2 },
      hoverinfo: "skip", showlegend: false, name: `rotor-cross-h-${k + 1}`,
    });
    traces.push({
      type: "scatter3d", mode: "lines",
      x: [cx, cx], y: [cy - rr, cy + rr], z: [rz, rz],
      line: { color: ROTOR_EDGE, width: 2 },
      hoverinfo: "skip", showlegend: false, name: `rotor-cross-v-${k + 1}`,
    });
  }

  /* Microphone markers — one trace per mic so each gets its own colour
     (matches draw_drone.m's coloured wheel). */
  for (let m = 0; m < pm.length; m++) {
    const color = MIC_SCENE_COLORS[m % MIC_SCENE_COLORS.length];
    const isRef = (m === refMic);
    traces.push({
      type: "scatter3d", mode: "markers+text",
      x: [pm[m][0]], y: [pm[m][1]], z: [pm[m][2]],
      marker: {
        size: isRef ? 9 : 7,
        color, symbol: "circle",
        line: isRef ? { color: "#ffffff", width: 1.6 } : { width: 0 },
      },
      text: [`M${m + 1}${isRef ? " ref" : ""}`],
      textposition: "top center",
      textfont: { color, size: 9 },
      hoverinfo: "skip", showlegend: false, name: `mic-${m + 1}`,
    });
  }
}


/* =====================================================================
   sceneLayout / helpers
   ===================================================================== */
function sceneLayout(geom) {
  const b = (geom && geom.pos_human) ? computeSceneBounds([
    geom.pos_human, geom.pos_drone, geom.pos_env,
    geom.pos_img_src || [geom.pos_human[0], geom.pos_human[1], -geom.pos_human[2]],
    ...(geom.pos_mics || []),
  ], 0.6) : null;
  const scene = {
    bgcolor: SURFACE_BG,
    xaxis: axis3D("x (m)", b ? b.xr : undefined),
    yaxis: axis3D("y (m)", b ? b.yr : undefined),
    zaxis: axis3D("z (m)", b ? b.zr : undefined),
    aspectmode: b ? "manual" : "data",
    aspectratio: { x: 1.1, y: 1.1, z: 0.9 },     // matches MATLAB PlotBoxAspectRatio
    camera: { eye: { x: 1.8, y: -1.5, z: 0.95 } },
  };
  return {
    paper_bgcolor: SURFACE_BG,
    plot_bgcolor:  SURFACE_BG,
    margin: { l: 0, r: 0, t: 4, b: 4 },
    showlegend: false,
    scene,
  };
}


function axis3D(title, range) {
  return {
    title: { text: title, font: { color: TEXT_MID, size: 10 } },
    color: TEXT_LO,
    gridcolor: "#1f2a16",
    zerolinecolor: "#1f2a16",
    showbackground: false,
    range: range || undefined,
  };
}


function scene3DTarget() {
  const card = document.querySelector("#card-scene .card-body");
  card.classList.remove("card-placeholder");
  card.innerHTML = "";
  const host = document.createElement("div");
  host.id = "plotly-scene";
  host.style.width = "100%";
  host.style.height = "100%";
  card.appendChild(host);
  return host;
}


/* ----- small geometry helpers (kept inline so the scene file is one
        self-contained unit, mirroring the MATLAB helper layout) -------- */
function computeSceneBounds(pts, margin) {
  const xs = pts.map(p => p[0]);
  const ys = pts.map(p => p[1]);
  const zs = pts.map(p => p[2]);
  let xr = [Math.min(...xs) - margin, Math.max(...xs) + margin];
  let yr = [Math.min(...ys) - margin, Math.max(...ys) + margin];
  let zr = [Math.min(...zs) - margin, Math.max(...zs) + margin];

  // x/y share the same half-span; enforce a 1.5 m minimum.
  const cx = (xr[0] + xr[1]) / 2;
  const cy = (yr[0] + yr[1]) / 2;
  let halfXY = Math.max(xr[1] - xr[0], yr[1] - yr[0]) / 2;
  halfXY = Math.max(halfXY, 1.5);
  xr = [cx - halfXY, cx + halfXY];
  yr = [cy - halfXY, cy + halfXY];

  // Ground plane must always be in view.
  zr[0] = Math.min(zr[0], -0.2);
  zr[1] = Math.max(zr[1], 0.5);

  return { xr, yr, zr };
}


function makeGroundSurface(bounds) {
  const N = 10;
  const x = linspace(bounds.xr[0], bounds.xr[1], N);
  const y = linspace(bounds.yr[0], bounds.yr[1], N);
  // Plotly surface expects z as 2D [j][i] for given x[i], y[j].
  const z = new Array(N);
  for (let j = 0; j < N; j++) {
    z[j] = new Array(N).fill(0);
  }
  return {
    type: "surface",
    x, y, z,
    showscale: false,
    colorscale: [[0, GROUND_FACE], [1, GROUND_FACE]],
    opacity: 0.25,
    contours: { x: { highlight: false }, y: { highlight: false } },
    lighting: { ambient: 0.95, diffuse: 0.1, specular: 0.0 },
    hoverinfo: "skip", showlegend: false, name: "ground",
  };
}


function linspace(a, b, n) {
  if (n <= 1) return [a];
  const out = new Array(n);
  const step = (b - a) / (n - 1);
  for (let i = 0; i < n; i++) out[i] = a + i * step;
  return out;
}


function euclid(a, b) {
  const dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}


function midpoint(a, b) {
  return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2];
}


/* ---------- spectrograms ----------------------------------------- */
function makeFreshSpec(nFreq) {
  const z = new Array(nFreq);
  for (let i = 0; i < nFreq; i++) {
    z[i] = new Float32Array(SPEC_COLS);   // dB units, start at 0; -inf is too noisy
    z[i].fill(FFT_MIN_DB);
  }
  return z;
}


function pushSpecColumn(spec, audio) {
  const mag = fftMagnitudeDb(audio, frameSize);
  const col = specCol;
  for (let k = 0; k < mag.length; k++) {
    spec[k][col] = mag[k];
  }
}


/* MATLAB 'hot' colormap — black → red → orange → yellow → white.
   Sampled in 7 stops so Plotly's linear-interp matches the MATLAB
   gradient closely. Pairs with the MATLAB ``clim([-80 -20])`` window. */
const SPEC_HOT_COLORSCALE = [
  [0.00, "#000000"],
  [0.16, "#3a0000"],
  [0.33, "#a00000"],
  [0.50, "#ff2a00"],
  [0.66, "#ff8a00"],
  [0.83, "#ffd400"],
  [1.00, "#ffffff"],
];
const SPEC_DB_MIN = -80;     // matches MATLAB clim min
const SPEC_DB_MAX = -20;     // matches MATLAB clim max


function buildSpectrogram(cardSelector, spec) {
  const host = mountPlot(cardSelector);
  if (!host) return;
  const fs = cfg.fs;
  const fHalf = (frameSize >> 1) + 1;
  const yFreq = new Array(fHalf);
  for (let k = 0; k < fHalf; k++) yFreq[k] = (k * fs) / (frameSize * 1000); // kHz
  window.Plotly.newPlot(host, [{
    type: "heatmap",
    z: shiftedSpec(spec),
    y: yFreq,
    colorscale: SPEC_HOT_COLORSCALE,
    zmin: SPEC_DB_MIN,
    zmax: SPEC_DB_MAX,
    showscale: false,
  }], heatmapLayout(), { displaylogo: false, responsive: true });
}


function redrawSpec(cardSelector, spec) {
  const host = document.querySelector(`${cardSelector} .plotly-host`);
  if (!host || !window.Plotly) return;
  window.Plotly.restyle(host, { z: [shiftedSpec(spec)] });
}


function shiftedSpec(spec) {
  // Return a new 2-D array with columns ordered oldest→newest so the
  // spectrogram scrolls right-to-left like the MATLAB GUI.
  const nFreq = spec.length;
  const out = new Array(nFreq);
  const start = (specCol + 1) % SPEC_COLS;
  for (let k = 0; k < nFreq; k++) {
    out[k] = new Float32Array(SPEC_COLS);
    for (let i = 0; i < SPEC_COLS; i++) {
      out[k][i] = spec[k][(start + i) % SPEC_COLS];
    }
  }
  return out;
}


function heatmapLayout() {
  return {
    paper_bgcolor: SURFACE_BG, plot_bgcolor: SURFACE_BG,
    margin: { l: 32, r: 6, t: 4, b: 22 },
    xaxis: {
      visible: false, showgrid: false, zeroline: false,
    },
    yaxis: {
      title: { text: "kHz", font: { color: TEXT_MID, size: 10 } },
      tickfont: { color: TEXT_LO, size: 9 },
      gridcolor: "#1f2a16",
    },
  };
}


/* ---------- per-mic waveforms ------------------------------------ */
function buildMicWaveforms() {
  const host = $("#mic-rows");
  if (!host) return;
  host.innerHTML = "";

  const x = new Array(frameSize);
  for (let i = 0; i < frameSize; i++) x[i] = i;

  for (let m = 0; m < nMics; m++) {
    const row = document.createElement("div");
    row.className = "mic-row";
    const tag = document.createElement("span");
    tag.className = "mic-tag";
    const refMic = cfg.mwf.ref_mic;
    if (m + 1 === refMic) tag.classList.add("is-ref");
    tag.textContent = (m + 1 === refMic) ? `Mic ${m + 1} (ref)` : `Mic ${m + 1}`;
    const plotHost = document.createElement("div");
    plotHost.className = "mic-wave";
    plotHost.id = `wave-mic-${m + 1}`;
    row.appendChild(tag);
    row.appendChild(plotHost);
    host.appendChild(row);

    if (!window.Plotly) continue;
    window.Plotly.newPlot(plotHost, [{
      type: "scattergl", mode: "lines",
      x, y: new Array(frameSize).fill(0),
      line: { color: MIC_COLORS[m % MIC_COLORS.length], width: 1 },
    }], waveLayout(), {
      displaylogo: false, responsive: true, staticPlot: true,
    });
  }
}


function redrawMicWaveforms() {
  if (!window.Plotly || !micBufs.length) return;
  for (let m = 0; m < nMics; m++) {
    const host = document.getElementById(`wave-mic-${m + 1}`);
    if (!host) continue;
    window.Plotly.restyle(host, { y: [Array.from(micBufs[m])] });
  }
}


/* ---------- MWF output waveform ---------------------------------- */
function buildMwfWaveform() {
  const host = mountPlot("#card-mwf-wave");
  if (!host) return;
  const x = new Array(frameSize);
  for (let i = 0; i < frameSize; i++) x[i] = i;
  window.Plotly.newPlot(host, [{
    type: "scattergl", mode: "lines",
    x, y: new Array(frameSize).fill(0),
    line: { color: ACCENT, width: 1.2 },
  }], waveLayout({ yrange: 1.05 }), {
    displaylogo: false, responsive: true, staticPlot: true,
  });
}


function redrawMwfWaveform() {
  if (!mwfBuf) return;
  const host = document.querySelector("#card-mwf-wave .plotly-host");
  if (!host || !window.Plotly) return;
  window.Plotly.restyle(host, { y: [Array.from(mwfBuf)] });
}


function waveLayout(opts = {}) {
  const yr = opts.yrange ?? 1.05;
  return {
    paper_bgcolor: SURFACE_BG, plot_bgcolor: SURFACE_BG,
    margin: { l: 6, r: 6, t: 4, b: 4 },
    xaxis: { visible: false },
    yaxis: {
      visible: false,
      range: [-yr, yr],
      fixedrange: true,
    },
  };
}


/* ---------- VAD trace -------------------------------------------- */
function buildVad() {
  const host = mountPlot("#card-vad-trace");
  if (!host) return;
  const x = new Array(VAD_LEN);
  for (let i = 0; i < VAD_LEN; i++) x[i] = i;
  window.Plotly.newPlot(host, [
    {
      type: "scatter", mode: "lines",
      x, y: new Array(VAD_LEN).fill(0),
      line: { color: ACCENT, width: 1.5 },
      fill: "tozeroy",
      fillcolor: "rgba(70,158,0,0.20)",
      name: "VAD",
    },
    {
      type: "scatter", mode: "lines",
      x: [0, VAD_LEN - 1], y: [0.5, 0.5],
      line: { color: TEXT_LO, dash: "dot", width: 1 },
      name: "threshold",
      hoverinfo: "skip",
    },
  ], {
    paper_bgcolor: SURFACE_BG, plot_bgcolor: SURFACE_BG,
    margin: { l: 24, r: 6, t: 4, b: 22 },
    showlegend: false,
    xaxis: {
      title: { text: "frames (past → now)", font: { color: TEXT_MID, size: 9 } },
      tickfont: { color: TEXT_LO, size: 9 },
      showgrid: false, zeroline: false,
    },
    yaxis: {
      range: [-0.05, 1.1],
      tickvals: [0, 0.5, 1],
      tickfont: { color: TEXT_LO, size: 9 },
      gridcolor: "#1f2a16",
    },
  }, { displaylogo: false, responsive: true });
}


function redrawVad() {
  if (!window.Plotly) return;
  const host = document.querySelector("#card-vad-trace .plotly-host");
  if (!host) return;
  // Roll the ring so the newest sample is at index VAD_LEN-1.
  const ordered = new Array(VAD_LEN);
  for (let i = 0; i < VAD_LEN; i++) {
    ordered[i] = vadScores[(vadPtr + i) % VAD_LEN];
  }
  window.Plotly.restyle(host, { y: [ordered] }, [0]);
}


/* ---------- redraw throttle -------------------------------------- */
function scheduleRedraw() {
  if (redrawPending) return;
  redrawPending = true;
  requestAnimationFrame(() => {
    redrawPending = false;
    try {
      redrawMicWaveforms();
      redrawMwfWaveform();
      redrawVad();
      // Spectrograms are heavier — throttle.
      if ((lastFrameIdx % specRedrawEvery) === 0) {
        redrawSpec("#card-spec-noisy", specNoisy);
        redrawSpec("#card-spec-mwf",   specMwf);
      }
    } catch (err) {
      console.warn("Q-WiSE: plot redraw error:", err);
    }
  });
}


/* ---------- DOM helpers ------------------------------------------ */
function $(sel) { return document.querySelector(sel); }

function mountPlot(cardSelector) {
  const card = document.querySelector(`${cardSelector} .card-body`);
  if (!card) return null;
  card.classList.remove("card-placeholder");
  card.innerHTML = "";
  const host = document.createElement("div");
  host.className = "plotly-host";
  host.style.width = "100%";
  host.style.height = "100%";
  card.appendChild(host);
  return host;
}
