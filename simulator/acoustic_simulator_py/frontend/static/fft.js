/* =========================================================================
   Q-WiSE frontend FFT — minimal iterative radix-2 magnitude FFT.

   Used by the spectrogram plots in plots.js. Frame size in this app is
   always a power of two (1024 by default), so we don't bother with
   Bluestein / mixed-radix variants.

   Exports `fftMagnitudeDb(buf, n)`:
     * `buf`  — Float32Array (or any array-like) length >= n
     * `n`    — FFT size, must be a positive power of two
   Returns a length-`n/2 + 1` Float32Array of half-spectrum magnitudes
   in dBFS (clipped at -120 dB so empty frames don't blow up the log).

   The window applied internally is the periodic Hann (same definition
   as scipy.signal.windows.hann(sym=False)) so frame energies match the
   server-side spectrogram from backend/mwf/stft.py.
   ========================================================================= */

const MIN_DB = -120;
const EPS    = 1e-12;

let cachedWindow = null;
let cachedWindowN = -1;

function periodicHann(n) {
  if (cachedWindowN === n && cachedWindow) return cachedWindow;
  const w = new Float32Array(n);
  const denom = n;                              // periodic = N (not N-1)
  for (let i = 0; i < n; i++) {
    w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / denom);
  }
  cachedWindow = w;
  cachedWindowN = n;
  return w;
}


/* In-place iterative radix-2 FFT. `re` / `im` are Float32Arrays of
   length `n`. */
function fftInPlace(re, im, n) {
  // Bit-reversal permutation.
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) {
      j ^= bit;
    }
    j ^= bit;
    if (i < j) {
      let t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }

  // Cooley-Tukey butterflies.
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1;
    const angle = (-2 * Math.PI) / len;
    const wStepRe = Math.cos(angle);
    const wStepIm = Math.sin(angle);
    for (let i = 0; i < n; i += len) {
      let wRe = 1.0;
      let wIm = 0.0;
      for (let k = 0; k < half; k++) {
        const aRe = re[i + k];
        const aIm = im[i + k];
        const bRe = re[i + k + half];
        const bIm = im[i + k + half];
        const tRe = wRe * bRe - wIm * bIm;
        const tIm = wRe * bIm + wIm * bRe;
        re[i + k]        = aRe + tRe;
        im[i + k]        = aIm + tIm;
        re[i + k + half] = aRe - tRe;
        im[i + k + half] = aIm - tIm;
        const nwRe = wRe * wStepRe - wIm * wStepIm;
        const nwIm = wRe * wStepIm + wIm * wStepRe;
        wRe = nwRe;
        wIm = nwIm;
      }
    }
  }
}


export function fftMagnitudeDb(buf, n) {
  // Defensive copy + window into work buffers.
  const win = periodicHann(n);
  const re = new Float32Array(n);
  const im = new Float32Array(n);    // already zero-filled by Float32Array
  for (let i = 0; i < n; i++) {
    re[i] = (buf[i] || 0) * win[i];
  }
  fftInPlace(re, im, n);

  const half = (n >> 1) + 1;
  const out = new Float32Array(half);
  for (let i = 0; i < half; i++) {
    const mag = Math.sqrt(re[i] * re[i] + im[i] * im[i]);
    const db  = 20 * Math.log10(mag + EPS);
    out[i] = db < MIN_DB ? MIN_DB : db;
  }
  return out;
}


export { MIN_DB as FFT_MIN_DB };
