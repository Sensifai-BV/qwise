# VAD Analyzer – Threshold Documentation & Tuning Guide

---

## 1. Ground Truth (GT) VAD — `energy_vad_gt()`

Ground truth is **not** produced by a neural model. It is built purely from the **energy envelope of the clean audio signal**, making it a reliable and model-independent reference.

### Processing Pipeline

```
Clean File → Framing → RMS per frame (dB) → Adaptive Threshold → Morphological Smoothing → Binary labels (0/1)
```

---

### GT Threshold Parameters

| Parameter | Default | Description |
|---|---|---|
| `energy_th_db` | `-55.0 dB` | **Absolute floor threshold** – frames with RMS below this are always silence, regardless of the peak level |
| `relative_th_db` | `35.0 dB` | **Relative threshold** – subtracted from peak RMS to compute a signal-adaptive threshold |
| `frame_dur` | `0.01 s` | Duration of each analysis frame (10 ms) |
| `merge_gap` | `0.10 s` | Silence gaps shorter than this are filled in (merged into speech) |
| `min_dur` | `0.05 s` | Speech segments shorter than this are discarded as spurious detections |
| `min_silence` | `0.15 s` | Silence segments shorter than this are merged away |

---

### Adaptive Threshold Logic

```python
adaptive_th = max(energy_th_db, peak_db - relative_th_db)
```

The final threshold used is the **larger** of:
- `energy_th_db` — an absolute floor that prevents the threshold from going too low
- `peak_db − relative_th_db` — a signal-relative level anchored to the loudest frame

**Example A — Normal recording** (`peak = −10 dB`):
```
relative threshold = −10 − 35 = −45 dB
adaptive_th = max(−55, −45) = −45 dB   ← used
```
Frames above `−45 dB` → **speech**, below → **silence**.

**Example B — Quiet recording** (`peak = −30 dB`):
```
relative threshold = −30 − 35 = −65 dB
adaptive_th = max(−55, −65) = −55 dB   ← floor kicks in
```
The absolute floor prevents the threshold from dropping into the noise floor, avoiding over-detection.

---

### Three-Pass Morphological Smoothing

```
Raw output:   ████ ░░ ████████ ░ ████   (░ = silence, █ = speech)

Pass 1 – merge_gap (0.10 s):   silences ≤ 100 ms are filled → continuous speech blocks
Pass 2 – min_dur   (0.05 s):   speech bursts < 50 ms are removed → removes click/energy artefacts
Pass 3 – min_silence (0.15 s): remaining silence gaps < 150 ms are merged → no syllable-boundary fragmentation
```

---

### Suggested GT Parameters for Better Detection

| Scenario | `energy_th_db` | `relative_th_db` | `merge_gap` | `min_dur` | `min_silence` |
|---|---|---|---|---|---|
| **Default (NOIZEUS)** | `-55 dB` | `35 dB` | `0.10 s` | `0.05 s` | `0.15 s` |
| **Very quiet speakers** | `-60 dB` | `30 dB` | `0.15 s` | `0.05 s` | `0.10 s` |
| **High-energy noise floor** | `-45 dB` | `40 dB` | `0.10 s` | `0.08 s` | `0.20 s` |
| **Fast / clipped speech** | `-55 dB` | `35 dB` | `0.20 s` | `0.03 s` | `0.10 s` |
| **Read speech (long pauses)** | `-55 dB` | `35 dB` | `0.05 s` | `0.10 s` | `0.30 s` |

> **Key rule:** If the GT is missing speech at the start/end of utterances, **lower `relative_th_db`** (e.g. 30 dB) or **lower `energy_th_db`** (e.g. −60 dB).  
> If the GT is marking background noise as speech, **raise `energy_th_db`** or **increase `relative_th_db`**.

---

## 2. Neural VAD Model Thresholds (SpeechBrain & Silero)

These thresholds control how the **neural model probability output** is converted to binary speech/silence labels when processing the **noisy audio**.

| Parameter | Default | Description |
|---|---|---|
| `activation_th` | `0.50` | Probability ≥ this value **activates** (starts) a speech segment |
| `deactivation_th` | `0.25` | Probability < this value **deactivates** (ends) a speech segment |
| `close_th` | `0.25 s` | Speech segments closer than 250 ms are merged together |
| `len_th` | `0.25 s` | Speech segments shorter than 250 ms are discarded |

---

### Hysteresis Mechanism

The two-threshold design prevents rapid on/off flickering at decision boundaries:

```
Model probability:  0.1  0.3  0.6  0.7  0.4  0.3  0.2  0.1
                              ↑ ≥0.5 → activated
                                               ↑ <0.25 → deactivated
Binary output:       0    0    1    1    1    1    0    0
```

Once speech is activated (≥ `activation_th`), it stays active until the probability drops below `deactivation_th`. This bridges minor dips in model confidence during continuous speech.

---

### Suggested Neural VAD Parameters for Better Detection

| Scenario | `activation_th` | `deactivation_th` | `close_th` | `len_th` |
|---|---|---|---|---|
| **Default** | `0.50` | `0.25` | `0.25 s` | `0.25 s` |
| **Low SNR / heavy noise (sn0–sn5)** | `0.40` | `0.20` | `0.30 s` | `0.20 s` |
| **High SNR / clean conditions (sn15+)** | `0.55` | `0.30` | `0.20 s` | `0.25 s` |
| **Reduce false alarms (FP)** | `0.60` | `0.35` | `0.25 s` | `0.30 s` |
| **Reduce missed speech (FN)** | `0.35` | `0.15` | `0.35 s` | `0.15 s` |
| **Short / burst speech** | `0.45` | `0.20` | `0.15 s` | `0.10 s` |

> **Key rule:**  
> - To catch **more speech** (fewer missed detections / lower FN): lower `activation_th` and `deactivation_th`.  
> - To reject **more noise** (fewer false alarms / lower FP): raise both thresholds.  
> - `close_th` and `len_th` act as post-processing smoothing — increase them in noisy conditions to prevent fragmented output.

---

## 3. Complete Processing Flow

```
                  ┌─────────────────────────────────┐
  Clean File ────→│ energy_vad_gt()                  │──→ GT labels (0/1)
                  │ adaptive_th + 3-pass smoothing   │         │
                  └─────────────────────────────────┘         │
                                                             compare
                  ┌─────────────────────────────────┐         │
  Noisy File ────→│ Neural VAD (SpeechBrain / Silero)│──→ Predicted labels ──→ TP / TN / FP / FN
                  │ activation_th + deactivation_th  │
                  │ + close_th + len_th              │
                  └─────────────────────────────────┘
```

**Important:** GT is entirely independent of the neural model — it is derived solely from the energy of the clean signal. The assumption is that wherever the clean signal has high energy = speech, and low energy = silence.

---

## 4. Quick Tuning Reference

| Observed Problem | Likely Cause | Suggested Fix |
|---|---|---|
| GT misses soft speech onsets/offsets | `energy_th_db` or `relative_th_db` too high | Lower `energy_th_db` to `−60 dB` or `relative_th_db` to `30 dB` |
| GT marks noise/silence as speech | Threshold too low | Raise `energy_th_db` to `−45 dB` |
| GT fragments speech at pauses | `merge_gap` too small | Increase `merge_gap` to `0.20–0.30 s` |
| GT keeps very short noise bursts | `min_dur` too small | Increase `min_dur` to `0.08–0.10 s` |
| Model misses speech in heavy drone noise | `activation_th` too high | Lower to `0.35–0.40` |
| Model outputs too many short false alarms | `len_th` too small | Increase `len_th` to `0.30–0.40 s` |
| Model output is fragmented / choppy | `close_th` too small | Increase `close_th` to `0.30–0.40 s` |
| Model activates on drone harmonics | `activation_th` too low | Raise to `0.55–0.65` |

