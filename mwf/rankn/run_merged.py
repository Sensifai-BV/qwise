#!/usr/bin/env python3
"""
run_merged.py — end-to-end Silero-VAD + rank-N MWF + OMLSA enhancer.

This runs the complete combined pipeline (the same one the single merged ONNX
implements): Silero VAD drives the speech/noise masks, rank-N MWF does the
spatial filtering, OMLSA cleans the residual, and the result is written to a
clean WAV. It uses your existing silero_vad.onnx plus the rankn math.

Examples
--------
    python run_merged.py noise_speech.wav --save clean.wav
    python run_merged.py noise_speech.wav --play
    python run_merged.py noise_speech.wav --save clean.wav \
        --silero ~/Projects/qwise/silero-vad/src/silero_vad/data/silero_vad.onnx

Input WAV: multichannel (one channel per mic), 8 kHz or 16 kHz. A mono file
runs too (VAD + OMLSA only; no spatial gain).
"""

import argparse
import os
import sys
import numpy as np
import soundfile as sf

import merged_ref as MR


def main():
    ap = argparse.ArgumentParser(description="Silero-VAD + rank-N MWF + OMLSA enhancer")
    ap.add_argument("input", nargs="+",
                    help="one multichannel WAV, OR several mono mic WAVs "
                         "(e.g. mic01.wav mic02.wav mic03.wav) to stack as the array")
    ap.add_argument("--save", metavar="OUT.wav", help="write enhanced WAV")
    ap.add_argument("--play", action="store_true", help="play enhanced audio")
    ap.add_argument("--silero", default=MR.SILERO_DEFAULT, help="path to silero_vad.onnx")
    ap.add_argument("--mu", type=float, help="SDW trade-off (>=1, more = stronger)")
    ap.add_argument("--floor-db", type=float, help="OMLSA spectral floor dB (e.g. -30)")
    ap.add_argument("--no-omlsa", action="store_true", help="disable OMLSA post-filter")
    ap.add_argument("--no-gate", action="store_true", help="disable VAD output gating")
    ap.add_argument("--gate-floor", type=float, help="non-speech level 0..1 (0 = full silence)")
    ap.add_argument("--gate-hangover", type=float, help="seconds of speech hangover (default 0.2)")
    ap.add_argument("--onnx", action="store_true",
                    help="run the spatial stage via rankn_vadmask_{rate}.onnx")
    ap.add_argument("--single", action="store_true",
                    help="run the fully merged rankn_vad_{rate}.onnx (VAD embedded)")
    args = ap.parse_args()

    if not args.single and not os.path.exists(args.silero):
        # --single embeds the VAD in the graph; no host silero_vad.onnx needed
        sys.exit(f"silero model not found: {args.silero}\nPass --silero <path>.")

    # one multichannel file, or several mono mics stacked into the array
    chans, fs = [], None
    for p in args.input:
        xi, fi = sf.read(p, always_2d=True)
        fs = fs or fi
        if fi != fs:
            sys.exit(f"sample-rate mismatch: {p} is {fi} Hz, expected {fs} Hz")
        chans.append(xi)
    if len(chans) == 1:
        x = chans[0]
    else:
        L = min(c.shape[0] for c in chans)
        x = np.concatenate([c[:L, :1] for c in chans], axis=1)   # one channel per file
    if fs not in (8000, 16000):
        sys.exit(f"Unsupported sample rate {fs}; resample to 8000 or 16000 Hz.")
    print(f"[run] {len(args.input)} file(s) -> {x.shape[1]} mic(s), {fs} Hz, {x.shape[0]/fs:.1f}s")
    if x.shape[1] < 2:
        print("[run] NOTE: 1 mic -> spatial MWF is inactive; only VAD gating + "
              "OMLSA apply. Pass all mics for beamforming.")

    opts = {}
    if args.mu is not None:
        opts["mu"] = args.mu
    if args.floor_db is not None:
        opts["omlsa_floor_db"] = args.floor_db
    if args.no_omlsa:
        opts["post_omlsa"] = False
    if args.no_gate:
        opts["vad_gate"] = False
    if args.gate_floor is not None:
        opts["gate_floor"] = args.gate_floor
    if args.gate_hangover is not None:
        opts["gate_hangover_s"] = args.gate_hangover

    here = os.path.dirname(os.path.abspath(__file__))
    if args.single:
        model = os.path.join(here, f"rankn_vad_{fs//1000}k.onnx")
        if not os.path.exists(model):
            sys.exit(f"single-file model not found: {model}\n"
                     f"Run:  python build_merged.py  (needs silero_vad_{fs//1000}k.safetensors)")
        y = MR.enhance_single(x, fs, model, opts=opts)
        print(f"[run] fully merged (VAD embedded) via {os.path.basename(model)}; "
              f"enhanced {len(y)/fs:.1f}s")
    elif args.onnx:
        model = os.path.join(here, f"rankn_vadmask_{fs//1000}k.onnx")
        if not os.path.exists(model):
            sys.exit(f"merged model not found: {model}\nRun:  python build_merged.py")
        y = MR.enhance_onnx(x, fs, model, silero_path=args.silero, opts=opts)
        print(f"[run] spatial stage via {os.path.basename(model)}; enhanced {len(y)/fs:.1f}s")
    else:
        y, speech, noise = MR.enhance(x, fs, silero_path=args.silero, opts=opts)
        print(f"[run] VAD: {int(speech.sum())} speech / {int(noise.sum())} noise frames; "
              f"enhanced {len(y)/fs:.1f}s")

    if args.save:
        sf.write(args.save, y.astype(np.float32), fs)
        print(f"[run] wrote {args.save}")
    if args.play:
        try:
            import sounddevice as sd
            sd.play(y.astype(np.float32), fs); sd.wait()
        except Exception:
            print("[run] sounddevice not available; use --save instead.")
    if not args.save and not args.play:
        sf.write("clean.wav", y.astype(np.float32), fs)
        print("[run] (no --save/--play) wrote clean.wav")


if __name__ == "__main__":
    main()
