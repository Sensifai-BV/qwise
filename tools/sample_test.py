import os
import glob
import warnings
from pathlib import Path
from typing import List, Optional

import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from packaging import version

torch.set_num_threads(1)


# Default values for command-line arguments
DEFAULT_ONNX_MODEL_PATH = "../models/qwise_vad_int8.onnx"
DEFAULT_NOISY_INPUT_DIR = "../datasets/noise"
DEFAULT_CLEAN_OUTPUT_DIR = "../datasets/clean"
DEFAULT_RESULT_OUTPUT_PATH = "../datasets/result.png"
DEFAULT_SAMPLING_RATE = 8000
DEFAULT_VAD_THRESHOLD = 0.3
DEFAULT_FORCE_ONNX_CPU = True

class OnnxWrapper:

    def __init__(self, path: str, force_onnx_cpu: bool = True):
        import onnxruntime

        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1

        if force_onnx_cpu and "CPUExecutionProvider" in onnxruntime.get_available_providers():
            self.session = onnxruntime.InferenceSession(
                path, providers=["CPUExecutionProvider"], sess_options=opts
            )
        else:
            self.session = onnxruntime.InferenceSession(path, sess_options=opts)

        self.reset_states()

        if "16k" in path:
            warnings.warn("This model supports only 16000 Hz sampling rate!")
            self.sample_rates = [16000]
        else:
            self.sample_rates = [8000, 16000]

    # ------------------------------------------------------------------
    def _validate_input(self, x: torch.Tensor, sr: int):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() > 2:
            raise ValueError(f"Too many dimensions for input audio chunk: {x.dim()}")

        if sr != 16000 and (sr % 16000 == 0):
            step = sr // 16000
            x = x[:, ::step]
            sr = 16000

        if sr not in self.sample_rates:
            raise ValueError(
                f"Supported sampling rates: {self.sample_rates} (or multiples of 16000)"
            )
        if sr / x.shape[1] > 31.25:
            raise ValueError("Input audio chunk is too short")

        return x, sr

    # ------------------------------------------------------------------
    def reset_states(self, batch_size: int = 1):
        self._state = torch.zeros((2, batch_size, 128)).float()
        self._context = torch.zeros(0)
        self._last_sr = 0
        self._last_batch_size = 0

    # ------------------------------------------------------------------
    def __call__(self, x: torch.Tensor, sr: int) -> torch.Tensor:
        x, sr = self._validate_input(x, sr)
        num_samples = 512 if sr == 16000 else 256
        if x.shape[-1] != num_samples:
            raise ValueError(
                f"Expected {num_samples} samples per chunk for sr={sr}, got {x.shape[-1]}"
            )

        batch_size = x.shape[0]
        context_size = 64 if sr == 16000 else 32

        if not self._last_batch_size:
            self.reset_states(batch_size)
        if self._last_sr and self._last_sr != sr:
            self.reset_states(batch_size)
        if self._last_batch_size and self._last_batch_size != batch_size:
            self.reset_states(batch_size)

        if not len(self._context):
            self._context = torch.zeros(batch_size, context_size)

        x = torch.cat([self._context, x], dim=1)

        ort_inputs = {
            "input": x.numpy(),
            "state": self._state.numpy(),
            "sr": np.array(sr, dtype="int64"),
        }
        ort_outs = self.session.run(None, ort_inputs)
        out, state = ort_outs
        self._state = torch.from_numpy(state)

        self._context = x[..., -context_size:]
        self._last_sr = sr
        self._last_batch_size = batch_size

        return torch.from_numpy(out)


def read_audio(path: str, sampling_rate: int = 16000) -> torch.Tensor:
    """Load an audio file and resample to *sampling_rate*, returns 1-D tensor."""
    ta_ver = version.parse(torchaudio.__version__)
    if ta_ver < version.parse("2.9"):
        try:
            effects = [["channels", "1"], ["rate", str(sampling_rate)]]
            wav, sr = torchaudio.sox_effects.apply_effects_file(path, effects=effects)
        except Exception:
            wav, sr = torchaudio.load(path)
    else:
        try:
            wav, sr = torchaudio.load(path)
        except Exception:
            try:
                from torchcodec.decoders import AudioDecoder
                samples = AudioDecoder(path).get_all_samples()
                wav, sr = samples.data, samples.sample_rate
            except ImportError:
                raise RuntimeError(
                    f"torchaudio {torchaudio.__version__} requires torchcodec for audio I/O. "
                    "Install torchcodec or pin torchaudio < 2.9"
                )

    # Mix down to mono
    if wav.ndim > 1 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != sampling_rate:
        wav = torchaudio.transforms.Resample(sr, sampling_rate)(wav)

    return wav.squeeze(0)


def save_audio(path: str, tensor: torch.Tensor, sampling_rate: int = 16000):
    """Save a 1-D or 2-D tensor as a 16-bit WAV file."""
    tensor = tensor.detach().cpu()
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)

    ta_ver = version.parse(torchaudio.__version__)
    try:
        torchaudio.save(path, tensor, sampling_rate, bits_per_sample=16)
    except Exception:
        if ta_ver >= version.parse("2.9"):
            try:
                from torchcodec.encoders import AudioEncoder
                AudioEncoder(tensor, sample_rate=sampling_rate).to_file(path)
            except ImportError:
                raise RuntimeError(
                    "Install torchcodec or pin torchaudio < 2.9 for saving audio."
                )
        else:
            raise


def collect_chunks(tss: List[dict], wav: torch.Tensor) -> torch.Tensor:
    """Concatenate speech segments from *wav* using sample-index timestamps."""
    chunks = [wav[seg["start"]: seg["end"]] for seg in tss]
    return torch.cat(chunks)

@torch.no_grad()
def get_speech_timestamps(
    audio: torch.Tensor,
    model,
    threshold: float = 0.5,
    sampling_rate: int = 16000,
    min_speech_duration_ms: int = 250,
    max_speech_duration_s: float = float("inf"),
    min_silence_duration_ms: int = 100,
    speech_pad_ms: int = 30,
    return_seconds: bool = False,
) -> List[dict]:
    """
    Return a list of {start, end} dicts (sample indices, or seconds when
    *return_seconds=True*) identifying speech regions in *audio*.
    """
    if not torch.is_tensor(audio):
        try:
            audio = torch.Tensor(audio)
        except Exception:
            raise TypeError("Audio cannot be cast to tensor.")

    if audio.dim() == 2 and audio.shape[0] == 1:
        audio = audio.squeeze(0)
    if audio.dim() != 1:
        raise ValueError("Audio must be a 1-D tensor.")

    num_samples = 512 if sampling_rate == 16000 else 256
    min_speech_samples = sampling_rate * min_speech_duration_ms / 1000
    speech_pad_samples = sampling_rate * speech_pad_ms / 1000
    min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
    min_silence_samples_at_max = sampling_rate * 98 / 1000   # 98 ms hard floor
    max_speech_samples = (
        sampling_rate * max_speech_duration_s - num_samples - 2 * speech_pad_samples
        if max_speech_duration_s < float("inf")
        else float("inf")
    )

    audio_length_samples = len(audio)

    # Pad to a multiple of num_samples
    if audio_length_samples % num_samples:
        pad = num_samples - (audio_length_samples % num_samples)
        audio = torch.nn.functional.pad(audio, (0, pad))

    model.reset_states()

    speech_probs: List[float] = []
    for start in range(0, len(audio), num_samples):
        chunk = audio[start: start + num_samples]
        speech_prob = model(chunk, sampling_rate).item()
        speech_probs.append(speech_prob)

    triggered = False
    speeches: List[dict] = []
    current_speech: dict = {}
    neg_threshold = threshold - 0.15
    temp_end = 0
    prev_end = next_start = 0

    for i, speech_prob in enumerate(speech_probs):
        if speech_prob >= threshold and temp_end:
            temp_end = 0
            if next_start < prev_end:
                next_start = num_samples * i

        if speech_prob >= threshold and not triggered:
            triggered = True
            current_speech["start"] = max(0, num_samples * i - speech_pad_samples)
            continue

        if triggered and num_samples * i - current_speech["start"] > max_speech_samples:
            if prev_end:
                current_speech["end"] = prev_end
                speeches.append(current_speech)
                current_speech = {}
                if next_start < prev_end:
                    triggered = False
                else:
                    current_speech["start"] = next_start
                prev_end = next_start = temp_end = 0
            else:
                current_speech["end"] = num_samples * i
                speeches.append(current_speech)
                current_speech = {}
                prev_end = next_start = temp_end = 0
                triggered = False
            continue

        if speech_prob < neg_threshold and triggered:
            if not temp_end:
                temp_end = num_samples * i
            if num_samples * i - temp_end > min_silence_samples_at_max:
                prev_end = temp_end
            if num_samples * i - temp_end < min_silence_samples:
                continue
            else:
                current_speech["end"] = temp_end
                if (
                    current_speech["end"] - current_speech["start"]
                ) > min_speech_samples:
                    speeches.append(current_speech)
                current_speech = {}
                prev_end = next_start = temp_end = 0
                triggered = False
                continue

    if current_speech and (
        audio_length_samples - current_speech["start"]
    ) > min_speech_samples:
        current_speech["end"] = audio_length_samples
        speeches.append(current_speech)

    # Pad speech boundaries
    for i, speech in enumerate(speeches):
        speech["start"] = int(max(0, speech["start"] - speech_pad_samples))
        if i < len(speeches) - 1:
            silence = speeches[i + 1]["start"] - speech["end"]
            if silence < 2 * speech_pad_samples:
                speech["end"] = int(
                    min(audio_length_samples, speech["end"] + silence // 2)
                )
                speeches[i + 1]["start"] = int(
                    max(0, speeches[i + 1]["start"] - silence // 2)
                )
            else:
                speech["end"] = int(
                    min(audio_length_samples, speech["end"] + speech_pad_samples)
                )
                speeches[i + 1]["start"] = int(
                    max(0, speeches[i + 1]["start"] - speech_pad_samples)
                )
        else:
            speech["end"] = int(
                min(audio_length_samples, speech["end"] + speech_pad_samples)
            )

    if return_seconds:
        audio_length_seconds = audio_length_samples / sampling_rate
        for sd in speeches:
            sd["start"] = max(round(sd["start"] / sampling_rate, 1), 0)
            sd["end"] = min(
                round(sd["end"] / sampling_rate, 1), audio_length_seconds
            )

    return speeches

def create_vad_visualization(
    wav: torch.Tensor,
    speech_timestamps: List[dict],
    output_path: str,
    title: str,
):
    wav_np = wav.numpy() if torch.is_tensor(wav) else wav
    duration = len(wav_np) / args.sampling_rate
    time = np.linspace(0, duration, len(wav_np))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    # — Waveform panel
    ax1.plot(time, wav_np, linewidth=0.5, color="gray", alpha=0.7)
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("Amplitude")
    ax1.set_title(f"{title} — Waveform with Speech Detection")
    ax1.grid(True, alpha=0.3)
    for seg in speech_timestamps:
        ax1.axvspan(
            seg["start"] / args.sampling_rate,
            seg["end"] / args.sampling_rate,
            alpha=0.3, color="green", label="Speech",
        )
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys())

    # — Binary mask panel
    speech_mask = np.zeros(len(wav_np))
    for seg in speech_timestamps:
        speech_mask[seg["start"]: seg["end"]] = 1

    ax2.fill_between(time, 0, speech_mask, color="green", alpha=0.6, label="Speech")
    ax2.fill_between(time, speech_mask, 1, color="red", alpha=0.6, label="Noise")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Detection")
    ax2.set_title("Speech Activity Detection (Green = Speech, Red = Noise)")
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["Noise", "Speech"])
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    speech_ratio = speech_mask.sum() / len(speech_mask) * 100
    fig.suptitle(
        f"{title}\nSpeech: {speech_ratio:.1f}%  |  Noise: {100 - speech_ratio:.1f}%",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def create_summary_visualization(
    results: dict,
    output_path: str = "datasets/result.png",
):
    file_stats = []
    for wav_path, speech_timestamps in results.items():
        if not speech_timestamps:
            continue
        try:
            wav = read_audio(wav_path, sampling_rate=args.sampling_rate)
            total_samples = len(wav)
            speech_samples = sum(s["end"] - s["start"] for s in speech_timestamps)
            noise_samples = total_samples - speech_samples
            file_stats.append(
                {
                    "filename": Path(wav_path).stem,
                    "speech_pct": speech_samples / total_samples * 100,
                    "noise_pct": noise_samples / total_samples * 100,
                    "total_duration": total_samples / args.sampling_rate,
                    "speech_duration": speech_samples / args.sampling_rate,
                    "noise_duration": noise_samples / args.sampling_rate,
                }
            )
        except Exception as e:
            print(f"  Warning: stats skipped for {wav_path}: {e}")

    if not file_stats:
        print("  No statistics to visualise.")
        return

    file_stats.sort(key=lambda x: x["filename"])

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Stacked bar chart
    ax1 = fig.add_subplot(gs[0:2, :])
    filenames = [s["filename"] for s in file_stats]
    speech_pcts = [s["speech_pct"] for s in file_stats]
    noise_pcts = [s["noise_pct"] for s in file_stats]
    x_pos = np.arange(len(filenames))

    ax1.bar(x_pos, speech_pcts, 0.8, label="Speech", color="green", alpha=0.7)
    ax1.bar(x_pos, noise_pcts, 0.8, bottom=speech_pcts, label="Noise", color="red", alpha=0.7)
    ax1.set_xlabel("Audio Files", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Percentage (%)", fontsize=12, fontweight="bold")
    ax1.set_title("Speech vs Noise Detection — Per File Analysis", fontsize=14, fontweight="bold")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(filenames, rotation=45, ha="right", fontsize=8)
    ax1.set_ylim(0, 100)
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(True, alpha=0.3, axis="y")
    for i, (sp, np_) in enumerate(zip(speech_pcts, noise_pcts)):
        if sp > 5:
            ax1.text(i, sp / 2, f"{sp:.1f}%", ha="center", va="center",
                     fontsize=7, fontweight="bold", color="white")
        if np_ > 5:
            ax1.text(i, sp + np_ / 2, f"{np_:.1f}%", ha="center", va="center",
                     fontsize=7, fontweight="bold", color="white")

    # Pie chart
    ax2 = fig.add_subplot(gs[2, 0])
    total_speech = sum(s["speech_duration"] for s in file_stats)
    total_noise = sum(s["noise_duration"] for s in file_stats)
    total_dur = total_speech + total_noise
    ax2.pie(
        [total_speech / total_dur * 100, total_noise / total_dur * 100],
        labels=["Speech", "Noise"],
        autopct="%1.1f%%",
        colors=["green", "red"],
        explode=(0.05, 0.05),
        shadow=True,
        startangle=90,
        textprops={"fontsize": 10, "fontweight": "bold"},
    )
    ax2.set_title("Overall Speech vs Noise Distribution", fontsize=12, fontweight="bold")

    # Summary text
    ax3 = fig.add_subplot(gs[2, 1])
    ax3.axis("off")
    summary = (
        f"  SUMMARY STATISTICS\n"
        f"  {'='*38}\n"
        f"  Total Files Processed : {len(file_stats)}\n"
        f"  Total Duration        : {total_dur:.2f} s\n\n"
        f"  Speech:\n"
        f"    Total   : {total_speech:.2f} s\n"
        f"    Share   : {total_speech/total_dur*100:.1f}%\n"
        f"    Avg/file: {total_speech/len(file_stats):.2f} s\n\n"
        f"  Noise:\n"
        f"    Total   : {total_noise:.2f} s\n"
        f"    Share   : {total_noise/total_dur*100:.1f}%\n"
        f"    Avg/file: {total_noise/len(file_stats):.2f} s\n"
    )
    ax3.text(
        0.05, 0.5, summary,
        fontsize=9, verticalalignment="center", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    fig.suptitle(
        f"Qwise VAD — Complete Analysis Report\n{len(file_stats)} Audio Files Processed",
        fontsize=16, fontweight="bold", y=0.98,
    )
    os.makedirs(Path(output_path).parent, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n  Summary visualisation saved: {output_path}")
    plt.close()


def process_audio_file(
    wav_path: str,
    model: OnnxWrapper,
    clean_output_dir: str,
) -> Optional[List[dict]]:
    """
    Run VAD on *wav_path*, save the clean (speech-only) audio, and
    save a per-file visualisation PNG alongside the clean audio.

    Returns the list of speech timestamp dicts, or None if nothing detected.
    """
    os.makedirs(clean_output_dir, exist_ok=True)

    print(f"Processing: {wav_path}")
    wav = read_audio(wav_path, sampling_rate=args.sampling_rate)

    speech_timestamps = get_speech_timestamps(
        wav, model,
        threshold=args.vad_threshold,
        sampling_rate=args.sampling_rate,
    )

    if not speech_timestamps:
        print(f"  No speech detected in {wav_path}")
        return None

    print(f"  Found {len(speech_timestamps)} speech segment(s)")

    base_name = Path(wav_path).stem

    # Save clean audio
    clean_path = os.path.join(clean_output_dir, f"{base_name}_clean.wav")
    save_audio(clean_path, collect_chunks(speech_timestamps, wav), sampling_rate=args.sampling_rate)
    print(f"  Saved clean audio : {clean_path}")

    # Save per-file visualisation in the same clean output folder
    viz_path = os.path.join(clean_output_dir, f"{base_name}_vad.png")
    create_vad_visualization(wav, speech_timestamps, viz_path, base_name)
    print(f"  Saved visualisation: {viz_path}")

    return speech_timestamps


def main():
    print("=" * 60)
    print("Qwise VAD — Batch Audio Processor")
    print("=" * 60)


    import argparse
    parser = argparse.ArgumentParser(description="Qwise VAD — Batch Audio Processor")
    parser.add_argument('--onnx-model-path', type=str, default=DEFAULT_ONNX_MODEL_PATH, help='Path to ONNX model file')
    parser.add_argument('--noisy-input-dir', type=str, default=DEFAULT_NOISY_INPUT_DIR, help='Directory with input .wav files')
    parser.add_argument('--clean-output-dir', type=str, default=DEFAULT_CLEAN_OUTPUT_DIR, help='Directory to save cleaned audio')
    parser.add_argument('--result-output-path', type=str, default=DEFAULT_RESULT_OUTPUT_PATH, help='Path to save summary chart')
    parser.add_argument('--sampling-rate', type=int, default=DEFAULT_SAMPLING_RATE, choices=[8000, 16000], help='Sampling rate (8000 or 16000)')
    parser.add_argument('--vad-threshold', type=float, default=DEFAULT_VAD_THRESHOLD, help='Speech probability threshold (0.0 – 1.0)')
    parser.add_argument('--force-onnx-cpu', action='store_true', default=DEFAULT_FORCE_ONNX_CPU, help='Force ONNX to use CPU')
    parser.add_argument('--no-force-onnx-cpu', dest='force_onnx_cpu', action='store_false', help='Allow ONNX GPU execution providers')
    global args
    args = parser.parse_args()

    # Validate ONNX path
    if not os.path.isfile(args.onnx_model_path):
        raise FileNotFoundError(
            f"ONNX model not found: {args.onnx_model_path}\n"
            "Please set --onnx-model-path to a valid ONNX file."
        )

    print(f"\nLoading ONNX model: {args.onnx_model_path}")
    model = OnnxWrapper(args.onnx_model_path, force_onnx_cpu=args.force_onnx_cpu)
    print("Model loaded successfully!")

    # Discover input WAV files
    wav_files = sorted(glob.glob(os.path.join(args.noisy_input_dir, "*.wav")))
    if not wav_files:
        print(f"\nNo WAV files found in: {args.noisy_input_dir}")
        return

    print(f"\nFound {len(wav_files)} WAV file(s) in: {args.noisy_input_dir}")
    print(f"Clean audio  → {args.clean_output_dir}/")
    print(f"Summary chart→ {args.result_output_path}")
    print("=" * 60)

    # Process each file

    results: dict = {}
    for idx, wav_file in enumerate(wav_files, 1):
        print(f"\n[{idx}/{len(wav_files)}]")
        try:
            results[wav_file] = process_audio_file(wav_file, model, args.clean_output_dir)
        except Exception as e:
            print(f"  ERROR: {e}")
            results[wav_file] = None

    # Summary visualisation
    print("\n" + "=" * 60)
    print("Creating summary visualisation …")
    create_summary_visualization(results, output_path=args.result_output_path)

    # Final report
    successful = sum(1 for v in results.values() if v is not None)
    print("\n" + "=" * 60)
    print("Done!")
    print(f"  Processed : {successful}/{len(wav_files)} files successfully")
    print(f"  Clean audio: {args.clean_output_dir}/")
    print(f"  Summary    : {args.result_output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()