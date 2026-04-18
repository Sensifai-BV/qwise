"""
Mix clean speech files with drone noise at multiple SNR levels.
Outputs files named like: sp01_falcon-drone_sn0.wav, sp01_micro-drone_sn5.wav, etc.
"""

import os
import glob
import random
import numpy as np
import librosa
import soundfile as sf


# --- Configuration ---
CLEAN_DIR = '/Users/javad/Data/Work/Datasets/vad/clean_speech/random_noizeus'
DRONE_NOISE_DIR = '/Users/javad/Data/Work/Datasets/vad/noise/drone-raw'
OUTPUT_DIR = '/Users/javad/Data/Work/Datasets/vad/noise/drone-mixed'
SNR_LEVELS = [0, 5, 10, 15]
TARGET_SR = 16000


def load_and_resample(path, sr):
    data, orig_sr = librosa.load(path, sr=sr, mono=True)
    return data


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Mix clean speech with noise at the given SNR (dB).
    SNR = 20 * log10(rms_clean / rms_noise)
    """
    # Tile noise to cover clean length
    if len(noise) < len(clean):
        repeats = len(clean) // len(noise) + 1
        noise = np.tile(noise, repeats)

    # Random start crop
    if len(noise) > len(clean):
        start = random.randint(0, len(noise) - len(clean))
        noise = noise[start: start + len(clean)]

    rms_clean = np.sqrt(np.mean(clean ** 2)) + 1e-10
    rms_noise = np.sqrt(np.mean(noise ** 2)) + 1e-10

    # Scale noise to desired SNR
    target_rms_noise = rms_clean / (10 ** (snr_db / 20.0))
    noise_scaled = noise * (target_rms_noise / rms_noise)

    mixed = clean + noise_scaled
    return mixed


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Collect clean speech files
    clean_files = sorted(glob.glob(os.path.join(CLEAN_DIR, 'sp*.wav')))
    if not clean_files:
        print(f"No clean speech files found in {CLEAN_DIR}")
        return
    print(f"Found {len(clean_files)} clean speech files.")

    # Collect drone noise subdirectories / files
    # Structure: drone-raw/falcon-drone/*.wav, drone-raw/micro-drone/*.wav
    noise_types = {}
    for noise_subdir in sorted(os.listdir(DRONE_NOISE_DIR)):
        subdir_path = os.path.join(DRONE_NOISE_DIR, noise_subdir)
        if not os.path.isdir(subdir_path):
            continue
        wav_files = sorted(glob.glob(os.path.join(subdir_path, '*.wav')))
        if wav_files:
            noise_types[noise_subdir] = wav_files
            print(f"  Noise type '{noise_subdir}': {len(wav_files)} files")

    if not noise_types:
        print(f"No noise subdirectories found in {DRONE_NOISE_DIR}")
        return

    total = len(clean_files) * len(noise_types) * len(SNR_LEVELS)
    print(f"\nGenerating {total} mixed files...\n")

    count = 0
    for clean_path in clean_files:
        speaker = os.path.splitext(os.path.basename(clean_path))[0]  # e.g. sp01
        clean_audio = load_and_resample(clean_path, TARGET_SR)

        for noise_type, noise_files in noise_types.items():
            # Concatenate all noise files for this type into one long signal
            noise_segments = [load_and_resample(f, TARGET_SR) for f in noise_files]
            noise_audio = np.concatenate(noise_segments)

            for snr_db in SNR_LEVELS:
                mixed = mix_at_snr(clean_audio, noise_audio, snr_db)

                out_name = f"{speaker}_{noise_type}_sn{snr_db}.wav"
                out_path = os.path.join(OUTPUT_DIR, out_name)
                sf.write(out_path, mixed, TARGET_SR)
                count += 1
                print(f"[{count}/{total}] {out_name}")

    print(f"\nDone! {count} files written to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()

