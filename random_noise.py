import os
import re
import glob
import numpy as np
import librosa
import soundfile as sf


def parse_filename(filename):
    """
    Parse NOIZEUS-style filename like 'sp01_airport_sn0.wav'
    Returns (speaker, noise_type, snr_db).
    """
    name = os.path.splitext(filename)[0]
    match = re.match(r'(sp\d+)_(.+)_sn([n\-]?\d+)', name)
    if not match:
        return None, None, None
    speaker = match.group(1)
    noise_type = match.group(2)
    snr_db = int(match.group(3).replace('n', '-'))
    return speaker, noise_type, snr_db


def extract_noise(noisy, clean):
    """Extract noise component by subtracting clean speech from noisy signal."""
    min_len = min(len(noisy), len(clean))
    return noisy[:min_len] - clean[:min_len]


def scale_noise_to_snr(noise_chunk, reference_rms, target_snr_db):
    """Scale a noise chunk so that it has the right level relative to reference_rms."""
    noise_rms = np.sqrt(np.mean(noise_chunk ** 2)) + 1e-10
    target_noise_rms = reference_rms / (10 ** (target_snr_db / 20.0))
    return noise_chunk * (target_noise_rms / noise_rms)


def get_noise_chunk(noise_signal, length, reference_rms, snr_db):
    """
    Create a noise chunk of the given length using real noise from the file,
    scaled to match the SNR level encoded in the filename.
    """
    if len(noise_signal) == 0:
        return np.zeros(length)

    # Loop or randomly crop the noise to fill the desired length
    if len(noise_signal) >= length:
        start = np.random.randint(0, len(noise_signal) - length + 1)
        chunk = noise_signal[start:start + length]
    else:
        repeats = length // len(noise_signal) + 1
        chunk = np.tile(noise_signal, repeats)[:length]

    # Scale to match the file's SNR
    chunk = scale_noise_to_snr(chunk, reference_rms, snr_db)
    return chunk


def add_augmentations(data, sr, noise_signal, snr_db, duration_range=(0.5, 1.5)):
    """
    Randomly injects real noise (matching the file's noise type and SNR)
    at the start, middle, and end of the audio signal.
    Returns (augmented_noisy, augmented_clean, mid_point) so clean can be padded identically.
    """
    reference_rms = np.sqrt(np.mean(data ** 2)) + 1e-10

    start_len = int(np.random.uniform(*duration_range) * sr)
    mid_len = int(np.random.uniform(*duration_range) * sr)
    end_len = int(np.random.uniform(*duration_range) * sr)

    start_chunk = get_noise_chunk(noise_signal, start_len, reference_rms, snr_db)
    mid_chunk = get_noise_chunk(noise_signal, mid_len, reference_rms, snr_db)
    end_chunk = get_noise_chunk(noise_signal, end_len, reference_rms, snr_db)

    mid_point = np.random.randint(0, len(data))

    augmented = np.concatenate([
        start_chunk,
        data[:mid_point],
        mid_chunk,
        data[mid_point:],
        end_chunk
    ])

    return augmented, (start_len, mid_len, end_len, mid_point)


def add_silence_to_clean(clean_data, params):
    """
    Insert silent frames into the clean signal at the same positions/lengths
    as the noise chunks in the noisy signal, so both have identical duration.
    """
    start_len, mid_len, end_len, mid_point = params

    min_len = min(mid_point, len(clean_data))
    augmented_clean = np.concatenate([
        np.zeros(start_len),
        clean_data[:min_len],
        np.zeros(mid_len),
        clean_data[min_len:],
        np.zeros(end_len)
    ])

    return augmented_clean


def main():
    # --- Configuration ---
    input_dir = '/Users/javad/Data/Work/Datasets/vad/noise/NOIZEUS/resampled_16k'
    output_dir = '/Users/javad/Data/Work/Datasets/vad/noise/NOIZEUS/resampled_16k/rand_noise'
    os.makedirs(output_dir, exist_ok=True)

    wav_files = glob.glob(os.path.join(input_dir, "*.wav"))

    if not wav_files:
        print(f"No .wav files found in {input_dir}")
        return

    print(f"Found {len(wav_files)} files. Starting augmentation...")

    # Track aug_params per speaker (use the first noisy file's params for the clean file)
    speaker_params = {}

    # First pass: process noisy files
    for file_path in wav_files:
        filename = os.path.basename(file_path)

        if re.match(r'^sp\d+\.wav$', filename):
            continue

        speaker, noise_type, snr_db = parse_filename(filename)
        if speaker is None:
            print(f"Skipping (cannot parse filename): {filename}")
            continue

        data, sr = librosa.load(file_path, sr=None)

        clean_path = os.path.join(input_dir, f"{speaker}.wav")
        if not os.path.exists(clean_path):
            print(f"Clean file not found for {filename}, skipping. Expected: {clean_path}")
            continue

        clean_data, _ = librosa.load(clean_path, sr=sr)
        noise_signal = extract_noise(data, clean_data)

        augmented_data, aug_params = add_augmentations(data, sr, noise_signal, snr_db)

        output_path = os.path.join(output_dir, filename)
        sf.write(output_path, augmented_data, sr)

        # Save first aug_params for this speaker's clean file
        if speaker not in speaker_params:
            speaker_params[speaker] = (aug_params, sr)

        print(f"Success: {filename} (noise={noise_type}, SNR={snr_db}dB)")

    # Second pass: process clean files with matching silence
    for file_path in wav_files:
        filename = os.path.basename(file_path)
        if not re.match(r'^sp\d+\.wav$', filename):
            continue

        speaker = os.path.splitext(filename)[0]
        if speaker not in speaker_params:
            print(f"No noisy file processed for {filename}, skipping clean.")
            continue

        aug_params, sr = speaker_params[speaker]
        clean_data, _ = librosa.load(file_path, sr=sr)
        augmented_clean = add_silence_to_clean(clean_data, aug_params)

        output_path = os.path.join(output_dir, filename)
        sf.write(output_path, augmented_clean, sr)
        print(f"Success (clean): {filename} with silence frames matching noisy")

    print("\nAugmentation Complete!")


if __name__ == "__main__":
    main()