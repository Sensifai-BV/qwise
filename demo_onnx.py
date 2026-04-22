import os
import numpy as np
import onnxruntime as ort
import soundfile as sf
import librosa


class SileroVAD:
    def __init__(self, model_path, sample_rate=16000):
        # On M4 Pro, ONNX Runtime will handle the inference efficiently
        self.session = ort.InferenceSession(model_path)
        self.sample_rate = sample_rate

        # V5 models use a single state tensor [2, 1, 128]
        # (Concatenation of h and c states)
        self._state = np.zeros((2, 1, 128)).astype('float32')

    def process_chunk(self, chunk):
        """Processes a single audio chunk using the consolidated state."""
        if len(chunk) < 512:
            chunk = np.pad(chunk, (0, 512 - len(chunk)))

        # The model expects 'input', 'sr', and 'state'
        input_data = {
            'input': chunk.reshape(1, -1).astype('float32'),
            'sr': np.array([self.sample_rate], dtype='int64'),
            'state': self._state
        }

        # The model returns two outputs: [probability, next_state]
        out, self._state = self.session.run(None, input_data)

        # 'out' is typically a tensor of shape [1, 1] or [1]
        return out.item()

def main(audio_input, model_path, output_dir, threshold=0.5):
    # 1. Prepare Environment
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 2. Load Audio (Silero works best at 16kHz)
    target_sr = 16000
    audio, fs = librosa.load(audio_input, sr=target_sr)

    vad = SileroVAD(model_path, sample_rate=target_sr)

    # 3. Processing parameters (Silero uses 512 sample frames for 16kHz)
    frame_size = 512
    clean_audio = np.zeros_like(audio)
    vad_mask = np.zeros_like(audio)

    print(f"Processing {os.path.basename(audio_input)}...")

    for i in range(0, len(audio), frame_size):
        chunk = audio[i:i + frame_size]
        if len(chunk) < frame_size:
            break

        prob = vad.process_chunk(chunk)

        # Apply Threshold to create a mask
        if prob > threshold:
            clean_audio[i:i + frame_size] = chunk
            vad_mask[i:i + frame_size] = 1.0
        else:
            clean_audio[i:i + frame_size] = 0.0 # Silence non-speech

    # 4. Save Outputs
    file_name = os.path.basename(audio_input).replace(".wav", "_clean.wav")
    output_path = os.path.join(output_dir, file_name)

    sf.write(output_path, clean_audio, target_sr)

    # Also save the VAD mask (useful for your MATLAB MWF simulation)
    mask_path = os.path.join(output_dir, "vad_mask.wav")
    sf.write(mask_path, vad_mask, target_sr)

    print(f"Done! Clean audio saved to: {output_path}")

if __name__ == "__main__":
    # Update these paths for your local setup
    MODEL = "./silero-vad/src/silero_vad/data/silero_vad.onnx"
    INPUT = "./mwf/sp0-noise.wav"
    OUTPUT_FOLDER = "./mwf/outputs"

    main(INPUT, MODEL, OUTPUT_FOLDER)