"""
recon.py – Reconstruct Silero VAD 16 kHz from safetensors, then provide:
  1. Fine-tuning / QAT helpers
  2. VAD-guided Multichannel Wiener Filter (MWF) speech enhancement
  3. ONNX export
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from safetensors.torch import load_model


# ---------------------------------------------------------------------------
# 1.  Reconstructed Silero VAD (16 kHz, single-sample-rate variant)
# ---------------------------------------------------------------------------
class SileroVAD16k(nn.Module):
    """
    Architecture reverse-engineered from the JIT model and safetensors weights.

    STFT front-end:
        ReflectionPad1d(0, 64) → Conv1d(1, 258, 256, stride=128) → split → magnitude
    Encoder (4 × Conv1d + ReLU, SE is identity):
        conv1: Conv1d(129, 128, 3, pad=1) → ReLU
        conv2: Conv1d(128, 64,  3, pad=1) → ReLU
        conv3: Conv1d(64,  64,  3, pad=1) → ReLU
        conv4: Conv1d(64,  128, 3, pad=1) → ReLU
    Decoder:
        LSTMCell(128, 128)
        h → unsqueeze(-1) → Dropout(0.1) → ReLU → Conv1d(128, 1, 1) → Sigmoid
    Output:
        squeeze(1) → mean(dim=1) → unsqueeze(1) → squeeze → scalar
    """

    def __init__(self):
        super().__init__()
        # Learnt STFT front-end
        self.stft_pad = nn.ReflectionPad1d((0, 64))
        self.stft_conv = nn.Conv1d(1, 258, kernel_size=256, stride=128, bias=False)

        # Encoder convolutions
        self.conv1 = nn.Conv1d(129, 128, 3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(128, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(64, 64, 3, stride=2, padding=1)
        self.conv4 = nn.Conv1d(64, 128, 3, stride=1, padding=1)

        # Recurrent bottleneck
        self.lstm_cell = nn.LSTMCell(128, 128)

        # Decoder head: Dropout → ReLU → Conv1d → Sigmoid
        self.final_conv = nn.Conv1d(128, 1, 1)
        self.decoder_dropout = nn.Dropout(0.1)

    # ---- helpers for streaming state ----------------------------------
    def reset_states(self, batch_size: int = 1):
        self._h = torch.zeros(batch_size, 128)
        self._c = torch.zeros(batch_size, 128)
        self._context = torch.zeros(batch_size, 64)   # 64-sample context

    # ---- forward (streaming-compatible) --------------------------------
    def forward(self, x: torch.Tensor, sr: int = 16000,
                state=None):
        """
        Parameters
        ----------
        x : (batch, 512) raw 16 kHz audio chunk  **or**  (batch, 576) with context prepended
        sr: ignored (kept for API compat)
        state : (2, batch, 128) packed [h, c]  – optional external state

        Returns
        -------
        prob  : (batch,)  speech probability  ∈ [0, 1]
        state : (2, batch, 128)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)                      # (1, T)

        B = x.shape[0]

        # --- context handling -------------------------------------------
        if x.shape[-1] == 512:
            if not hasattr(self, '_context') or self._context.shape[0] != B:
                self.reset_states(B)
            dev = x.device
            self._context = self._context.to(dev)
            x = torch.cat([self._context, x], dim=1)  # → (B, 576)
            self._context = x[:, -64:]                  # store last 64 for next call

        # --- unpack / init state ----------------------------------------
        if state is not None:
            h, c = state[0], state[1]
        elif hasattr(self, '_h'):
            h, c = self._h.to(x.device), self._c.to(x.device)
        else:
            h = torch.zeros(B, 128, device=x.device)
            c = torch.zeros(B, 128, device=x.device)

        # --- Learnt STFT -----------------------------------------------
        x = self.stft_pad(x)                           # (B, 576+64=640)
        x = x.unsqueeze(1)                             # (B, 1, 640)
        x = self.stft_conv(x)                          # (B, 258, T') stride=128
        # split into real / imag and take magnitude
        cutoff = 129  # filter_length // 2 + 1
        real = x[:, :cutoff, :]
        imag = x[:, cutoff:, :]
        x = torch.sqrt(real.pow(2) + imag.pow(2))     # (B, 129, T')

        # --- Encoder convolutions (+ ReLU) -----------------------------
        x = F.relu(self.conv1(x))                      # (B, 128, T')
        x = F.relu(self.conv2(x))                      # (B, 64, T')
        x = F.relu(self.conv3(x))                      # (B, 64, T')
        x = F.relu(self.conv4(x))                      # (B, 128, T')

        # Squeeze last dim (encoder output → decoder input)
        x = x.squeeze(-1)                              # (B, 128) if T'==1, else pool

        # If T' > 1, we need global pool to get (B, 128)
        if x.dim() == 3:
            x = x.mean(dim=-1)

        # --- LSTM cell --------------------------------------------------
        h, c = self.lstm_cell(x, (h, c))

        # --- Decoder head -----------------------------------------------
        x = h.unsqueeze(-1).float()                     # (B, 128, 1)
        x = self.decoder_dropout(x)                     # Dropout(0.1)
        x = F.relu(x)                                  # ReLU
        x = self.final_conv(x)                          # (B, 1, 1)
        x = torch.sigmoid(x)                            # Sigmoid

        # Output: squeeze(1) → mean(dim=1) → scalar
        prob = x.squeeze(1).mean(dim=-1)                # (B,)

        # pack state
        new_state = torch.stack([h, c])                 # (2, B, 128)
        self._h, self._c = h.detach(), c.detach()
        return prob, new_state


# ---------------------------------------------------------------------------
# 2.  Load pre-trained weights
# ---------------------------------------------------------------------------
def load_silero_vad_pt(source: str = "silero-vad/src/silero_vad/data/silero_vad.jit",
                       device: str = "cpu") -> SileroVAD16k:
    """Load weights into reconstructed SileroVAD16k.

    Supports two sources:
      - .jit file  → extracts weights from the JIT _model (16k sub-model)
      - .safetensors file → loads directly (keys must match)
    """
    model = SileroVAD16k()

    if source.endswith(".jit"):
        jit = torch.jit.load(source, map_location=device)
        m = jit._model
        state = {}
        state['stft_conv.weight'] = m.stft.forward_basis_buffer.data
        for i, name in enumerate(['conv1', 'conv2', 'conv3', 'conv4']):
            layer = getattr(m.encoder, str(i)).reparam_conv
            state[f'{name}.weight'] = layer.weight.data
            state[f'{name}.bias'] = layer.bias.data
        state['lstm_cell.weight_ih'] = m.decoder.rnn.weight_ih.data
        state['lstm_cell.weight_hh'] = m.decoder.rnn.weight_hh.data
        state['lstm_cell.bias_ih'] = m.decoder.rnn.bias_ih.data
        state['lstm_cell.bias_hh'] = m.decoder.rnn.bias_hh.data
        fc = getattr(m.decoder.decoder, '2')
        state['final_conv.weight'] = fc.weight.data
        state['final_conv.bias'] = fc.bias.data
        model.load_state_dict(state, strict=False)
    elif source.endswith(".safetensors"):
        load_model(model, source)
    else:
        raise ValueError(f"Unsupported source format: {source}")

    model.to(device)
    model.eval()
    model.reset_states()
    return model


# ---------------------------------------------------------------------------
# 3.  Wiener Filter speech enhancement (single-channel / mono)
# ---------------------------------------------------------------------------
class WienerFilter(nn.Module):
    """
    Spectral Wiener-like filter.
    Given a noisy STFT and a per-frame speech-presence mask (from VAD),
    estimates the clean speech STFT.

    H(f) = |S(f)|² / (|S(f)|² + |N(f)|²)

    Noise PSD is estimated from non-speech frames; speech+noise PSD from
    speech frames.  A learnable over-subtraction `alpha` and spectral
    floor `beta` are added for fine-tuning flexibility.
    """

    def __init__(self, n_fft: int = 512, hop_length: int = 256):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        # Learnable parameters for fine-tuning the filter
        self.alpha = nn.Parameter(torch.tensor(1.0))   # over-subtraction
        self.beta = nn.Parameter(torch.tensor(0.02))   # spectral floor

    def forward(self, noisy_wav: torch.Tensor,
                vad_mask: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        noisy_wav : (B, T)  time-domain noisy signal
        vad_mask  : (B, F)  per-frame binary/soft speech mask  (0=noise, 1=speech)
                    F = number of STFT frames

        Returns
        -------
        clean_wav : (B, T)  enhanced time-domain signal
        """
        # STFT
        window = torch.hann_window(self.n_fft, device=noisy_wav.device)
        X = torch.stft(noisy_wav, self.n_fft, self.hop_length,
                        window=window, return_complex=True)  # (B, freq, frames)

        power = X.abs() ** 2                                # (B, F_bins, frames)

        # Align vad_mask length to STFT frames
        n_frames = power.shape[-1]
        if vad_mask.shape[-1] != n_frames:
            vad_mask = F.interpolate(vad_mask.unsqueeze(1).float(),
                                     size=n_frames, mode='nearest').squeeze(1)

        speech_mask = (vad_mask > 0.5).unsqueeze(1)          # (B, 1, frames)
        noise_mask = ~speech_mask

        # Estimate noise PSD from non-speech frames
        noise_count = noise_mask.float().sum(dim=-1, keepdim=True).clamp(min=1)
        noise_psd = (power * noise_mask.float()).sum(dim=-1, keepdim=True) / noise_count  # (B, F_bins, 1)

        # Wiener gain
        gain = torch.clamp(1.0 - self.alpha * noise_psd / (power + 1e-8),
                           min=self.beta.abs())               # (B, F_bins, frames)

        # Apply
        Y = X * gain
        clean = torch.istft(Y, self.n_fft, self.hop_length,
                            window=window, length=noisy_wav.shape[-1])
        return clean


# ---------------------------------------------------------------------------
# 4.  Combined VAD + Wiener Enhancement pipeline
# ---------------------------------------------------------------------------
class VADWienerPipeline(nn.Module):
    """End-to-end: noisy speech → VAD probabilities → Wiener filter → clean speech."""

    def __init__(self, vad: SileroVAD16k, wiener: WienerFilter,
                 sr: int = 16000, chunk_samples: int = 512):
        super().__init__()
        self.vad = vad
        self.wiener = wiener
        self.sr = sr
        self.chunk_samples = chunk_samples

    @torch.no_grad()
    def get_vad_probs(self, wav: torch.Tensor) -> torch.Tensor:
        """Run VAD in streaming mode over full waveform → per-chunk probabilities."""
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        B, T = wav.shape
        self.vad.reset_states(B)
        probs = []
        for start in range(0, T, self.chunk_samples):
            chunk = wav[:, start:start + self.chunk_samples]
            if chunk.shape[-1] < self.chunk_samples:
                chunk = F.pad(chunk, (0, self.chunk_samples - chunk.shape[-1]))
            p, _ = self.vad(chunk, self.sr)
            probs.append(p)
        return torch.stack(probs, dim=-1)              # (B, n_chunks)

    def forward(self, noisy_wav: torch.Tensor) -> tuple:
        """
        Returns (clean_wav, vad_probs)
        """
        vad_probs = self.get_vad_probs(noisy_wav)
        if noisy_wav.dim() == 1:
            noisy_wav = noisy_wav.unsqueeze(0)
        clean_wav = self.wiener(noisy_wav, vad_probs)
        return clean_wav, vad_probs


# ---------------------------------------------------------------------------
# 5.  Fine-tuning & QAT utilities
# ---------------------------------------------------------------------------
def setup_finetune(safetensors_path: str = "silero-vad/src/silero_vad/data/silero_vad_16k.safetensors",
                   lr: float = 1e-4, device: str = "cpu"):
    """Return (pipeline, optimizer, vad_criterion, enhancement_criterion)."""
    vad = load_silero_vad_pt(safetensors_path, device)
    wiener = WienerFilter().to(device)
    pipe = VADWienerPipeline(vad, wiener).to(device)
    pipe.train()

    optimizer = torch.optim.Adam([
        {"params": vad.parameters(), "lr": lr},
        {"params": wiener.parameters(), "lr": lr * 10},  # wiener can train faster
    ])
    vad_criterion = nn.BCELoss()
    enh_criterion = nn.L1Loss()   # time-domain L1 for enhancement
    return pipe, optimizer, vad_criterion, enh_criterion


def apply_qat(model: SileroVAD16k, backend: str = "qnnpack"):
    """Prepare VAD model for Quantization-Aware Training and return the QAT-ready model."""
    from torch.ao.quantization import get_default_qat_qconfig, prepare_qat
    model.train()
    model.qconfig = get_default_qat_qconfig(backend)
    model_qat = prepare_qat(model)
    return model_qat


def convert_qat_to_int8(model_qat: nn.Module):
    from torch.ao.quantization import convert
    return convert(model_qat.eval(), inplace=False)


# ---------------------------------------------------------------------------
# 6.  ONNX export
# ---------------------------------------------------------------------------
def export_onnx(model: SileroVAD16k,
                output_path: str = "silero_vad_16k_finetuned.onnx",
                opset: int = 16):
    """Export the VAD model to ONNX (without Wiener – that runs in numpy/torch)."""
    model.eval()
    model.reset_states(1)
    dummy_input = torch.randn(1, 512)

    # We export a simplified wrapper that only returns prob
    class _Wrapper(nn.Module):
        def __init__(self, vad):
            super().__init__()
            self.vad = vad
        def forward(self, x):
            p, _ = self.vad(x, 16000)
            return p

    wrapper = _Wrapper(model)
    torch.onnx.export(
        wrapper, (dummy_input,),
        output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=opset,
    )
    print(f"✅ ONNX exported → {output_path}")


# ---------------------------------------------------------------------------
# 7.  Example usage / smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading Silero VAD 16k from safetensors …")
    vad = load_silero_vad_pt()
    print(f"  Parameters: {sum(p.numel() for p in vad.parameters()):,}")

    # Smoke test with random noise
    wav = torch.randn(1, 16000)  # 1 s of noise
    vad.reset_states(1)
    probs = []
    for i in range(0, 16000, 512):
        chunk = wav[:, i:i+512]
        if chunk.shape[-1] < 512:
            chunk = F.pad(chunk, (0, 512 - chunk.shape[-1]))
        p, _ = vad(chunk)
        probs.append(p.item())
    print(f"  VAD probs on noise (first 5 chunks): {probs[:5]}")

    # Full pipeline
    wiener = WienerFilter()
    pipe = VADWienerPipeline(vad, wiener)
    clean, vad_p = pipe(wav)
    print(f"  Enhanced output shape: {clean.shape}")

    # ONNX export
    export_onnx(vad, "silero_vad_16k_recon.onnx")
    print("Done.")
