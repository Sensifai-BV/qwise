"""
recon.py – Reconstruct Silero VAD 16 kHz from safetensors, then provide:
  1. Fine-tuning / QAT helpers
  2. VAD-guided Multichannel Wiener Filter (MWF) speech enhancement
  3. ONNX export
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    Decision-directed Wiener filter with recursive noise estimation.

    Unlike simple spectral subtraction, this uses:
      1. Recursive noise PSD tracking — adapts to non-stationary noise
         (e.g. drone noise) by updating the noise estimate every frame,
         controlled by the VAD speech probability.
      2. Decision-directed a-priori SNR — smooths the SNR estimate across
         frames to suppress musical-noise artefacts.
      3. Proper Wiener gain  H = ξ / (ξ + 1)  where ξ = a-priori SNR.
      4. Gain floor to avoid complete nulling of low-energy bins.

    Learnable parameters (for optional fine-tuning):
        alpha_s   – noise tracking speed  (higher = slower adaptation)
        dd_alpha  – decision-directed smoothing  (higher = more temporal smoothing)
        gain_min  – spectral floor (dB)
    """

    def __init__(self, n_fft: int = 512, hop_length: int = 128,
                 alpha_s: float = 0.95, dd_alpha: float = 0.98,
                 gain_min_db: float = -25.0):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        # Learnable parameters
        self.alpha_s = nn.Parameter(torch.tensor(alpha_s))    # noise smoothing
        self.dd_alpha = nn.Parameter(torch.tensor(dd_alpha))  # decision-directed
        self.gain_min_db = nn.Parameter(torch.tensor(gain_min_db))  # spectral floor

    def forward(self, noisy_wav: torch.Tensor,
                vad_mask: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        noisy_wav : (B, T)  time-domain noisy signal
        vad_mask  : (B, F)  per-chunk VAD probability (0=noise, 1=speech)

        Returns
        -------
        clean_wav : (B, T)  enhanced time-domain signal
        """
        # STFT
        window = torch.hann_window(self.n_fft, device=noisy_wav.device)
        X = torch.stft(noisy_wav, self.n_fft, self.hop_length,
                        window=window, return_complex=True)   # (B, F, T)

        B, Freq, T = X.shape
        power = X.abs().pow(2)                                # (B, Freq, T)

        # --- Align VAD mask to STFT frames --------------------------
        if vad_mask.shape[-1] != T:
            vad_mask = F.interpolate(
                vad_mask.unsqueeze(1).float(), size=T, mode='linear',
                align_corners=False
            ).squeeze(1)                                      # (B, T)

        # Clamp learnable params to valid ranges
        alpha_s  = torch.clamp(self.alpha_s,  0.8, 0.99)
        dd_alpha = torch.clamp(self.dd_alpha, 0.8, 0.995)
        gain_min = 10.0 ** (torch.clamp(self.gain_min_db, -40.0, -6.0) / 20.0)

        # --- Recursive noise PSD estimation -------------------------
        # Initialise noise PSD from first few frames (assume start is noise)
        n_init = max(1, min(5, T))
        noise_psd = power[:, :, :n_init].mean(dim=-1)        # (B, Freq)

        # Pre-allocate gain
        gain = torch.zeros_like(power)                        # (B, Freq, T)
        prev_gain = torch.zeros(B, Freq, device=X.device)    # for DD smoothing

        for t in range(T):
            frame_power = power[:, :, t]                      # (B, F)
            speech_prob = vad_mask[:, t].unsqueeze(-1)        # (B, 1)

            # Update noise PSD: fast update in noise, slow in speech
            # α_adapt = α_s when speech, (1-speech_prob)*α_s when noise
            alpha_t = alpha_s * speech_prob + (1.0 - speech_prob) * 0.5
            noise_psd = alpha_t * noise_psd + (1.0 - alpha_t) * frame_power

            # A-posteriori SNR
            gamma = frame_power / (noise_psd + 1e-10)        # (B, F)

            # Decision-directed a-priori SNR
            xi_ml = torch.clamp(gamma - 1.0, min=0.0)
            # Smooth with previous frame's clean estimate
            xi = dd_alpha * prev_gain.pow(2) * gamma + (1.0 - dd_alpha) * xi_ml
            xi = torch.clamp(xi, min=1e-4)

            # Wiener gain:  H = ξ / (ξ + 1)
            G = xi / (xi + 1.0)
            G = torch.clamp(G, min=gain_min)

            gain[:, :, t] = G
            prev_gain = G

        # Apply gain and reconstruct
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
