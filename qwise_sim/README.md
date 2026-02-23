# **Q-WiSE: Quantized AI-Powered Multi-Channel Wiener Filter**

Q-WiSE is an ultra-low-consumption speech enhancement framework designed for the edge. This repository contains the **Month 1–2 Simulation Environment**, implemented in Rust, which serves as the algorithmic baseline for neural-guided Wiener filtering.

The project addresses **dAIEdge Challenge \#3**, targeting energy-efficient, real-time denoising for audio-based multimodal applications (e.g., drone surveillance, smart warehouses) with power budgets strictly under.

## Q-WiSE: Neural-Guided Wiener Filter Algorithm

The Q-WiSE framework implements a frequency-domain speech enhancement pipeline. Unlike traditional Wiener filters that 
rely on statistical SNR estimation, Q-WiSE uses a Neural-Guided approach where a quantized AI core (Mamba/Transformer) predicts 
the ideal gain maps.

The core objective is to estimate a clean speech signal $S(t)$ from a noisy observation $X(t) = S(t) + N(t)$. 
In the frequency domain, the Wiener filter provides an optimal gain $G(f)$ to minimize the mean square error:

$$Y(f) = X(f) \cdot G(f)$$

Where:
- $X(f)$ is the Short-Time Fourier Transform (STFT) of the noisy signal.
- $G(f)$ is the gain map predicted by the AI core, constrained to $0 \le G(f) \le 1$.
- $Y(f)$ is the estimated clean speech spectrum.

### Step 1: Framing and Windowing
To prevent spectral leakage at the frame boundaries, we apply a Hann Window to each $256$-sample frame ($N=256$):

$$w(n) = 0.5 \left( 1 - \cos \left( \frac{2\pi n}{N-1} \right) \right)$$

### Step 2: Short-Time Fourier Transform (STFT)
We convert the real-valued time-domain signal into the complex frequency domain using a Fast Fourier Transform.

$$X(k) = \sum_{n=0}^{N-1} x(n) w(n) e^{-j\frac{2\pi}{N}nk}$$

### Step 3: Neural Gain Application
This is the "Neural-Guided" innovation. The AI core (Mamba/Transformer) analyzes the noisy features and produces a real-valued 
mask $G(k)$. We perform point-wise complex multiplication:

$$\text{Re}\{Y(k)\} = \text{Re}\{X(k)\} \cdot G(k)$$

$$\text{Im}\{Y(k)\} = \text{Im}\{X(k)\} \cdot G(k)$$

This approach suppresses noise while preserving the original phase of the speech signal, which is critical for perceptual quality.

### Step 4: Inverse FFT via Conjugate Symmetry
Because the microfft crate is optimized for bare-metal and only provides a forward transform, 
we implement the Inverse FFT (IFFT) using the Conjugate-FFT-Conjugate method:

- Conjugate: $Z(k) = \text{conj}(Y(k))$
- Forward FFT: $z(n) = \text{FFT}(Z(k))$
- Final Conjugate & Scale: $y(n) = \frac{1}{N} \text{conj}(z(n))$

### Step 5: Synthesis and Normalization
The resulting real part of the complex buffer is extracted as the clean PCM audio. We normalize the floating-point samples 
back to the 16-bit integer range ($i16$) for output:

$$\text{sample}_{i16} = \text{clamp}(y(n) \cdot 32767, -32768, 32767)$$

### Step 6: Metadata Matrix Output

Simultaneously, the secondary head of the AI model outputs a $4 \times 4$ classification matrix. This represents the 
environmental context (e.g., Drone Rotor vs. Human Speech), enabling the system to provide multimodal awareness alongside audio enhancement.

## **📅 Roadmap (7-Month Plan)**

* **Months 1–2 (Current):** Concept design, baseline Rust simulation, and dataset benchmarking.
* **Months 3–4:** Model development (Transformer/Mamba hybrid), Quantization-Aware Training (QAT), and ONNX export.
* **Months 5–6:** Edge porting to ESP32-S3/STM32 MP1 and comprehensive power/latency profiling.
* **Month 7:** Final TRL-5 validation and integration into the dAIEDGE multimodal library.
