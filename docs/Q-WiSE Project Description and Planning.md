**Q-WiSE** introduces a novel **Quantized AI-Powered Multi-Channel Wiener Filter** framework for **ultra-low-consumption speech enhancement at the edge**, directly addressing dAIEdge Challenge \#3. The project targets **energy-efficient, real-time denoising** for audio-based multimodal edge applications such as drone-mounted surveillance or smart warehouse monitoring, where speech clarity is crucial but power budgets are extremely constrained (\< 50 mW).

Building on state-of-the-art **deep Wiener filtering** and **Transformer/Mamba hybrid denoisers**, Q-WiSE will develop a **quantized, multi-channel, ONNX-compliant speech enhancement model** optimized for micro-edge hardware (e.g., STM32 MP1, ESP32-S3, or Nordic nRF53). The system will combine the interpretability and low-latency properties of traditional Wiener filters with the adaptive learning capabilities of deep networks. The innovation lies in a **neural-guided filter parameterization**, where a lightweight quantized AI core predicts time-frequency gain maps with dynamic sparsity and multi-head temporal attention, achieving superior enhancement quality at an order-of-magnitude lower energy cost than LSTM baselines.

Q-WiSE has **two-stage implementation**:

1. **Algorithmic design and quantization-aware training** using noisy datasets (LibriSpeech, VoxCeleb \+ synthetic noise), benchmarking against DCCRN and TinyDenoiser.

2. **Embedded deployment and evaluation** on a selected \< 50 mW hardware target integrated with the **dAIEDGE Virtual Lab**, enabling automated benchmarking through PESQ/SNR and hardware metrics (inference time, power draw).

Q-WiSE starts at TRL3 to TRL5 with a functional ONNX model, energy-profiling scripts, and reproducible open-source documentation for inclusion in the **dAIEDGE multimodal algorithm library**. Q-WiSE is rooted Sensifai’s **Edge AI and Audio Intelligence capabilities**, proven through its **NGI TRUST–funded work** on privacy-preserving on-device enhancement. Dr. Bahari’s influential research in **signal enhancement and distributed acoustic processing**, Q-WiSE leverages advanced concepts from his IEEE and EUSIPCO publications to develop **adaptive, cooperative, and low-power multi-channel speech enhancement** optimized for edge deployment.

The expected outcome is a validated **low-power, quantized AI speech enhancement module** ready for integration into multimodal drone and smart-edge use cases, demonstrating up to **30 % energy savings** over current LSTM-based solutions while maintaining high perceptual quality.

**Planning**

The project will follow a **7-month structured research and implementation plan** designed to ensure smooth progression from concept to validated prototype:

**Month 1–2: Concept Design and Baseline Analysis**

* Define hardware and software specifications.

* Develop initial deep Wiener-filter model structure and simulation setup.

* Benchmark existing models on noisy speech datasets.

**Month 3–4: Model Development and Training**

* Implement hybrid neural-guided Wiener filter.

* Perform quantization-aware training (QAT) and sparsity optimization.

* Validate on LibriSpeech and VoxCeleb datasets (PESQ, STOI, SNR).

**Month 5–6: Edge Optimization and Integration**

* Port model to selected edge device (e.g., STM32 MP1).

* Measure power consumption, inference latency, and quality metrics.

* Integrate model with dAIEDGE Virtual Lab (ONNX format).

**Month 7: Validation and Documentation**

* Conduct final testing and cross-platform verification.

* Prepare final deliverables: prototype report, reproducible scripts, and TRL validation summary.

This planning ensures a **logical TRL progression (3 → 5\)** with continuous validation and integration, guaranteeing that Q-WiSE meets dAIEDGE’s objectives for low-power, multimodal edge AI innovation.

