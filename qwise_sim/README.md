# **Q-WiSE: Quantized AI-Powered Multi-Channel Wiener Filter**

Q-WiSE is an ultra-low-consumption speech enhancement framework designed for the edge. This repository contains the **Month 1–2 Simulation Environment**, implemented in Rust, which serves as the algorithmic baseline for neural-guided Wiener filtering.

The project addresses **dAIEdge Challenge \#3**, targeting energy-efficient, real-time denoising for audio-based multimodal applications (e.g., drone surveillance, smart warehouses) with power budgets strictly under.

## **🚀 Key Features**

* **Neural-Guided Wiener Filter:** Combines traditional signal processing interpretability with adaptive AI-driven gain parameterization.
* **Rust-Native DSP Pipeline:** A high-performance, no\_std compatible audio processing core utilizing microfft and num-complex.
* **Dual-Output Architecture:** Simultaneously produces enhanced audio and a classification metadata matrix for environment awareness.
* **Ultra-Low Latency:** Optimized for 256-sample frames to meet the requirements of real-time multi-modal edge systems.
* **Hardware Agnostic Simulation:** Validated on Linux (Fedora/Ubuntu) with a clear path to deployment on **ESP32-S3**, **Nordic nRF53**, and **STM32 MP1**.

## **🏗 System Architecture**

The Q-WiSE framework utilizes a hybrid approach between Deep Learning and Classical Signal Processing:

1. **Windowing:** A Hann window is applied to each sample frame to minimize spectral leakage:
2. **Forward FFT:** Converting time-domain PCM to complex frequency bins.
3. **Neural-Guided Gain Prediction:** A quantized AI core (Transformer/Mamba hybrid) predicts a time-frequency gain map.
4. **Filtering Stage:** The enhanced spectrum is calculated as:
5. **Inverse FFT:** Reconstructing audio via the Conjugate-FFT-Conjugate method for maximum efficiency on fixed-point hardware.
6. **Metadata Extraction:** The AI core's secondary head identifies noise types (e.g., "Drone Rotor", "Wind") and outputs a classification matrix.

## **🛠 Prerequisites**

* **Rust Toolchain:** Stable Rust (Edition 2021/2024).
* **System Libraries:** Standard development tools for cross-compilation if targeting ARM/Xtensa.
* **Input Audio:** Mono/Multi-channel .wav files.

## **📦 Installation & Setup**

1. **Initialize Project:**  
   git clone \<repository-url\>  
   cd qwise\_sim  
   mkdir \-p data/input data/output

2. **Add Dependencies:** Ensure your Cargo.toml includes hound, microfft, and num-complex.
3. **Build:**  
   cargo build \--release

## **💻 Usage & Benchmarking**

Run the simulation to process noisy audio datasets:

cargo run \-- data/input/noisy\_sample.wav data/output/clean\_sample.wav

### **Performance Metrics**

The system is evaluated against the following benchmarks:

* **PESQ:** Perceptual Evaluation of Speech Quality.
* **STOI:** Short-Time Objective Intelligibility.
* **SNR Improvement:** Signal-to-Noise Ratio gain.
* **Power Profiling:** Target using the Nordic Power Profiler Kit II.

## **📂 Project Structure**

* src/main.rs: Core Wiener filter logic, FFT pipeline, and CLI entry point.
* data/: Directory for input/output datasets (LibriSpeech/VoxCeleb).
* Cargo.toml: Project manifest.
* models/: (Planned) Storage for Quantized INT8 ONNX models.

## **📅 Roadmap (7-Month Plan)**

* **Months 1–2 (Current):** Concept design, baseline Rust simulation, and dataset benchmarking.
* **Months 3–4:** Model development (Transformer/Mamba hybrid), Quantization-Aware Training (QAT), and ONNX export.
* **Months 5–6:** Edge porting to ESP32-S3/STM32 MP1 and comprehensive power/latency profiling.
* **Month 7:** Final TRL-5 validation and integration into the dAIEDGE multimodal library.
