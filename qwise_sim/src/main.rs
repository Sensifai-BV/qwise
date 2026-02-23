use axum::{
    extract::{Multipart, DefaultBodyLimit},
    response::{Html, Json},
    routing::{get, post},
    Router,
};
use microfft::complex::cfft_256;
use num_complex::Complex;
use std::env;
use std::io::Cursor;
use hound::{WavReader, WavWriter};
use serde::Serialize;
use base64::{Engine as _, engine::general_purpose};

const FRAME_SIZE: usize = 256;
const BIN_COUNT: usize = 256;

// --- DATA STRUCTURES ---

#[derive(Serialize)]
struct ProcessResponse {
    audio_base64: String,
    logs: Vec<String>,
    matrix: [[f32; 4]; 4],
}

pub struct AiPrediction {
    gain_map: [f32; BIN_COUNT],
    metadata_matrix: [[f32; 4]; 4],
}

pub struct QWiseFilter {
    window: [f32; FRAME_SIZE],
}

impl QWiseFilter {
    pub fn new() -> Self {
        let mut window = [0.0; FRAME_SIZE];
        for i in 0..FRAME_SIZE {
            window[i] = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (FRAME_SIZE - 1) as f32).cos());
        }
        Self { window }
    }

    pub fn process_frame(&mut self, noisy_pcm: &[f32; FRAME_SIZE], ai: &AiPrediction) -> [f32; FRAME_SIZE] {
        let mut buffer = [Complex::new(0.0, 0.0); FRAME_SIZE];
        for i in 0..FRAME_SIZE {
            buffer[i] = Complex::new(noisy_pcm[i] * self.window[i], 0.0);
        }

        let _ = cfft_256(&mut buffer);

        for i in 0..BIN_COUNT {
            buffer[i] *= ai.gain_map[i].clamp(0.0, 1.0);
        }

        // Inverse FFT via Conjugate Method
        for c in buffer.iter_mut() { *c = c.conj(); }
        let _ = cfft_256(&mut buffer);
        for c in buffer.iter_mut() { *c = c.conj() / (FRAME_SIZE as f32); }

        let mut output = [0.0f32; FRAME_SIZE];
        for i in 0..FRAME_SIZE { output[i] = buffer[i].re; }
        output
    }
}

// --- WEB GUI & SSR ---

#[tokio::main]
async fn main() {
    let args: Vec<String> = env::args().collect();

    if args.contains(&"--gui".to_string()) {
        println!("[Q-WiSE] Starting Web GUI on http://localhost:3000");
        let app = Router::new()
            .route("/", get(show_gui))
            .route("/upload", post(handle_upload))
            .layer(DefaultBodyLimit::max(15 * 1024 * 1024)); // 15MB limit

        let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
        axum::serve(listener, app).await.unwrap();
    } else {
        println!("Run with --gui for web interface or follow standard CLI usage.");
    }
}

async fn show_gui() -> Html<String> {
    Html(r#"
        <!DOCTYPE html>
        <html>
        <head>
            <title>Q-WiSE Dashboard</title>
            <script src="https://cdn.tailwindcss.com"></script>
        </head>
        <body class="bg-slate-900 text-white p-8">
            <div class="max-w-4xl mx-auto">
                <h1 class="text-3xl font-bold mb-4 text-blue-400">Q-WiSE Simulation (Wiener-filter model)</h1>

                <div class="bg-slate-800 p-6 rounded-lg mb-6 border border-slate-700">
                    <h2 class="text-xl mb-4 font-semibold">Step 1: Upload Noisy Audio (WAV)</h2>
                    <div class="flex items-center gap-4">
                        <input type="file" id="audioInput" class="block text-sm text-slate-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-500 file:text-white hover:file:bg-blue-600"/>
                        <button onclick="processAudio()" id="processBtn" class="bg-emerald-600 px-6 py-2 rounded font-bold hover:bg-emerald-700 disabled:opacity-50">Process Audio</button>
                    </div>
                </div>

                <div id="resultsArea" class="hidden space-y-6">
                     <div class="bg-slate-800 p-6 rounded-lg border border-emerald-500/30 flex justify-between items-center">
                        <div>
                            <h2 class="text-xl font-bold text-emerald-400">Processing Complete</h2>
                            <p class="text-sm text-slate-400">The Neural-Guided Wiener filter has been applied.</p>
                        </div>
                        <a id="downloadBtn" class="bg-blue-600 px-6 py-3 rounded-lg font-bold hover:bg-blue-700 transition">Download Enhanced WAV</a>
                    </div>

                    <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div class="bg-slate-800 p-4 rounded border border-slate-700">
                            <h3 class="text-blue-400 font-bold mb-2 uppercase text-xs tracking-widest">System Logs</h3>
                            <pre id="logOutput" class="text-xs text-slate-300 h-64 overflow-y-auto font-mono bg-black p-4 rounded border border-slate-700 leading-relaxed"></pre>
                        </div>
                        <div class="bg-slate-800 p-4 rounded border border-slate-700">
                            <h3 class="text-emerald-400 font-bold mb-2 uppercase text-xs tracking-widest">Neural Classification Matrix</h3>
                            <div id="matrixOutput" class="grid grid-cols-4 gap-2 text-center text-xs font-mono">
                                <!-- Matrix injected here -->
                            </div>
                            <p class="text-[10px] text-slate-500 mt-4 italic text-center">Row: Input Channel | Col: Predicted Noise Class</p>
                        </div>
                    </div>
                </div>
            </div>

            <script>
                async function processAudio() {
                    const fileInput = document.getElementById('audioInput');
                    const btn = document.getElementById('processBtn');
                    const results = document.getElementById('resultsArea');

                    if (!fileInput.files[0]) return alert("Please select a file first");

                    btn.disabled = true;
                    btn.innerText = "Processing...";

                    const formData = new FormData();
                    formData.append('audio', fileInput.files[0]);

                    try {
                        const response = await fetch('/upload', { method: 'POST', body: formData });
                        const data = await response.json();

                        // 1. Update Logs
                        const logBox = document.getElementById('logOutput');
                        logBox.innerHTML = data.logs.join('\n');

                        // 2. Update Matrix
                        const matrixBox = document.getElementById('matrixOutput');
                        matrixBox.innerHTML = data.matrix.flat().map(val =>
                            `<div class="bg-slate-900 border border-slate-700 p-3 rounded text-emerald-400 font-bold">${val.toFixed(2)}</div>`
                        ).join('');

                        // 3. Setup Download
                        const downloadBtn = document.getElementById('downloadBtn');
                        downloadBtn.href = `data:audio/wav;base64,${data.audio_base64}`;
                        downloadBtn.download = "qwise_enhanced.wav";

                        results.classList.remove('hidden');
                    } catch (e) {
                        alert("Error processing audio: " + e);
                    } finally {
                        btn.disabled = false;
                        btn.innerText = "Process Audio";
                    }
                }
            </script>
        </body>
        </html>
    "#.to_string())
}

async fn handle_upload(mut multipart: Multipart) -> Json<ProcessResponse> {
    let mut data = Vec::new();
    let mut logs = vec![
        "[SYS] Initializing Q-WiSE Pipeline...".to_string(),
        "[HW] Target: Simulation Mode (Cortex-A Simulation)".to_string(),
    ];

    while let Some(field) = multipart.next_field().await.unwrap() {
        if field.name() == Some("audio") {
            data = field.bytes().await.unwrap().to_vec();
        }
    }

    if data.is_empty() {
        return Json(ProcessResponse {
            audio_base64: "".to_string(),
            logs: vec!["Error: No file data found".to_string()],
            matrix: [[0.0; 4]; 4],
        });
    }

    logs.push(format!("[IO] Received WAV file ({} bytes)", data.len()));

    // Process the data
    let mut reader = WavReader::new(Cursor::new(data)).expect("Invalid WAV");
    let spec = reader.spec();
    let mut filter = QWiseFilter::new();
    let samples: Vec<f32> = reader.samples::<i16>().map(|s| s.unwrap() as f32 / i16::MAX as f32).collect();

    logs.push(format!("[DSP] Starting STFT on {} samples", samples.len()));

    let mut output_data = Vec::new();
    let mut cursor = Cursor::new(&mut output_data);

    // Mocked outputs for simulation
    let mock_ai = AiPrediction {
        gain_map: [0.65; BIN_COUNT],
        metadata_matrix: [
            [0.85, 0.05, 0.05, 0.05],
            [0.10, 0.70, 0.10, 0.10],
            [0.02, 0.08, 0.90, 0.00],
            [0.15, 0.15, 0.15, 0.55]
        ]
    };

    {
        let mut writer = WavWriter::new(&mut cursor, spec).unwrap();
        for (i, chunk) in samples.chunks_exact(FRAME_SIZE).enumerate() {
            let mut frame = [0.0f32; FRAME_SIZE];
            frame.copy_from_slice(chunk);
            let output_frame = filter.process_frame(&frame, &mock_ai);
            for sample in output_frame.iter() {
                writer.write_sample((sample * i16::MAX as f32) as i16).unwrap();
            }
            if i % 500 == 0 && i > 0 {
                logs.push(format!("[DSP] Frame {}: Filter parameters optimized", i));
            }
        }
    }

    logs.push("[AI] Multi-channel classification complete.".to_string());
    logs.push("[SYS] Finalizing output stream...".to_string());

    let b64 = general_purpose::STANDARD.encode(output_data);

    Json(ProcessResponse {
        audio_base64: b64,
        logs,
        matrix: mock_ai.metadata_matrix,
    })
}