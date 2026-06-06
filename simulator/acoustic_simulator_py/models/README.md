# models/

Bundled inference assets.

* `qwise_vad.onnx` — Q-WiSE voice-activity-detection ONNX export, copied
  unchanged from the MATLAB project (`vad/qwise_vad.onnx`). Loaded by
  `backend.vad.qwise` via `onnxruntime`.

Files placed here are read by the backend via the path
`<repo>/models/<name>` (no environment variable needed for assets that
ship with the image).
