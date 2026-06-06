# wavs/

Looped noise sources shared with the MATLAB project.

* `drone_fan.wav`    — rotor / fan loop fed into `SourceMixer` as the
  drone source.
* `env_ambient.wav`  — environmental noise loop.

`backend.audio` reads these once at startup and feeds chunks via
`loop_chunk` (ported from `core/loop_chunk.m`).

User-supplied clean-speech WAVs (Task 16) are uploaded at runtime and
land under `/data/uploads/` inside the container; they never get
committed to the repo.
