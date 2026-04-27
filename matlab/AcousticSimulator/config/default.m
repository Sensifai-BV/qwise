function cfg = default()
%DEFAULT  Q-WiSE Acoustic Simulator default configuration.
%
%   cfg = default() returns a struct with every runtime parameter used by
%   the simulator.  Edit this file to reshape the simulator — everything
%   from sample rate and mic count down to VAD thresholds and UI sizes is
%   driven from here.

    % ---------------- Audio / framing --------------------------------
    cfg.fs              = 16000;         % sample rate [Hz]
    cfg.frame_size      = 1024;          % block size (samples)
    cfg.loop_sec        = 120;           % pre-loaded noise loop length [s]
    cfg.c               = 343;           % speed of sound [m/s]

    % ---------------- Microphone + virtual array ---------------------
    cfg.n_mics          = 3;             % virtual uniform linear array
    cfg.mic_spacing     = 0.10;          % m

    % ---------------- Scene geometry ---------------------------------
    cfg.human_height    = 1.70;          % m
    cfg.mouth_height    = 0.88 * cfg.human_height;
    cfg.slant_dist      = 2.50;          % m, speaker-to-drone slant
    cfg.elev_deg        = 30;            % deg, elevation of drone
    cfg.drone_rpm       = 8000;
    cfg.drone_blades    = 3;
    cfg.ground_R        = 0.90;          % asphalt reflection coeff
    cfg.alpha_air_dB    = 0.004;         % dB/m/kHz air absorption
    cfg.distance_ref    = 1.0;           % m  — 1/d law reference; gains
                                         %       are clamped so d<d_ref
                                         %       gives unity (no boost).

    % ---------------- Noise sources ----------------------------------
    cfg.drone_wav_path  = fullfile('wavs','drone_fan.wav');
    cfg.env_wav_path    = fullfile('wavs','env_ambient.wav');
    cfg.speech_gain_init= 1.00;          % input-mic speech level at mic-1
    cfg.drone_gain_init = 0.03;
    cfg.env_gain_init   = 0.01;

    % ---------------- Array wiring -----------------------------------
    %   'perChannel' : mic-1 = input speech + drone + env  (realistic
    %                  noisy speech the input mic actually picks up)
    %                  mic-2 = drone wav   (noise-only reference)
    %                  mic-3 = env wav     (noise-only reference)
    %                  composite = mic-1 (already the full noisy mix)
    %                  feeds the VAD and listening playback.
    %   'physical'   : every mic picks up every source with per-source
    %                  TDOA + 1/d spreading gain (research-grade acoustic
    %                  model — keeps the SourceMixer tests green).
    cfg.mixer.mode      = 'perChannel';
    cfg.mixer.composite = 'mic1';        % 'mic1' | 'sum' | 'mean'

    % ---------------- VAD --------------------------------------------
    cfg.vad.backend           = 'qwise';     % 'auto' | 'qwise' | 'energy'
    cfg.vad.onnx_path         = fullfile('vad','qwise_vad.onnx');
    cfg.vad.qwise_frame       = 512;         % 32 ms @ 16 kHz — fixed by the VAD ONNX
    cfg.vad.qwise_threshold   = 0.50;
    cfg.vad.energy_threshold  = -45;         % dBFS
    cfg.vad.sfm_threshold     = 0.45;        % spectral flatness
    cfg.vad.hang_frames       = 8;           % release hangover
    cfg.vad.smoothing         = 0.30;        % score EMA

    % ---------------- MWF --------------------------------------------
    cfg.mwf.enabled           = true;
    cfg.mwf.stft_win          = 512;
    cfg.mwf.stft_hop          = 256;
    cfg.mwf.ref_mic           = 1;
    cfg.mwf.mu                = 1.0;         % SDW speech-distortion weight
    cfg.mwf.eps_reg           = 1e-4;        % diagonal loading
    cfg.mwf.alpha_nn          = 0.92;        % Rnn EMA (noise-only frames)
    cfg.mwf.alpha_ss          = 0.88;        % Rss EMA (speech frames)
    cfg.mwf.passthrough       = true;        % <-- current stub behaviour

    % ---------------- Recording --------------------------------------
    cfg.record.dir            = 'recordings';
    cfg.record.prefix         = 'qwise';
    % 'composite' : sum of all mic channels (input + drone + env)
    % 'raw_mic'   : only the live input microphone signal
    % 'speech'    : the speech channel (mic-1 post-gain)
    cfg.record.source         = 'composite';

    % ---------------- Visualization ----------------------------------
    cfg.ui.spec_ncols         = 90;
    cfg.ui.vad_hist_sec       = 8;           % VAD trace horizon
    cfg.ui.waveform_span      = cfg.frame_size;
    cfg.ui.fig_position       = [50 40 1560 900];
end
