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
    cfg.n_mics          = 3;             % virtual array size (>=1)
    cfg.mic_spacing     = 0.10;          % m
    cfg.mic_geometry    = 'linear';      % 'linear' | 'circular'
                                         %   array is centred on the drone

    % ---------------- Scene geometry ---------------------------------
    cfg.human_height    = 1.70;          % m
    cfg.mouth_height    = 0.88 * cfg.human_height;
    cfg.slant_dist      = 2.50;          % m, speaker-to-drone slant
    cfg.elev_deg        = 30;            % deg, elevation of drone

    % Drone source (relative to human mouth).
    %   slant_dist + elev_deg + drone.azimuth_deg describe where the
    %   drone sits in the simulation; the mic array is centred on
    %   pos_drone in build_geometry.m.
    cfg.drone.azimuth_deg   = 0;         % deg, 0 = drone along +x from mouth

    % Environment noise source (relative to human mouth).
    %   Treated as a single point source, independent of the drone, so the
    %   physical mixer can give it its own TDOA + 1/r gain on every mic.
    cfg.env.distance_from_mouth = 8.0;   % m
    cfg.env.azimuth_deg         = 135;   % deg
    cfg.env.elevation_deg       = 0;     % deg

    cfg.drone_rpm       = 8000;
    cfg.drone_blades    = 3;
    cfg.ground_R        = 0.90;          % asphalt reflection coeff (scene viz only)
    cfg.alpha_air_dB    = 0.004;         % dB/m/kHz air absorption  (scene viz only)
    cfg.distance_ref    = 1.0;           % m  — 1/r law reference; gains
                                         %       are clamped so d<d_ref
                                         %       gives unity (no boost).

    % ---------------- Noise sources ----------------------------------
    cfg.drone_wav_path  = fullfile('wavs','drone_fan.wav');
    cfg.env_wav_path    = fullfile('wavs','env_ambient.wav');
    cfg.speech_gain_init= 1.00;          % live-mic speech level pre-mixer
    cfg.drone_gain_init = 0.03;
    cfg.env_gain_init   = 0.01;

    % ---------------- Array wiring -----------------------------------
    %   The SourceMixer is a single physical model: every mic receives
    %   every source with fractional delay + 1/r gain (mixer_1.m-style).
    %   `composite` controls how the N-channel block is reduced to one
    %   mono signal for the VAD ('mic1' = reference mic, 'sum', 'mean').
    cfg.mixer.mode      = 'physical';    % only valid value (legacy field)
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
    %   Beamformer method   : 'gev'  — Generalized-Eigenvalue / Max-SNR
    %                                   (recommended for non-stationary
    %                                   drone-rotor noise — matches Q-WiSE
    %                                   reference Python pipeline)
    %                         'mwf'  — Speech-Distortion-Weighted MWF
    %                         'mvdr' — MVDR with eigenvector steering
    %   Postfilter          : single-channel Wiener post-filter after
    %                                   beamforming (decision-directed,
    %                                   frequency-smoothed)
    %   Mask params (batch) : threshold + context frames driving the
    %                                   speech mask built from VAD audio
    cfg.mwf.enabled           = true;
    cfg.mwf.method            = 'gev';       % 'gev' | 'mwf' | 'mvdr'
    cfg.mwf.n_fft             = 1024;        % batch-mode STFT size  (matches Python)
    cfg.mwf.hop               = 256;         % batch-mode STFT hop   (matches Python)
    cfg.mwf.stft_win          = 512;         % streaming STFT window
    cfg.mwf.stft_hop          = 256;         % streaming STFT hop
    cfg.mwf.ref_mic           = 1;
    cfg.mwf.mu                = 1.0;         % SDW speech-distortion weight
    cfg.mwf.eps_reg           = 1e-10;       % core regularization (matches Python)
    cfg.mwf.diag_load_ratio   = 1e-4;        % trace-proportional diagonal loading
    cfg.mwf.alpha_nn          = 0.92;        % Rnn EMA (streaming, noise-only frames)
    cfg.mwf.alpha_ss          = 0.88;        % Rss EMA (streaming, speech frames)
    cfg.mwf.postfilter        = true;        % apply Wiener post-filter
    cfg.mwf.gain_floor        = 0.08;        % post-filter floor (anti musical noise)
    cfg.mwf.noise_floor_alpha = 0.98;        % post-filter noise tracker EMA
    cfg.mwf.pf_smooth_kernel  = 3;           % freq smoothing kernel size
    cfg.mwf.mask_threshold    = 0.01;        % batch speech-mask RMS threshold
    cfg.mwf.mask_context      = 3;           % batch speech-mask context frames
    cfg.mwf.passthrough       = false;       % set true to bypass MWF (debug only)

    % ---------------- Recording --------------------------------------
    %   Mode is locked in at the moment the Record button is pressed:
    %     vad_on=F, mwf_on=F → 'multi'  — one WAV per mic, written into
    %                                      a timestamped sub-folder
    %     vad_on=T, mwf_on=F → 'mono'   — composite (ref-mic) only during
    %                                      VAD-detected speech
    %     vad_on=T, mwf_on=T → 'mono'   — MWF output only during
    %                                      VAD-detected speech
    cfg.record.dir            = 'recordings';
    cfg.record.prefix         = 'qwise';
    cfg.record.multi_subdir   = 'multi';     % name of subfolder for multi-mic captures

    % ---------------- Visualization ----------------------------------
    cfg.ui.spec_ncols         = 90;
    cfg.ui.vad_hist_sec       = 8;           % VAD trace horizon
    cfg.ui.waveform_span      = cfg.frame_size;
    cfg.ui.fig_position       = [50 40 1560 900];
end
