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

    % ---------------- Microphone + virtual array ---------
    cfg.mic_model       = 'MacBook Pro Microphone';  % search substring
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

    % ---------------- Noise sources ----------------------------------
    cfg.drone_wav_path  = fullfile('wavs','drone_fan.wav');
    cfg.env_wav_path    = fullfile('wavs','env_ambient.wav');
    cfg.drone_gain_init = 0.40;
    cfg.env_gain_init   = 0.25;

    % ---------------- VAD --------------------------------------------
    cfg.vad.backend           = 'silero';      % 'auto' | 'silero' | 'energy'
    cfg.vad.onnx_path         = fullfile('vad','silero_vad.onnx');
    cfg.vad.silero_frame      = 512;         % 32 ms @ 16 kHz (Silero spec)
    cfg.vad.silero_threshold  = 0.50;
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

    % ---------------- Playback ---------------------------------------
    cfg.playback.enabled      = true;
    cfg.playback.source       = 'noisy';     % 'noisy' | 'enhanced'

    % ---------------- Visualization ----------------------------------
    cfg.ui.spec_ncols         = 90;
    cfg.ui.vad_hist_sec       = 8;           % VAD trace horizon
    cfg.ui.waveform_span      = cfg.frame_size;
    cfg.ui.fig_position       = [50 40 1560 900];
end
