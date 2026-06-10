function cfg = default()
%DEFAULT  Q-WiSE Acoustic Simulator configuration.
%
%   cfg = default() returns a struct with every runtime parameter used by
%   the file-based simulator: a clean-speech sample is mixed with drone-fan
%   and environment noise through a physical N-mic array model, then the
%   noisy array is denoised by the fp32 ONNX model (onnx/qwise.onnx).

    % ---- Resolve project paths (independent of the current folder) ----
    here = fileparts(mfilename('fullpath'));   % .../lab/config
    lab  = fileparts(here);                     % .../lab
    proj = fileparts(lab);                      % .../qwise (project root)
    cfg.lab_root  = lab;
    cfg.proj_root = proj;

    % ---------------- Audio / framing --------------------------------
    cfg.fs              = 16000;         % sample rate [Hz] (ONNX requirement)
    cfg.frame_size      = 1024;          % display/animation block size [samples]
    cfg.loop_sec        = 120;           % pre-loaded noise loop length [s]
    cfg.c               = 343;           % speed of sound [m/s]

    % ---------------- Microphone array -------------------------------
    cfg.n_mics          = 3;             % array size (must match ONNX export)
    cfg.mic_spacing     = 0.10;          % m
    cfg.mic_geometry    = 'linear';      % 'linear' | 'circular'
                                         %   array is centred on the drone

    % ---------------- Scene geometry ---------------------------------
    cfg.human_height    = 1.70;          % m
    cfg.mouth_height    = 0.88 * cfg.human_height;
    cfg.slant_dist      = 1.00;          % m, speaker-to-drone slant distance
    cfg.elev_deg        = 30;            % deg, elevation of the drone
    cfg.drone.azimuth_deg       = 0;     % deg, 0 = drone straight ahead (center)

    % Environment noise: independent point source relative to the mouth.
    cfg.env.distance_from_mouth = 8.0;   % m
    cfg.env.azimuth_deg         = 135;   % deg
    cfg.env.elevation_deg       = 0;     % deg

    % Scene-visualisation only.
    cfg.drone_rpm       = 8000;
    cfg.drone_blades    = 3;
    cfg.ground_R        = 0.90;          % asphalt reflection coeff (viz label)
    cfg.distance_ref    = 1.0;           % m — 1/r law reference; gains are
                                         %     clamped so d < d_ref -> unity.

    % ---------------- Sources / mix levels ---------------------------
    cfg.drone_wav_path  = fullfile(lab, 'wavs', 'drone_fan.wav');
    cfg.env_wav_path    = fullfile(lab, 'wavs', 'env_ambient.wav');
    cfg.samples_dir     = fullfile(lab, 'samples', 'speech');
    %   Noise loops are unit-RMS; speech sits near ~0.09 RMS, so these
    %   gains set the working SNR. ~0.05 / 0.03 gives a clearly audible
    %   ~6 dB mix the ONNX still recovers well. Pushing them much higher
    %   drops the SNR far enough that the model's speech gate outputs
    %   silence — hence the slider cap (cfg.gain_max) in the UI.
    cfg.speech_gain     = 1.00;          % clean-speech level into the mixer
    cfg.drone_gain_init = 0.05;          % drone-fan mix level
    cfg.env_gain_init   = 0.03;          % environment mix level
    cfg.gain_max        = 0.30;          % UI slider upper bound for both

    % ---------------- ONNX enhancer ----------------------------------
    cfg.onnx_path       = fullfile(proj, 'onnx', 'qwise.onnx');

    % ---------------- Scene presets (radio buttons) ------------------
    %   Each preset overrides human height, drone slant + azimuth, and the
    %   environment distance. Azimuth convention: 0 = center, +90 = left,
    %   -90 = right (looking down the +x axis from the mouth).
    cfg.presets = struct( ...
        'name',        {'Drone 1m · center', 'Drone 1.5m · left', ...
                        'Drone 2m · right',  'Drone 2.5m · center'}, ...
        'human_h',     {1.63, 1.81, 1.90, 1.75}, ...
        'slant_dist',  {1.00, 1.5, 2.00, 2.50}, ...
        'drone_az',    {0,    90,   -90,  0}, ...
        'env_dist',    {4.0,  6.0,  3.0,  7.0});
    cfg.preset_default = 1;

    % ---------------- Recording --------------------------------------
    %   Pressing "Clean (ONNX)" auto-records the noisy mics + the enhanced
    %   output into a timestamped folder under cfg.record.dir:
    %       mic01.wav ... micNN.wav   (noisy array channels)
    %       clean.wav                 (ONNX-enhanced speech)
    cfg.record.dir      = fullfile(lab, 'recordings');
    cfg.record.prefix   = 'qwise';

    % ---------------- Visualisation ----------------------------------
    cfg.ui.spec_ncols    = 120;          % spectrogram history columns
    cfg.ui.waveform_span = cfg.frame_size;
    cfg.ui.fig_position  = [50 40 1560 900];
end
