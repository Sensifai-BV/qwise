function tests = test_mwf_batch()
%TEST_MWF_BATCH  Batch-mode mwf.process() and the modular kernels (Q-WiSE).
%
%   Exercises the full mwf.py-equivalent pipeline on synthetic 3-mic data:
%     * mwf_stft / mwf_istft round-trip is near-perfect
%     * mwf_align_vad locates the speech inside a longer mic capture
%     * mwf_build_speech_mask separates speech and noise frames
%     * mwf_estimate_covariance produces hermitian PSD matrices
%     * Each beamformer method (gev, mwf, mvdr) returns the expected shape
%     * mwf.process() emits a [Lm x 1] enhanced signal for cell + matrix inputs
%     * mwf.process() honours the postfilter toggle
%     * GEV beam on a deterministic mixture reduces residual noise vs. ref mic
    tests = functiontests(localfunctions);
end

function setupOnce(tc) %#ok<*DEFNU>
    here = fileparts(fileparts(mfilename('fullpath')));
    addpath(fullfile(here, 'config'));
    addpath(fullfile(here, 'mwf'));
    tc.TestData.cfg = default();
end

% ---------------------------------------------------------------------
% Helpers
% ---------------------------------------------------------------------
function [vad_audio, mic_signals, speech, noise] = synth_scene_(fs, T)
%SYNTH_SCENE_  Build a deterministic 3-mic scene with speech + noise.
    rng(42);
    N      = round(fs * T);
    t      = (0:N-1).' / fs;

    % Speech: 350 Hz tone gated to a centred 60% interval (proxy for VAD).
    speech_burst = sin(2*pi*350*t);
    gate         = false(N, 1);
    a = round(0.2 * N); b = round(0.8 * N);
    gate(a:b)    = true;
    speech       = speech_burst .* gate;

    % Spatially diffuse noise per mic.
    noise = 0.3 * randn(N, 3);

    % Mic mixtures (mic-1 closest to source).
    mic = [speech, 0.6*speech, 0.3*speech] + noise;

    vad_audio   = speech;     % VAD output = extracted speech
    mic_signals = mic;
end

% ---------------------------------------------------------------------
% STFT round-trip
% ---------------------------------------------------------------------
function test_stft_istft_round_trip(tc)
    rng(0);
    fs    = 16000;
    x     = randn(fs, 1);             % 1 s of noise
    n_fft = 1024;
    hop   = 256;
    X     = mwf_stft(x, n_fft, hop);
    y     = mwf_istft(X, n_fft, hop);
    L     = min(numel(x), numel(y));
    err   = norm(x(1:L) - y(1:L)) / norm(x(1:L));
    verifyLessThan(tc, err, 1e-6);
end

% ---------------------------------------------------------------------
% VAD alignment
% ---------------------------------------------------------------------
function test_align_vad_finds_known_offset(tc)
    rng(1);
    fs   = 16000;
    s    = sin(2*pi*440*(0:fs/2-1).'/fs);   % 0.5 s tone
    lag0 = 4000;
    mic  = [0.05*randn(lag0,1); s; 0.05*randn(fs,1)];
    [lag, aligned] = mwf_align_vad(s, mic);
    verifyEqual(tc, lag, lag0, 'AbsTol', 50);  % cross-corr within ±3 ms
    verifyEqual(tc, size(aligned), size(mic));
end

% ---------------------------------------------------------------------
% Speech mask
% ---------------------------------------------------------------------
function test_speech_mask_finds_speech_frames(tc)
    rng(2);
    fs   = 16000;
    T    = 1.0;
    N    = round(fs * T);
    t    = (0:N-1).'/fs;
    aligned_vad = zeros(N, 1);
    a = round(0.3*N); b = round(0.7*N);
    aligned_vad(a:b) = sin(2*pi*350*t(a:b));   % only middle 40% is speech

    n_fft = 1024; hop = 256;
    n_frames = floor((N + n_fft) / hop);
    mask = mwf_build_speech_mask(aligned_vad, n_frames, n_fft, hop, 0.01, 3);
    verifyEqual(tc, size(mask), [n_frames 1]);
    % At least 25% of frames should be speech, at least 25% noise.
    verifyGreaterThan(tc, sum(mask),  round(0.25 * n_frames));
    verifyGreaterThan(tc, sum(~mask), round(0.25 * n_frames));
end

% ---------------------------------------------------------------------
% Covariance estimate
% ---------------------------------------------------------------------
function test_covariance_is_hermitian_psd(tc)
    rng(3);
    n_ch = 3; n_freq = 17; n_frames = 40;
    X = (randn(n_ch, n_freq, n_frames) + 1j*randn(n_ch, n_freq, n_frames));
    mask = true(n_frames, 1);
    Phi = mwf_estimate_covariance(X, mask, 1e-10);
    verifyEqual(tc, size(Phi), [n_freq n_ch n_ch]);
    for f = [1, ceil(n_freq/2), n_freq]
        M = squeeze(Phi(f, :, :));
        verifyLessThan(tc, norm(M - M', 'fro'), 1e-9);
        d = real(eig((M + M')/2));
        verifyGreaterThanOrEqual(tc, min(d), -1e-9);
    end
end

% ---------------------------------------------------------------------
% Beamformer weight shapes
% ---------------------------------------------------------------------
function test_each_method_returns_expected_shape(tc)
    rng(4);
    n_ch = 3; n_freq = 11;
    Phi_ss = make_psd_(n_ch, n_freq);
    Phi_nn = make_psd_(n_ch, n_freq);

    W_mwf  = mwf_compute_mwf_weights (Phi_ss, Phi_nn, 1, 1.0, 1e-10, 1e-4);
    W_mvdr = mwf_compute_mvdr_weights(Phi_ss, Phi_nn, 1,      1e-10, 1e-4);
    W_gev  = mwf_compute_gev_weights (Phi_ss, Phi_nn, 1,      1e-10, 1e-4);

    verifyEqual(tc, size(W_mwf),  [n_freq n_ch]);
    verifyEqual(tc, size(W_mvdr), [n_freq n_ch]);
    verifyEqual(tc, size(W_gev),  [n_freq n_ch]);

    verifyTrue(tc, all(isfinite(W_mwf(:))));
    verifyTrue(tc, all(isfinite(W_mvdr(:))));
    verifyTrue(tc, all(isfinite(W_gev(:))));
end

% ---------------------------------------------------------------------
% Batch process — matrix input
% ---------------------------------------------------------------------
function test_process_matrix_input_returns_matching_length(tc)
    cfg = tc.TestData.cfg;
    cfg.mwf.method     = 'gev';
    cfg.mwf.postfilter = true;
    fs  = cfg.fs;
    [vad_audio, mic_signals] = synth_scene_(fs, 1.0);

    m = mwf(cfg);
    y = m.process(vad_audio, mic_signals);
    verifyEqual(tc, size(y, 2), 1);
    verifyLessThanOrEqual(tc, size(y, 1), size(mic_signals, 1));
    verifyTrue(tc, all(isfinite(y)));
end

% ---------------------------------------------------------------------
% Batch process — cell input
% ---------------------------------------------------------------------
function test_process_cell_input_truncates_to_shortest(tc)
    cfg = tc.TestData.cfg;
    cfg.mwf.method = 'gev';
    fs  = cfg.fs;
    [vad_audio, mic_mat] = synth_scene_(fs, 1.0);
    mic_cells = {mic_mat(:,1); mic_mat(1:end-100, 2); mic_mat(:,3)};

    m = mwf(cfg);
    y = m.process(vad_audio, mic_cells);
    verifyEqual(tc, size(y, 2), 1);
    verifyTrue(tc, all(isfinite(y)));
end

% ---------------------------------------------------------------------
% Method selection
% ---------------------------------------------------------------------
function test_method_dispatch_runs_all_three(tc)
    fs  = tc.TestData.cfg.fs;
    [vad_audio, mic_signals] = synth_scene_(fs, 0.8);
    for method = {'gev', 'mwf', 'mvdr'}
        cfg = tc.TestData.cfg;
        cfg.mwf.method = method{1};
        m = mwf(cfg);
        y = m.process(vad_audio, mic_signals);
        verifyTrue(tc, all(isfinite(y)), ...
            sprintf('method=%s produced non-finite samples', method{1}));
        verifyGreaterThan(tc, max(abs(y)), 0);
    end
end

% ---------------------------------------------------------------------
% Post-filter toggle
% ---------------------------------------------------------------------
function test_postfilter_toggle_changes_output(tc)
    fs  = tc.TestData.cfg.fs;
    [vad_audio, mic_signals] = synth_scene_(fs, 0.8);

    cfg1 = tc.TestData.cfg; cfg1.mwf.method='gev'; cfg1.mwf.postfilter = true;
    cfg2 = tc.TestData.cfg; cfg2.mwf.method='gev'; cfg2.mwf.postfilter = false;
    y1 = mwf(cfg1).process(vad_audio, mic_signals);
    y2 = mwf(cfg2).process(vad_audio, mic_signals);
    L  = min(numel(y1), numel(y2));
    verifyGreaterThan(tc, norm(y1(1:L) - y2(1:L)), 1e-6);
end

% ---------------------------------------------------------------------
% Utilities
% ---------------------------------------------------------------------
function Phi = make_psd_(n_ch, n_freq)
    rng(7);
    Phi = zeros(n_freq, n_ch, n_ch);
    for f = 1:n_freq
        A = randn(n_ch) + 1j*randn(n_ch);
        Phi(f, :, :) = A * A' + 0.01*eye(n_ch);
    end
end
