function tests = test_source_mixer_update_geometry()
%TEST_SOURCE_MIXER_UPDATE_GEOMETRY  Tests for live geometry updates.
%
%   The GUI scene sliders rebuild geometry on every change and call
%   SourceMixer.update_geometry(new_geo) so the next mix() block picks
%   up the new gains/delays. These tests guarantee:
%     * gains and frac-delays are actually swapped in
%     * per-source history rings are resized correctly
%     * a step change in geometry does NOT click on the boundary
%       (the mixer produces a finite, bounded output)
    tests = functiontests(localfunctions);
end

function setupOnce(tc) %#ok<*DEFNU>
    here = fileparts(fileparts(mfilename('fullpath')));
    addpath(fullfile(here, 'config'));
    addpath(fullfile(here, 'core'));
    tc.TestData.cfg = default();
end

function test_update_changes_gains(tc)
    cfg  = tc.TestData.cfg;
    geo1 = build_geometry(cfg);
    mix  = SourceMixer(cfg, geo1);

    cfg2 = cfg;
    cfg2.slant_dist = 6.0;                  % human further → smaller speech gain
    geo2 = build_geometry(cfg2);
    mix.update_geometry(geo2);

    g_old = geo1.gains_speech(cfg.mwf.ref_mic);
    g_new = geo2.gains_speech(cfg.mwf.ref_mic);
    verifyLessThan(tc, g_new, g_old, ...
        'Speech gain must drop when the human moves farther away.');

    % And the mixer must actually use the new value on the next block.
    N   = cfg.frame_size;
    s   = randn(N, 1);
    out = mix.mix(s, zeros(N,1), zeros(N,1));
    [~, m0] = min(geo2.frac_delays_speech);
    ratio   = max(abs(out(:, m0))) / max(abs(s));
    verifyEqual(tc, ratio, g_new, 'AbsTol', 1e-6);
end

function test_update_resizes_history_without_click(tc)
%TEST_UPDATE_RESIZES_HISTORY_WITHOUT_CLICK  Run a steady-state block,
%   step the geometry, then run another block of the same input — the
%   second block must be finite and free of NaN/Inf regardless of the
%   history-buffer resize.
    cfg = tc.TestData.cfg;
    geo = build_geometry(cfg);
    mix = SourceMixer(cfg, geo);

    N = cfg.frame_size;
    s = sin(2*pi*250*(0:N-1).'/cfg.fs);

    mix.mix(s, zeros(N,1), zeros(N,1));     % prime history

    cfg2 = cfg;
    cfg2.slant_dist = 0.5;                  % much closer → tiny delays
    mix.update_geometry(build_geometry(cfg2));
    out_small = mix.mix(s, zeros(N,1), zeros(N,1));
    verifyTrue(tc, all(isfinite(out_small(:))), ...
        'Mixer output contains NaN/Inf after shrinking history.');
    verifyLessThanOrEqual(tc, max(abs(out_small(:))), 2, ...
        'Mixer output blew up after shrinking history.');

    cfg3 = cfg;
    cfg3.slant_dist = 8.5;                  % much farther → large delays
    mix.update_geometry(build_geometry(cfg3));
    out_big = mix.mix(s, zeros(N,1), zeros(N,1));
    verifyTrue(tc, all(isfinite(out_big(:))), ...
        'Mixer output contains NaN/Inf after growing history.');
    verifyLessThanOrEqual(tc, max(abs(out_big(:))), 2, ...
        'Mixer output blew up after growing history.');
end

function test_update_rejects_n_mic_change(tc)
%TEST_UPDATE_REJECTS_N_MIC_CHANGE  Changing cfg.n_mics at runtime would
%   break downstream buffer shapes, so update_geometry must throw.
    cfg = tc.TestData.cfg;
    mix = SourceMixer(cfg, build_geometry(cfg));
    cfg2 = cfg;
    cfg2.n_mics = cfg.n_mics + 2;
    geo2 = build_geometry(cfg2);
    f = @() mix.update_geometry(geo2);
    verifyError(tc, f, 'SourceMixer:update_geometry:NMicChange');
end
