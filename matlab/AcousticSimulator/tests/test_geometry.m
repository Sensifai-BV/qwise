function tests = test_geometry()
%TEST_GEOMETRY  Unit tests for build_geometry / expand_channels.
    tests = functiontests(localfunctions);
end

function setupOnce(tc) %#ok<*DEFNU>
    here = fileparts(fileparts(mfilename('fullpath')));
    addpath(fullfile(here, 'config'));
    addpath(fullfile(here, 'core'));
    tc.TestData.cfg = default();
end

function test_positions_and_shapes(tc)
    cfg = tc.TestData.cfg;
    geo = build_geometry(cfg);
    verifyEqual(tc, size(geo.pos_mics), [cfg.n_mics 3]);
    verifyEqual(tc, geo.pos_human, [0 0 cfg.mouth_height], 'AbsTol', 1e-9);
    verifyEqual(tc, geo.pos_img_src(3), -cfg.mouth_height, 'AbsTol', 1e-9);
    verifyEqual(tc, size(geo.delays), [cfg.n_mics 1]);
end

function test_delays_are_nonneg_and_min_zero(tc)
    cfg = tc.TestData.cfg;
    geo = build_geometry(cfg);
    verifyGreaterThanOrEqual(tc, min(geo.delays), 0);
    verifyEqual(tc, min(geo.delays), 0);
end

function test_drone_distance_matches_slant(tc)
    cfg = tc.TestData.cfg;
    geo = build_geometry(cfg);
    d = norm(geo.pos_drone - geo.pos_human);
    verifyEqual(tc, d, cfg.slant_dist, 'AbsTol', 1e-6);
end

function test_expand_channels_shape_and_energy(tc)
    cfg = tc.TestData.cfg;
    geo = build_geometry(cfg);
    N   = cfg.frame_size;
    raw = randn(N, 1);
    out = expand_channels(raw, geo, cfg.n_mics);
    verifyEqual(tc, size(out), [N cfg.n_mics]);
    % The mic with zero delay must be identical to the source.
    [~, idx0] = min(geo.delays);
    verifyEqual(tc, out(:, idx0), raw, 'AbsTol', 1e-12);
end

function test_mic_spacing_matches_cfg(tc)
    cfg = tc.TestData.cfg;
    geo = build_geometry(cfg);
    d12 = norm(geo.pos_mics(1, :) - geo.pos_mics(2, :));
    verifyEqual(tc, d12, cfg.mic_spacing, 'AbsTol', 1e-9);
end

function test_per_source_delays_exist(tc)
    cfg = tc.TestData.cfg;
    geo = build_geometry(cfg);
    for f = {'delays_speech','delays_drone','delays_env'}
        verifyTrue(tc, isfield(geo, f{1}), sprintf('Missing %s', f{1}));
        verifyEqual(tc, size(geo.(f{1})), [cfg.n_mics 1]);
        verifyGreaterThanOrEqual(tc, min(geo.(f{1})), 0);
        verifyEqual(tc, min(geo.(f{1})), 0);
    end
end

function test_per_source_gains_obey_inverse_distance(tc)
    cfg = tc.TestData.cfg;
    geo = build_geometry(cfg);
    % Human at slant_dist (>d_ref) must attenuate overall.
    verifyLessThanOrEqual(tc, max(geo.gains_speech), 1 + 1e-9);
    % Gain must exactly match the 1/d law clamped at d_ref.
    expected = cfg.distance_ref ./ max(geo.dist_speech, cfg.distance_ref);
    verifyEqual(tc, geo.gains_speech, expected, 'AbsTol', 1e-12);
    % Sanity: human at slant_dist (~3.5 m) → gain ~ d_ref/slant ~ 0.29.
    g_expected = cfg.distance_ref / cfg.slant_dist;
    verifyGreaterThan(tc, geo.gains_speech(cfg.mwf.ref_mic), g_expected*0.90);
    verifyLessThan   (tc, geo.gains_speech(cfg.mwf.ref_mic), g_expected*1.10);
    % Drone co-located with mics → gain clamps to 1.0.
    verifyEqual(tc, max(geo.gains_drone), 1.0, 'AbsTol', 1e-9);
    % Env noise is far → gain < 0.5.
    verifyLessThan(tc, max(geo.gains_env), 0.5);
end
