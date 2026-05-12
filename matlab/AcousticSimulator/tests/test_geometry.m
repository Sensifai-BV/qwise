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

function test_env_position_comes_from_cfg(tc)
%TEST_ENV_POSITION_COMES_FROM_CFG  The environment-noise source must sit
%   at the spherical position described by cfg.env.* (independent of the
%   drone). Verified by setting concrete values and checking the
%   resolved Cartesian distance/elevation.
    cfg = tc.TestData.cfg;
    cfg.env.distance_from_mouth = 6.0;
    cfg.env.azimuth_deg         = 90;
    cfg.env.elevation_deg       = 0;
    geo = build_geometry(cfg);
    % azimuth=90° → +y direction, elevation=0 → same height as mouth
    expected = [0, 6.0, cfg.mouth_height];
    verifyEqual(tc, geo.pos_env, expected, 'AbsTol', 1e-9);
    verifyEqual(tc, geo.pos_env_noise, expected, 'AbsTol', 1e-9, ...
        'pos_env_noise legacy alias must equal pos_env.');
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

function test_per_source_frac_delays_exist(tc)
%TEST_PER_SOURCE_FRAC_DELAYS_EXIST  The SourceMixer relies on these
%   fractional-sample delay arrays for its interp-based taps.
    cfg = tc.TestData.cfg;
    geo = build_geometry(cfg);
    for f = {'frac_delays_speech','frac_delays_drone','frac_delays_env'}
        verifyTrue(tc, isfield(geo, f{1}), sprintf('Missing %s', f{1}));
        verifyEqual(tc, size(geo.(f{1})), [cfg.n_mics 1]);
        verifyGreaterThanOrEqual(tc, min(geo.(f{1})), 0);
        verifyEqual(tc, min(geo.(f{1})), 0, 'AbsTol', 1e-12, ...
            sprintf('%s must reference the closest mic to zero.', f{1}));
    end
    % Round-to-int agreement with the legacy integer delays.
    verifyEqual(tc, round(geo.frac_delays_speech), geo.delays_speech);
    verifyEqual(tc, round(geo.frac_delays_drone),  geo.delays_drone);
    verifyEqual(tc, round(geo.frac_delays_env),    geo.delays_env);
end

function test_per_source_gains_obey_inverse_distance(tc)
    cfg = tc.TestData.cfg;
    geo = build_geometry(cfg);
    % Human at slant_dist (>d_ref) must attenuate overall.
    verifyLessThanOrEqual(tc, max(geo.gains_speech), 1 + 1e-9);
    % Gain must exactly match the 1/d law clamped at d_ref — this is the
    % real correctness contract.
    expected = cfg.distance_ref ./ max(geo.dist_speech, cfg.distance_ref);
    verifyEqual(tc, geo.gains_speech, expected, 'AbsTol', 1e-12);
    % Ref-mic gain must agree with the 1/r law at the ref mic's actual
    % distance from the speaker — NOT against 1/slant_dist, because for
    % wider arrays the ref mic does not sit at slant_dist from the human.
    ref      = cfg.mwf.ref_mic;
    g_ref    = cfg.distance_ref / max(geo.dist_speech(ref), cfg.distance_ref);
    verifyEqual(tc, geo.gains_speech(ref), g_ref, 'AbsTol', 1e-12);
    % Drone co-located with mic centroid → at least one mic clamps to 1.0.
    verifyEqual(tc, max(geo.gains_drone), 1.0, 'AbsTol', 1e-9);
    % Env noise is far (8 m default) → gain < 0.5.
    verifyLessThan(tc, max(geo.gains_env), 0.5);
end

function test_circular_geometry_chord_spacing(tc)
%TEST_CIRCULAR_GEOMETRY_CHORD_SPACING  Circular geometry must keep
%   adjacent-mic chord distance ≈ cfg.mic_spacing.
    cfg = tc.TestData.cfg;
    cfg.mic_geometry = 'circular';
    cfg.n_mics       = 4;
    geo = build_geometry(cfg);
    verifyEqual(tc, size(geo.pos_mics), [cfg.n_mics 3]);
    d12 = norm(geo.pos_mics(1, :) - geo.pos_mics(2, :));
    verifyEqual(tc, d12, cfg.mic_spacing, 'AbsTol', 1e-6);
end
