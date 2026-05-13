function tests = test_expand_channels()
%TEST_EXPAND_CHANNELS  TDOA virtual-mic expansion behaviour.
    tests = functiontests(localfunctions);
end

function setupOnce(tc) %#ok<*DEFNU>
    here = fileparts(fileparts(mfilename('fullpath')));
    addpath(fullfile(here, 'config'));
    addpath(fullfile(here, 'core'));
    tc.TestData.cfg = default();
end

function test_zero_delay_mic_matches_source(tc)
    cfg = tc.TestData.cfg;
    geo = build_geometry(cfg);
    raw = randn(cfg.frame_size, 1);
    out = expand_channels(raw, geo, cfg.n_mics);
    [~, idx0] = min(geo.delays);
    verifyEqual(tc, out(:, idx0), raw, 'AbsTol', 1e-12);
end

function test_delayed_mic_is_shifted_copy(tc)
    cfg = tc.TestData.cfg;
    geo = build_geometry(cfg);
    N   = cfg.frame_size;
    raw = sin(2*pi*(0:N-1)'/64);
    out = expand_channels(raw, geo, cfg.n_mics);
    for m = 1:cfg.n_mics
        tau = geo.delays(m);
        if tau == 0
            continue;
        end
        if tau >= N
            continue;
        end
        expected = [zeros(tau, 1); raw(1:N-tau)];
        verifyEqual(tc, out(:, m), expected, 'AbsTol', 1e-12, ...
            sprintf('Mic %d delay mismatch (tau=%d)', m, tau));
    end
end

function test_stereo_input_wires_outer_mics_directly(tc)
    cfg = tc.TestData.cfg;
    geo = build_geometry(cfg);
    N   = cfg.frame_size;
    raw = [randn(N, 1), randn(N, 1)];
    out = expand_channels(raw, geo, cfg.n_mics);
    verifyEqual(tc, out(:, 1),           raw(:, 1), 'AbsTol', 1e-12);
    verifyEqual(tc, out(:, cfg.n_mics),  raw(:, 2), 'AbsTol', 1e-12);
end
