function tests = test_mwf_passthrough()
%TEST_MWF_PASSTHROUGH  Sanity tests for the MWF streaming pass-through path.
    tests = functiontests(localfunctions);
end

function setupOnce(tc) %#ok<*DEFNU>
    here = fileparts(fileparts(mfilename('fullpath')));
    addpath(fullfile(here, 'config'));
    addpath(fullfile(here, 'mwf'));
    tc.TestData.cfg = default();
end

function test_passthrough_returns_reference_mic(tc)
    cfg = tc.TestData.cfg;
    cfg.mwf.passthrough = true;
    m   = mwf(cfg);
    N   = cfg.frame_size;
    x   = randn(N, cfg.n_mics);
    y   = m.step(x, false);
    verifyEqual(tc, size(y), [N 1]);
    verifyEqual(tc, y, x(:, cfg.mwf.ref_mic), 'AbsTol', 1e-12);
end

function test_covariance_buffers_initialised(tc)
    cfg = tc.TestData.cfg;
    m   = mwf(cfg);
    verifyEqual(tc, size(m.Rnn), [m.nbin cfg.n_mics cfg.n_mics]);
    verifyEqual(tc, size(m.Rss), [m.nbin cfg.n_mics cfg.n_mics]);
    R0 = squeeze(m.Rnn(1, :, :));
    verifyEqual(tc, R0, cfg.mwf.eps_reg * eye(cfg.n_mics), 'AbsTol', 1e-12);
end

function test_gain_map_placeholder_is_ones(tc)
    cfg = tc.TestData.cfg;
    X = randn(cfg.mwf.stft_win/2 + 1, cfg.n_mics);
    G = get_tf_gain_map(X, cfg);
    verifyEqual(tc, size(G), [size(X, 1) 1]);
    verifyEqual(tc, G, ones(size(X, 1), 1), 'AbsTol', 1e-12);
end

function test_reset_restores_initial_covariances(tc)
    cfg = tc.TestData.cfg;
    cfg.mwf.passthrough = false;
    m   = mwf(cfg);
    N   = cfg.frame_size;
    m.step(randn(N, cfg.n_mics), true);
    m.reset();
    R0 = squeeze(m.Rnn(1, :, :));
    verifyEqual(tc, R0, cfg.mwf.eps_reg * eye(cfg.n_mics), 'AbsTol', 1e-12);
end

function test_invalid_method_rejected(tc)
    cfg = tc.TestData.cfg;
    cfg.mwf.method = 'not-a-method';
    f = @() mwf(cfg);
    verifyError(tc, f, 'mwf:bad_method');
end
