function tests = test_source_mixer()
%TEST_SOURCE_MIXER  Unit tests for the 3-source → N-mic acoustic mixer.
    tests = functiontests(localfunctions);
end

function setupOnce(tc) %#ok<*DEFNU>
    here = fileparts(fileparts(mfilename('fullpath')));
    addpath(fullfile(here, 'config'));
    addpath(fullfile(here, 'core'));
    cfg = default();
    % Physical acoustic model: every mic picks up every source via TDOA
    % + 1/d gain.  The default wiring is 'perChannel'; the legacy-physics
    % tests below need the research-grade model to exercise geometry.
    cfg.mixer.mode      = 'physical';
    cfg.mixer.composite = 'sum';
    tc.TestData.cfg = cfg;
    tc.TestData.geo = build_geometry(cfg);
end

function test_output_shape(tc)
    cfg = tc.TestData.cfg;
    mix = SourceMixer(cfg, tc.TestData.geo);
    N   = cfg.frame_size;
    out = mix.mix(randn(N,1), randn(N,1), randn(N,1));
    verifyEqual(tc, size(out), [N cfg.n_mics]);
end

function test_drone_only_gain_is_unity_on_closest(tc)
    cfg = tc.TestData.cfg;
    geo = tc.TestData.geo;
    mix = SourceMixer(cfg, geo);
    N   = cfg.frame_size;
    src = sin(2*pi*440*(0:N-1).'/cfg.fs);
    out = mix.mix(zeros(N,1), src, zeros(N,1));
    % The mic with delay==0 for the drone should receive gain==1 exactly.
    [~, m0] = min(geo.delays_drone);
    expected = geo.gains_drone(m0);
    pk = max(abs(out(:, m0))) / max(abs(src));
    verifyEqual(tc, pk, expected, 'AbsTol', 1e-9);
end

function test_speech_attenuates_at_distance(tc)
    cfg = tc.TestData.cfg;
    geo = tc.TestData.geo;
    mix = SourceMixer(cfg, geo);
    N   = cfg.frame_size;
    src = 0.7 * sin(2*pi*200*(0:N-1).'/cfg.fs);
    out = mix.mix(src, zeros(N,1), zeros(N,1));
    [~, m0] = min(geo.delays_speech);
    % Human is 2.5 m away with d_ref = 1 → gain ≈ 0.40
    ratio = max(abs(out(:, m0))) / max(abs(src));
    verifyEqual(tc, ratio, geo.gains_speech(m0), 'AbsTol', 1e-6);
    verifyLessThan(tc, ratio, 0.5);   % clearly attenuated
end

function test_per_source_tdoa_is_applied(tc)
    cfg = tc.TestData.cfg;
    geo = tc.TestData.geo;
    mix = SourceMixer(cfg, geo);
    N   = cfg.frame_size;
    % Impulse at sample 1 on speech source only.
    src = zeros(N,1); src(1) = 1.0;
    out = mix.mix(src, zeros(N,1), zeros(N,1));
    for m = 1:cfg.n_mics
        tau = geo.delays_speech(m);
        g   = geo.gains_speech(m);
        if tau < N
            verifyEqual(tc, out(tau+1, m), g, 'AbsTol', 1e-9);
            if tau > 0
                verifyEqual(tc, out(1:tau, m), zeros(tau,1), 'AbsTol', 1e-12);
            end
        end
    end
end

function test_history_carries_across_blocks(tc)
    cfg = tc.TestData.cfg;
    geo = tc.TestData.geo;
    % Pick whichever source has the largest delay so the tail must land
    % in the next block.
    mix = SourceMixer(cfg, geo);
    N   = cfg.frame_size;
    % Impulse at the last sample of the first speech block.
    src1 = zeros(N,1); src1(end) = 1.0;
    src2 = zeros(N,1);
    mix.mix(src1, zeros(N,1), zeros(N,1));
    out2 = mix.mix(src2, zeros(N,1), zeros(N,1));
    % On the mic with the largest speech delay, the impulse should show
    % up at sample (1 + tau - 1) of the second block for tau >= 1.
    for m = 1:cfg.n_mics
        tau = geo.delays_speech(m);
        g   = geo.gains_speech(m);
        if tau >= 1 && tau < N
            verifyEqual(tc, out2(tau, m), g, 'AbsTol', 1e-9);
        end
    end
end

function test_additivity_of_sources(tc)
    cfg = tc.TestData.cfg;
    geo = tc.TestData.geo;
    N   = cfg.frame_size;
    s = randn(N,1); d = randn(N,1); e = randn(N,1);

    m1 = SourceMixer(cfg, geo); all_ = m1.mix(s, d, e);
    m2 = SourceMixer(cfg, geo); only_s = m2.mix(s, zeros(N,1), zeros(N,1));
    m3 = SourceMixer(cfg, geo); only_d = m3.mix(zeros(N,1), d, zeros(N,1));
    m4 = SourceMixer(cfg, geo); only_e = m4.mix(zeros(N,1), zeros(N,1), e);

    verifyEqual(tc, all_, only_s + only_d + only_e, 'AbsTol', 1e-10);
end

% ====================================================================
%  Per-channel wiring tests  (cfg.mixer.mode = 'perChannel')
%  mic-1 = speech + drone + env  (input mic captures the full scene)
%  mic-2 = drone                 (noise-only reference)
%  mic-3 = env                   (noise-only reference)
% ====================================================================
function test_perchannel_mic_assignment(tc)
    cfg = pc_cfg_(tc);
    geo = tc.TestData.geo;
    mix = SourceMixer(cfg, geo);
    N   = cfg.frame_size;
    s = randn(N,1); d = randn(N,1); e = randn(N,1);
    out = mix.mix(s, d, e);
    g = distance_to_gain(geo.dist_speech(cfg.mwf.ref_mic));
    verifyEqual(tc, size(out), [N cfg.n_mics]);
    verifyEqual(tc, out(:, 1), g*s + d + e, 'AbsTol', 1e-12);
    verifyEqual(tc, out(:, 2), d,           'AbsTol', 1e-12);
    verifyEqual(tc, out(:, 3), e,           'AbsTol', 1e-12);
end

function test_perchannel_default_composite_is_mic1(tc)
    cfg = pc_cfg_(tc);   % default 'mic1'
    geo = tc.TestData.geo;
    mix = SourceMixer(cfg, geo);
    N   = cfg.frame_size;
    s = randn(N,1); d = randn(N,1); e = randn(N,1);
    out  = mix.mix(s, d, e);
    comp = mix.composite(out);
    g = distance_to_gain(geo.dist_speech(cfg.mwf.ref_mic));
    % composite = input mic = distance-attenuated speech + drone + env
    verifyEqual(tc, comp, g*s + d + e, 'AbsTol', 1e-12);
end

function test_perchannel_explicit_sum_composite(tc)
    cfg = pc_cfg_(tc);
    cfg.mixer.composite = 'sum';
    geo = tc.TestData.geo;
    mix = SourceMixer(cfg, geo);
    N   = cfg.frame_size;
    s = randn(N,1); d = randn(N,1); e = randn(N,1);
    out  = mix.mix(s, d, e);
    comp = mix.composite(out);
    g = distance_to_gain(geo.dist_speech(cfg.mwf.ref_mic));
    % sum = mic1 + mic2 + mic3 = (g*s + d + e) + d + e
    verifyEqual(tc, comp, g*s + 2*d + 2*e, 'AbsTol', 1e-12);
end

function test_perchannel_explicit_mean_composite(tc)
    cfg = pc_cfg_(tc);
    cfg.mixer.composite = 'mean';
    geo = tc.TestData.geo;
    mix = SourceMixer(cfg, geo);
    N   = cfg.frame_size;
    s = randn(N,1); d = randn(N,1); e = randn(N,1);
    out  = mix.mix(s, d, e);
    comp = mix.composite(out);
    g = distance_to_gain(geo.dist_speech(cfg.mwf.ref_mic));
    expected = ((g*s + d + e) + d + e) / cfg.n_mics;
    verifyEqual(tc, comp, expected, 'AbsTol', 1e-12);
end

% --------------------------------------------------------------------
%  Distance-band tests for the perChannel speech path.
%  The speech level at mic-1 must follow the piecewise table in
%  core/distance_to_gain.m (0.10–1 m → 0.95, 1–2 m → 0.75, 2–4 m → 0.50,
%  4–6 m → 0.30, 6–8 m → 0.10, >8 m → 0.00). Drone/env on mic-2 and
%  mic-3 must be unaffected.
% --------------------------------------------------------------------
function test_perchannel_speech_attenuates_close(tc)
    cfg = pc_cfg_(tc);
    cfg.slant_dist = 0.5;                       % well inside the 0–1 m band
    geo = build_geometry(cfg);
    mix = SourceMixer(cfg, geo);
    N   = cfg.frame_size;
    s   = randn(N,1);
    out = mix.mix(s, zeros(N,1), zeros(N,1));
    g_expected = distance_to_gain(geo.dist_speech(cfg.mwf.ref_mic));
    verifyEqual(tc, g_expected, 0.95, 'AbsTol', 1e-12);
    verifyEqual(tc, out(:, 1), g_expected * s, 'AbsTol', 1e-12);
end

function test_perchannel_speech_attenuates_mid(tc)
    cfg = pc_cfg_(tc);
    cfg.slant_dist = 3.5;                       % the 2–4 m band → 0.50
    geo = build_geometry(cfg);
    mix = SourceMixer(cfg, geo);
    N   = cfg.frame_size;
    s   = randn(N,1);
    out = mix.mix(s, zeros(N,1), zeros(N,1));
    g_expected = distance_to_gain(geo.dist_speech(cfg.mwf.ref_mic));
    verifyEqual(tc, g_expected, 0.50, 'AbsTol', 1e-12);
    verifyEqual(tc, out(:, 1), g_expected * s, 'AbsTol', 1e-12);
end

function test_perchannel_speech_silenced_far(tc)
    cfg = pc_cfg_(tc);
    cfg.slant_dist = 9.0;                       % beyond 8 m → 0.00
    geo = build_geometry(cfg);
    mix = SourceMixer(cfg, geo);
    N   = cfg.frame_size;
    s   = randn(N,1);
    out = mix.mix(s, zeros(N,1), zeros(N,1));
    g_expected = distance_to_gain(geo.dist_speech(cfg.mwf.ref_mic));
    verifyEqual(tc, g_expected, 0.00, 'AbsTol', 1e-12);
    verifyEqual(tc, out(:, 1), zeros(N,1), 'AbsTol', 1e-12);
end

function test_perchannel_drone_and_env_mics_unaffected_by_distance(tc)
    % mic-2 (drone) and mic-3 (env) must remain noise-only references
    % regardless of the human-to-drone distance.
    for d_slant = [0.5, 3.5, 9.0]
        cfg = pc_cfg_(tc);
        cfg.slant_dist = d_slant;
        geo = build_geometry(cfg);
        mix = SourceMixer(cfg, geo);
        N   = cfg.frame_size;
        s = randn(N,1); d = randn(N,1); e = randn(N,1);
        out = mix.mix(s, d, e);
        verifyEqual(tc, out(:, 2), d, 'AbsTol', 1e-12, ...
            sprintf('mic-2 polluted at slant=%.2f m', d_slant));
        verifyEqual(tc, out(:, 3), e, 'AbsTol', 1e-12, ...
            sprintf('mic-3 polluted at slant=%.2f m', d_slant));
    end
end

function cfg = pc_cfg_(tc)
%PC_CFG_  Clone the physical-mode cfg and flip it to perChannel.
    cfg = tc.TestData.cfg;
    cfg.mixer.mode      = 'perChannel';
    cfg.mixer.composite = 'mic1';
end
