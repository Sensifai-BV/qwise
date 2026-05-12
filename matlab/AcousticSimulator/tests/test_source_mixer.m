function tests = test_source_mixer()
%TEST_SOURCE_MIXER  Unit tests for the physical 3-source → N-mic mixer.
%
%   Contract: every microphone receives all three sources with a
%   fractional sample delay and a 1/r spreading gain clamped at
%   cfg.distance_ref.  The legacy perChannel wiring is gone — there is
%   only one physical model now.
    tests = functiontests(localfunctions);
end

function setupOnce(tc) %#ok<*DEFNU>
    here = fileparts(fileparts(mfilename('fullpath')));
    addpath(fullfile(here, 'config'));
    addpath(fullfile(here, 'core'));
    cfg = default();
    cfg.mixer.composite = 'mic1';   % ref-mic composite for the VAD feed
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
%TEST_DRONE_ONLY_GAIN_IS_UNITY_ON_CLOSEST  At the mic with frac-delay 0
%   for the drone, the gain should match geo.gains_drone exactly because
%   no interpolation is applied (delay is zero).
    cfg = tc.TestData.cfg;
    geo = tc.TestData.geo;
    mix = SourceMixer(cfg, geo);
    N   = cfg.frame_size;
    src = sin(2*pi*440*(0:N-1).'/cfg.fs);
    out = mix.mix(zeros(N,1), src, zeros(N,1));
    [~, m0] = min(geo.frac_delays_drone);
    verifyEqual(tc, geo.frac_delays_drone(m0), 0, 'AbsTol', 1e-12);
    expected = geo.gains_drone(m0);
    pk = max(abs(out(:, m0))) / max(abs(src));
    verifyEqual(tc, pk, expected, 'AbsTol', 1e-9);
end

function test_speech_attenuates_at_distance(tc)
%TEST_SPEECH_ATTENUATES_AT_DISTANCE  With the default 2.5 m slant the
%   1/r gain at the closest mic must be < 0.5 and equal to the geometry
%   prediction.
    cfg = tc.TestData.cfg;
    geo = tc.TestData.geo;
    mix = SourceMixer(cfg, geo);
    N   = cfg.frame_size;
    src = 0.7 * sin(2*pi*200*(0:N-1).'/cfg.fs);
    out = mix.mix(src, zeros(N,1), zeros(N,1));
    [~, m0] = min(geo.frac_delays_speech);
    ratio = max(abs(out(:, m0))) / max(abs(src));
    verifyEqual(tc, ratio, geo.gains_speech(m0), 'AbsTol', 1e-6);
    verifyLessThan(tc, ratio, 0.5);
end

function test_speech_impulse_smears_across_fractional_tap(tc)
%TEST_SPEECH_IMPULSE_SMEARS_ACROSS_FRACTIONAL_TAP  A unit impulse on the
%   speech source must appear on every mic at floor(tau)+1 and
%   floor(tau)+2 with weights (1-frac) and frac, summed to gain.
    cfg = tc.TestData.cfg;
    geo = tc.TestData.geo;
    mix = SourceMixer(cfg, geo);
    N   = cfg.frame_size;
    src = zeros(N,1); src(1) = 1.0;
    out = mix.mix(src, zeros(N,1), zeros(N,1));
    for m = 1:cfg.n_mics
        tau   = geo.frac_delays_speech(m);
        g     = geo.gains_speech(m);
        if tau >= N - 1
            continue;
        end
        i_a   = floor(tau) + 1;
        i_b   = i_a + 1;
        frac  = tau - floor(tau);
        % Linear-interp tap → impulse splits across the two integer bins
        verifyEqual(tc, out(i_a, m), g * (1 - frac), 'AbsTol', 1e-9, ...
            sprintf('Mic %d: leading bin mismatch at tau=%.4f', m, tau));
        if i_b <= N
            verifyEqual(tc, out(i_b, m), g * frac, 'AbsTol', 1e-9, ...
                sprintf('Mic %d: trailing bin mismatch at tau=%.4f', m, tau));
        end
        % Everything else in the leading window must be zero.
        if i_a > 1
            verifyEqual(tc, out(1:i_a-1, m), zeros(i_a-1, 1), 'AbsTol', 1e-12);
        end
    end
end

function test_history_carries_across_blocks(tc)
%TEST_HISTORY_CARRIES_ACROSS_BLOCKS  An impulse at the last sample of
%   block 1 must land inside block 2 at floor(tau)/floor(tau)+1 with the
%   correct interpolation weights, proving the per-source history ring
%   stitches blocks together.
    cfg = tc.TestData.cfg;
    geo = tc.TestData.geo;
    mix = SourceMixer(cfg, geo);
    N   = cfg.frame_size;
    src1 = zeros(N,1); src1(end) = 1.0;
    src2 = zeros(N,1);
    mix.mix(src1, zeros(N,1), zeros(N,1));
    out2 = mix.mix(src2, zeros(N,1), zeros(N,1));
    for m = 1:cfg.n_mics
        tau  = geo.frac_delays_speech(m);
        g    = geo.gains_speech(m);
        if tau < 1 || tau >= N
            continue;
        end
        i_a  = floor(tau);
        i_b  = i_a + 1;
        frac = tau - floor(tau);
        if i_a >= 1
            verifyEqual(tc, out2(i_a, m), g * (1 - frac), 'AbsTol', 1e-9, ...
                sprintf('Mic %d: block-2 leading bin mismatch', m));
        end
        if i_b <= N
            verifyEqual(tc, out2(i_b, m), g * frac, 'AbsTol', 1e-9, ...
                sprintf('Mic %d: block-2 trailing bin mismatch', m));
        end
    end
end

function test_additivity_of_sources(tc)
%TEST_ADDITIVITY_OF_SOURCES  Linear superposition: mix(s,d,e) ==
%   mix(s,0,0) + mix(0,d,0) + mix(0,0,e). Confirms each source path
%   is independent of the others.
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

function test_composite_default_is_ref_mic(tc)
%TEST_COMPOSITE_DEFAULT_IS_REF_MIC  In the default config, composite()
%   must return the reference-mic channel — the single mono signal that
%   feeds the VAD.
    cfg = tc.TestData.cfg;
    geo = tc.TestData.geo;
    mix = SourceMixer(cfg, geo);
    N   = cfg.frame_size;
    s = randn(N,1); d = randn(N,1); e = randn(N,1);
    out  = mix.mix(s, d, e);
    comp = mix.composite(out);
    verifyEqual(tc, comp, out(:, cfg.mwf.ref_mic), 'AbsTol', 1e-12);
end

function test_composite_sum_and_mean(tc)
%TEST_COMPOSITE_SUM_AND_MEAN  'sum' and 'mean' reductions still work for
%   diagnostics.
    geo = tc.TestData.geo;
    N   = tc.TestData.cfg.frame_size;
    s = randn(N,1); d = randn(N,1); e = randn(N,1);

    cfg_sum  = tc.TestData.cfg;  cfg_sum.mixer.composite  = 'sum';
    cfg_mean = tc.TestData.cfg;  cfg_mean.mixer.composite = 'mean';

    m_sum  = SourceMixer(cfg_sum,  geo);  out_sum  = m_sum.mix(s, d, e);
    m_mean = SourceMixer(cfg_mean, geo);  out_mean = m_mean.mix(s, d, e);

    verifyEqual(tc, m_sum.composite(out_sum),   sum(out_sum,  2), 'AbsTol', 1e-12);
    verifyEqual(tc, m_mean.composite(out_mean), mean(out_mean, 2), 'AbsTol', 1e-12);
end

function test_every_mic_receives_every_source(tc)
%TEST_EVERY_MIC_RECEIVES_EVERY_SOURCE  In the physical model no mic is
%   noise-only: each of N channels carries contributions from speech,
%   drone, and environment (with their respective gains).
    cfg = tc.TestData.cfg;
    geo = tc.TestData.geo;
    mix_speech = SourceMixer(cfg, geo);
    mix_drone  = SourceMixer(cfg, geo);
    mix_env    = SourceMixer(cfg, geo);
    N   = cfg.frame_size;
    src = randn(N,1);

    out_s = mix_speech.mix(src, zeros(N,1), zeros(N,1));
    out_d = mix_drone.mix(zeros(N,1), src, zeros(N,1));
    out_e = mix_env.mix(zeros(N,1), zeros(N,1), src);

    for m = 1:cfg.n_mics
        verifyGreaterThan(tc, max(abs(out_s(:, m))), 0, ...
            sprintf('Mic %d: missing speech contribution.', m));
        verifyGreaterThan(tc, max(abs(out_d(:, m))), 0, ...
            sprintf('Mic %d: missing drone contribution.', m));
        verifyGreaterThan(tc, max(abs(out_e(:, m))), 0, ...
            sprintf('Mic %d: missing env contribution.', m));
    end
end
