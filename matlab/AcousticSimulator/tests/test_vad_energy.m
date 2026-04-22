function tests = test_vad_energy()
%TEST_VAD_ENERGY  Unit tests for the statistical (energy+SFM) VAD fallback.
    tests = functiontests(localfunctions);
end

function setupOnce(tc) %#ok<*DEFNU>
    here = fileparts(fileparts(mfilename('fullpath')));
    addpath(fullfile(here, 'config'));
    addpath(fullfile(here, 'vad'));
    tc.TestData.cfg = default();
end

function test_silence_is_not_speech(tc)
    cfg = tc.TestData.cfg;
    v   = VADEnergy(cfg);
    N   = cfg.frame_size;
    for k = 1:5
        [is_speech, score] = v.step(1e-6 * randn(N, 1));
        verifyFalse(tc, is_speech);
        verifyLessThan(tc, score, 0.5);
    end
end

function test_tonal_speech_proxy_is_detected(tc)
    cfg = tc.TestData.cfg;
    v   = VADEnergy(cfg);
    N   = cfg.frame_size;
    t   = (0:N-1)' / cfg.fs;
    % A mix of formant-like sinusoids + mild noise ~ voiced speech proxy
    sig = 0.30 * sin(2*pi*220*t) + 0.18 * sin(2*pi*480*t) + ...
          0.12 * sin(2*pi*920*t) + 0.02 * randn(N, 1);
    detected = false;
    for k = 1:6
        [is_speech, ~] = v.step(sig);
        if is_speech, detected = true; break; end
    end
    verifyTrue(tc, detected, ...
        'Energy+SFM VAD must detect the voiced-speech proxy.');
end

function test_hangover_smooths_brief_drops(tc)
    cfg  = tc.TestData.cfg;
    cfg.vad.hang_frames = 5;
    v    = VADEnergy(cfg);
    N    = cfg.frame_size;
    t    = (0:N-1)' / cfg.fs;
    loud = 0.30 * sin(2*pi*220*t) + 0.20 * sin(2*pi*480*t);
    quiet = 1e-6 * randn(N, 1);
    for k = 1:4, v.step(loud); end
    [sp, ~] = v.step(quiet);  % first quiet frame within hangover
    verifyTrue(tc, sp, 'Hangover should keep speech-state for one frame.');
end
