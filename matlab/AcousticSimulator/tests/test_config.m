function tests = test_config()
%TEST_CONFIG  Sanity checks on the default configuration struct.
    tests = functiontests(localfunctions);
end

function setupOnce(tc) %#ok<*DEFNU>
    here = fileparts(fileparts(mfilename('fullpath')));
    addpath(fullfile(here, 'config'));
    tc.TestData.cfg = default();
end

function test_required_fields_present(tc)
    cfg = tc.TestData.cfg;
    fields_top = {'fs','frame_size','c','n_mics','mic_spacing', ...
                  'human_height','mouth_height','slant_dist','elev_deg', ...
                  'distance_ref', ...
                  'speech_gain_init','drone_gain_init','env_gain_init', ...
                  'drone_wav_path','env_wav_path','mixer','vad','mwf', ...
                  'record','ui'};
    for k = 1:numel(fields_top)
        verifyTrue(tc, isfield(cfg, fields_top{k}), ...
            sprintf('Missing cfg field: %s', fields_top{k}));
    end
    vad_req = {'backend','onnx_path','qwise_frame','qwise_threshold', ...
               'energy_threshold','sfm_threshold','hang_frames','smoothing'};
    for k = 1:numel(vad_req)
        verifyTrue(tc, isfield(cfg.vad, vad_req{k}), ...
            sprintf('Missing cfg.vad field: %s', vad_req{k}));
    end
    mwf_req = {'enabled','stft_win','stft_hop','ref_mic','mu','eps_reg', ...
               'alpha_nn','alpha_ss','passthrough'};
    for k = 1:numel(mwf_req)
        verifyTrue(tc, isfield(cfg.mwf, mwf_req{k}), ...
            sprintf('Missing cfg.mwf field: %s', mwf_req{k}));
    end
    mixer_req = {'mode','composite'};
    for k = 1:numel(mixer_req)
        verifyTrue(tc, isfield(cfg.mixer, mixer_req{k}), ...
            sprintf('Missing cfg.mixer field: %s', mixer_req{k}));
    end
    rec_req = {'dir','prefix'};
    for k = 1:numel(rec_req)
        verifyTrue(tc, isfield(cfg.record, rec_req{k}), ...
            sprintf('Missing cfg.record field: %s', rec_req{k}));
    end
    % mic_model is intentionally gone — do not resurrect it.
    verifyFalse(tc, isfield(cfg, 'mic_model'), ...
        'cfg.mic_model must be removed — find_input_mic() auto-detects now.');
end

function test_mixer_defaults_are_sane(tc)
    cfg = tc.TestData.cfg;
    verifyTrue(tc, ismember(lower(cfg.mixer.mode), {'perchannel','physical'}), ...
        'cfg.mixer.mode must be ''perChannel'' or ''physical''.');
    verifyTrue(tc, ismember(lower(cfg.mixer.composite), {'mic1','sum','mean'}), ...
        'cfg.mixer.composite must be ''mic1'', ''sum'' or ''mean''.');
    verifyGreaterThan(tc, cfg.speech_gain_init, 0);
    verifyGreaterThanOrEqual(tc, cfg.drone_gain_init, 0);
    verifyGreaterThanOrEqual(tc, cfg.env_gain_init,   0);
end

function test_sample_rate_and_frames_are_sane(tc)
    cfg = tc.TestData.cfg;
    verifyGreaterThan(tc, cfg.fs, 0);
    verifyGreaterThan(tc, cfg.frame_size, 0);
    verifyGreaterThanOrEqual(tc, cfg.n_mics, 1);
    verifyGreaterThan(tc, cfg.mic_spacing, 0);
    verifyEqual(tc, mod(log2(cfg.mwf.stft_win), 1), 0, ...
        'STFT window should be a power of two for efficient FFT.');
end
