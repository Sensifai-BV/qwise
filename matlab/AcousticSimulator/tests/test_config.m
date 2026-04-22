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
                  'drone_wav_path','env_wav_path','vad','mwf', ...
                  'playback','ui'};
    for k = 1:numel(fields_top)
        verifyTrue(tc, isfield(cfg, fields_top{k}), ...
            sprintf('Missing cfg field: %s', fields_top{k}));
    end
    vad_req = {'backend','onnx_path','silero_frame','silero_threshold', ...
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
