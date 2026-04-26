classdef VADSilero < handle
%VADSILERO  Silero-VAD ONNX wrapper with graceful multi-API fallback.
%
%   Attempts to load <cfg.vad.onnx_path> via whichever ONNX importer is
%   available in the current MATLAB release.  In order of preference:
%
%     1. importNetworkFromONNX  (R2024b+)  — returns a dlnetwork that is
%        evaluated with dlarray predict() calls.
%     2. importONNXFunction     (pre-R2024b) — generates a standalone
%        function stub in +silero_vad_imported.
%     3. importONNXNetwork      (deprecated DAGNetwork path).
%     4. Python onnxruntime via the MATLAB py.* bridge — used when none
%        of the above are present (Deep Learning Toolbox Converter for
%        ONNX not installed).  Requires Python + ``pip install onnxruntime
%        numpy`` inside the interpreter MATLAB is configured with.
%
%   Silero v5 is stateful (LSTM), so the hidden state is carried between
%   frames.  If none of the importers can load the model, obj.ready stays
%   false and the dispatcher (vad.m) transparently swaps to VADEnergy.

    properties
        cfg
        ready          = false
        frame_size
        sr
        backend_detail = ''    % human-readable flavour (for logs)
    end

    properties (Access = private)
        backend_tag = ''       % 'dlnet'|'onnxfn'|'dagnet'|'pyort'
        net                    % dlnetwork (backend = 'dlnet')
        net_fn                 % function handle (backend = 'onnxfn')
        dag_net                % DAGNetwork (backend = 'dagnet')
        state                  % single [2 x 1 x 128]
        ctx                    % single [64 x 1] — Silero v5 audio context
        fn_name    = 'silero_vad_net'
        py_helper              % py.module (backend = 'pyort')
        py_path    = ''        % onnx path used for pyort lookup
    end

    properties (Constant, Access = private)
        SILERO_CONTEXT = 64    % v5 expects 64-sample context prepended
    end

    methods
        function obj = VADSilero(cfg)
            obj.cfg        = cfg.vad;
            obj.frame_size = cfg.vad.silero_frame;
            obj.sr         = int64(cfg.fs);
            obj.state      = zeros(2, 1, 128, 'single');
            obj.ctx        = zeros(obj.SILERO_CONTEXT, 1, 'single');

            onnx_path = cfg.vad.onnx_path;
            if ~isfile(onnx_path)
                proj_root = fileparts(fileparts(mfilename('fullpath')));
                onnx_path = fullfile(proj_root, onnx_path);
            end
            if ~isfile(onnx_path)
                warning('QWiSE:VAD:SileroMissing', ...
                    '[Q-WiSE] Silero ONNX not found at %s', onnx_path);
                return;
            end

            if obj.try_import_dlnetwork_(onnx_path),  return; end
            if obj.try_import_onnxfn_(onnx_path),     return; end
            if obj.try_import_dagnetwork_(onnx_path), return; end
            obj.maybe_switch_to_project_venv_();
            if obj.try_import_pyort_(onnx_path),      return; end

            warning('QWiSE:VAD:SileroNoImporter', ...
                ['[Q-WiSE] No ONNX importer could load %s. ' ...
                 'Install the Deep Learning Toolbox Converter for ONNX ' ...
                 '(Add-On Explorer) or ''pip install onnxruntime numpy'' ' ...
                 'in the Python interpreter MATLAB uses (see pyenv). ' ...
                 'Falling back to energy VAD for now.'], onnx_path);
        end

        function [is_speech, score] = step(obj, x)
            if ~obj.ready
                is_speech = false;
                score     = 0;
                return;
            end
            try
                % The host frame (cfg.frame_size, e.g. 1024) is usually
                % a multiple of obj.frame_size (512 for Silero). Process
                % every Silero-sized sub-block and report the latest
                % score; the LSTM state carries through naturally.
                x = double(x(:));
                Nx = numel(x);
                Nf = obj.frame_size;
                if Nx < Nf
                    x(end+1:Nf, 1) = 0;
                    Nx = Nf;
                end
                last_prob = 0;
                for off = 0:Nf:Nx-Nf
                    xf = single(reshape(x(off+1:off+Nf), 1, Nf));
                    last_prob = obj.infer_chunk_(xf);
                end
                score     = double(last_prob);
                is_speech = score > obj.cfg.silero_threshold;
            catch ME
                warning('QWiSE:VAD:SileroStepFailed', ...
                    '[Q-WiSE] Silero step failed: %s (disabling).', ...
                    ME.message);
                obj.ready = false;
                is_speech = false;
                score     = 0;
            end
        end

        function prob = infer_chunk_(obj, xf)
        %INFER_CHUNK_  Dispatch one frame through the active backend.
            switch obj.backend_tag
                case 'dlnet'
                    [prob, obj.state] = obj.run_dlnet_(xf);
                case 'onnxfn'
                    [prob, obj.state] = obj.net_fn(xf, obj.sr, obj.state);
                case 'dagnet'
                    prob = predict(obj.dag_net, xf);
                case 'pyort'
                    [prob, obj.state] = obj.run_pyort_(xf);
                otherwise
                    prob = 0;
            end
            prob = double(prob(1));
        end

        function reset(obj)
            obj.state = zeros(2, 1, 128, 'single');
            obj.ctx   = zeros(obj.SILERO_CONTEXT, 1, 'single');
        end
    end

    methods (Access = private)
        function ok = try_import_dlnetwork_(obj, onnx_path)
        %TRY_IMPORT_DLNETWORK_  Modern R2024b+ API.
            ok = false;
            if exist('importNetworkFromONNX', 'file') ~= 2
                return;
            end
            fmts = { ...
                {'BT', 'B', 'BCT'}, ...
                {'BT', 'B', 'SSC'}, ...
                {'BT', 'B', 'CBT'}, ...
                {'BT', 'B', 'BSC'}  ...
            };
            last_err = '';
            for i = 1:numel(fmts)
                try
                    n = importNetworkFromONNX(onnx_path, ...
                        'InputDataFormats', fmts{i});
                    obj.net            = n;
                    obj.backend_tag    = 'dlnet';
                    obj.backend_detail = sprintf( ...
                        'importNetworkFromONNX (fmt=%s)', ...
                        strjoin(fmts{i}, ','));
                    obj.ready          = true;
                    fprintf('[Q-WiSE] Silero-VAD loaded via %s.\n', ...
                        obj.backend_detail);
                    ok = true;
                    return;
                catch ME
                    last_err = ME.message;
                end
            end
            warning('QWiSE:VAD:SileroDLNetFail', ...
                '[Q-WiSE] importNetworkFromONNX failed: %s', last_err);
        end

        function ok = try_import_onnxfn_(obj, onnx_path)
        %TRY_IMPORT_ONNXFN_  Legacy pre-R2024b API.
            ok = false;
            if exist('importONNXFunction', 'file') ~= 2
                return;
            end
            try
                if exist(obj.fn_name, 'file') ~= 2
                    importONNXFunction(onnx_path, obj.fn_name);
                end
                obj.net_fn         = str2func(obj.fn_name);
                obj.backend_tag    = 'onnxfn';
                obj.backend_detail = 'importONNXFunction';
                obj.ready          = true;
                fprintf('[Q-WiSE] Silero-VAD loaded via %s.\n', ...
                    obj.backend_detail);
                ok = true;
            catch ME
                warning('QWiSE:VAD:SileroFnFail', ...
                    '[Q-WiSE] importONNXFunction failed: %s', ME.message);
            end
        end

        function ok = try_import_dagnetwork_(obj, onnx_path)
        %TRY_IMPORT_DAGNETWORK_  Very-old DAGNetwork API (rarely works).
            ok = false;
            if exist('importONNXNetwork', 'file') ~= 2
                return;
            end
            try
                obj.dag_net        = importONNXNetwork(onnx_path, ...
                    'OutputLayerType', 'regression');
                obj.backend_tag    = 'dagnet';
                obj.backend_detail = 'importONNXNetwork';
                obj.ready          = true;
                fprintf('[Q-WiSE] Silero-VAD loaded via %s.\n', ...
                    obj.backend_detail);
                ok = true;
            catch ME
                warning('QWiSE:VAD:SileroDagFail', ...
                    '[Q-WiSE] importONNXNetwork failed: %s', ME.message);
            end
        end

        function maybe_switch_to_project_venv_(~)
        %MAYBE_SWITCH_TO_PROJECT_VENV_  Best-effort pyenv autoswitch.
        %   If the project ships a .pyenv venv (created by
        %   setup_silero_python.m) and MATLAB hasn't loaded any Python
        %   yet this session, point pyenv at it transparently so the
        %   user doesn't have to remember to run the setup script.
            try
                proj  = fileparts(fileparts(mfilename('fullpath')));
                vpy   = fullfile(proj, '.pyenv', 'bin', 'python');
                if ~isfile(vpy), return; end
                e = pyenv();
                if strcmp(e.Executable, vpy), return; end
                if strcmpi(char(e.Status), 'Loaded')
                    % Can't switch mid-session; warn once with the recipe.
                    warning('QWiSE:VAD:SileroPyEnvLocked', ...
                        ['[Q-WiSE] Python is already loaded as %s. ' ...
                         'Run ''setup_silero_python'' as the FIRST ' ...
                         'command after restarting MATLAB to switch ' ...
                         'to the project venv (%s) and enable Silero.'], ...
                        e.Executable, vpy);
                    return;
                end
                pyenv('Version', vpy, 'ExecutionMode', 'OutOfProcess');
                fprintf('[Q-WiSE] Switched pyenv to project venv: %s\n', vpy);
            catch ME
                warning('QWiSE:VAD:SileroPyEnvAutoFail', ...
                    '[Q-WiSE] Could not auto-switch pyenv: %s', ME.message);
            end
        end

        function ok = try_import_pyort_(obj, onnx_path)
        %TRY_IMPORT_PYORT_  Fallback: run the model via Python onnxruntime.
        %   Works on any MATLAB release that has a working py.* bridge
        %   (R2019b+) provided ``numpy`` and ``onnxruntime`` are installed
        %   in the Python environment reported by ``pyenv``.
            ok = false;
            try
                py.importlib.import_module('numpy');
                py.importlib.import_module('onnxruntime');
            catch ME
                warning('QWiSE:VAD:SileroPyMissing', ...
                    ['[Q-WiSE] Python onnxruntime backend unavailable ' ...
                     '(%s). Install with: pyenv -> pip install ' ...
                     'onnxruntime numpy.'], ME.message);
                return;
            end
            try
                helper_dir = fileparts(mfilename('fullpath'));
                sys_mod = py.importlib.import_module('sys');
                if count(sys_mod.path, helper_dir) == 0
                    sys_mod.path.insert(int32(0), helper_dir);
                end
                helper = py.importlib.import_module('silero_ort_helper');
                info = helper.load(onnx_path);
                in_names = cellfun(@char, cell(info{'input_names'}), ...
                                   'UniformOutput', false);
                obj.py_helper      = helper;
                obj.py_path        = onnx_path;
                obj.backend_tag    = 'pyort';
                obj.backend_detail = sprintf('onnxruntime [%s]', ...
                    strjoin(in_names, ','));
                obj.ready          = true;
                fprintf(['[Q-WiSE] Silero-VAD loaded via Python ' ...
                         '%s.\n'], obj.backend_detail);
                ok = true;
            catch ME
                warning('QWiSE:VAD:SileroPyORTFail', ...
                    '[Q-WiSE] Python onnxruntime load failed: %s', ...
                    ME.message);
            end
        end

        function [prob, new_state] = run_pyort_(obj, xf)
        %RUN_PYORT_  One inference step through the Python helper.
            % Silero v5 expects 64 samples of audio context prepended
            % to the new 512-sample chunk; obj.ctx tracks that context
            % across calls. MATLAB numeric arrays auto-convert to numpy
            % arrays via the py.* bridge in R2023b+; numpy handles the
            % reshape on the Python side, so we hand over flat vectors.
            x_vec  = double(xf(:));
            st_vec = double(obj.state(:));
            cx_vec = double(obj.ctx(:));
            result = obj.py_helper.step(obj.py_path, x_vec, st_vec, ...
                                        cx_vec, int64(obj.sr));
            prob      = double(result{'prob'});
            st_flat   = double(py.array.array('d', result{'state'}));
            new_state = single(reshape(st_flat, 2, 1, 128));
            cx_flat   = double(py.array.array('d', result{'context'}));
            obj.ctx   = single(cx_flat(:));
        end

        function [prob, new_state] = run_dlnet_(obj, xf)
        %RUN_DLNET_  Multi-input dlnetwork inference for Silero v5.
            x_dl  = dlarray(xf,        'BT');
            sr_dl = dlarray(single(obj.sr), 'B');
            st_dl = dlarray(obj.state, 'BCT');
            try
                [p_dl, s_dl] = predict(obj.net, x_dl, sr_dl, st_dl);
            catch
                % Some imports swap the input ordering; try again.
                [p_dl, s_dl] = predict(obj.net, x_dl, st_dl, sr_dl);
            end
            prob      = extractdata(p_dl);
            new_state = extractdata(s_dl);
            if ~isequal(size(new_state), [2 1 128])
                new_state = reshape(new_state, 2, 1, 128);
            end
        end
    end
end
