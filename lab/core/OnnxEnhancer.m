classdef OnnxEnhancer < handle
%ONNXENHANCER  Run the self-contained Q-WiSE ONNX from MATLAB via Python.
%
%   The model is the whole pipeline:  mic[M, L] (raw 16 kHz waveform)
%   -> clean[L].  Inference runs through onnxruntime, called via MATLAB's
%   Python bridge and the qwise_ort_helper.py module (which keeps all the
%   numpy/dtype handling on the Python side).
%
%   Setup is handled by lab/run_qwise.sh: it creates a venv at lab/.pyenv
%   with onnxruntime + numpy, and run_simulation.m points MATLAB's pyenv
%   at that interpreter.
%
%   Usage:
%       enh   = OnnxEnhancer(cfg.onnx_path, cfg.lab_root);
%       clean = enh.enhance(mic);     % mic = [L x N], clean = [L x 1]

    properties
        model_path
        helper_dir        % folder containing qwise_ort_helper.py (lab root)
    end

    properties (Access = private)
        helper_ = []      % cached py module handle
    end

    methods
        function obj = OnnxEnhancer(model_path, helper_dir)
            if nargin < 1 || isempty(model_path)
                error('OnnxEnhancer:NoPath', 'Provide a path to the ONNX file.');
            end
            if ~isfile(model_path)
                error('OnnxEnhancer:NotFound', 'ONNX model not found:\n  %s', ...
                      model_path);
            end
            obj.model_path = model_path;
            if nargin < 2 || isempty(helper_dir)
                helper_dir = fileparts(fileparts(mfilename('fullpath'))); % lab/
            end
            obj.helper_dir = helper_dir;
        end

        function ready = is_ready(obj)
            [ready, ~] = obj.check();
        end

        function [ready, msg] = check(obj)
        %CHECK  Try to import the helper + load the model; return readiness
        %   and a diagnostic message (empty when ready).
            ready = false;  msg = '';
            try
                obj.ensure_helper_();
                ready = ~isempty(obj.helper_);
            catch ME
                msg = ME.message;
            end
        end

        function clean = enhance(obj, mic)
        %ENHANCE  Denoise an N-mic noisy array.
        %   mic   : [L x N] (samples x microphones)
        %   clean : [L x 1] enhanced mono speech
            if isempty(mic)
                clean = zeros(0, 1);
                return;
            end
            obj.ensure_helper_();
            [L, N] = size(mic);

            % Column-major flatten (MATLAB mic(:)); the helper reshapes it
            % back with order='F' and transposes to [M, L].
            pa  = py.array.array('d', mic(:));
            res = obj.helper_.enhance(obj.model_path, pa, int64(L), int64(N));

            clean = double(res);
            clean = clean(:);
            if numel(clean) >= L
                clean = clean(1:L);
            end
        end
    end

    methods (Access = private)
        function ensure_helper_(obj)
        %ENSURE_HELPER_  Make sure MATLAB has a usable Python, import the
        %   qwise_ort_helper module, and pre-load the ORT session.
            if ~isempty(obj.helper_), return; end

            % --- verify a Python is configured -------------------------
            try
                pe = pyenv;
            catch
                pe = struct('Version', '', 'Executable', '');
            end
            if strlength(string(pe.Version)) == 0
                error('OnnxEnhancer:NoPython', ...
                    ['MATLAB has no Python configured. Run lab/run_qwise.sh ' ...
                     '(creates lab/.pyenv) or set:\n' ...
                     '  pyenv(''Version'', ''%s/.pyenv/bin/python3'')'], ...
                    obj.helper_dir);
            end

            % --- make the helper importable ----------------------------
            P = py.sys.path;
            found = false;
            Pc = cell(P);
            for i = 1:numel(Pc)
                if strcmp(char(Pc{i}), obj.helper_dir), found = true; break; end
            end
            if ~found
                insert(P, int32(0), obj.helper_dir);
            end

            % --- import + load the session -----------------------------
            try
                obj.helper_ = py.importlib.import_module('qwise_ort_helper');
            catch ME
                error('OnnxEnhancer:HelperImport', ...
                    ['Could not import qwise_ort_helper from\n  %s\n' ...
                     'using Python: %s\n' ...
                     'Make sure onnxruntime + numpy are installed for that ' ...
                     'interpreter (run lab/run_qwise.sh).\nUnderlying error: %s'], ...
                    obj.helper_dir, char(pe.Executable), ME.message);
            end
            obj.helper_.load(obj.model_path);
        end
    end
end
