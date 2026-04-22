classdef VADSilero < handle
%VADSILERO  Silero-VAD ONNX wrapper with graceful failure.
%
%   Tries to load <cfg.vad.onnx_path> via importONNXFunction() from the
%   Deep Learning Toolbox, then exposes a per-frame step() that returns
%   a speech probability in [0,1].  Silero is stateful (LSTM), so the
%   hidden state is carried between calls.  If the model cannot be
%   imported (common when ONNX opsets include dynamic-shape LSTMs not
%   supported by importONNXFunction), obj.ready stays false and the
%   caller should use VADEnergy instead.

    properties
        cfg
        ready      = false
        frame_size
        sr
    end

    properties (Access = private)
        net_fn
        state
        fn_name    = 'silero_vad_net'
    end

    methods
        function obj = VADSilero(cfg)
            obj.cfg        = cfg.vad;
            obj.frame_size = cfg.vad.silero_frame;
            obj.sr         = int64(cfg.fs);
            obj.state      = zeros(2, 1, 128, 'single');

            onnx_path = cfg.vad.onnx_path;
            if ~isfile(onnx_path)
                proj_root = fileparts(fileparts(mfilename('fullpath')));
                onnx_path = fullfile(proj_root, onnx_path);
            end
            if ~isfile(onnx_path)
                warning('[Q-WiSE] Silero ONNX not found at %s', onnx_path);
                return;
            end
            if exist('importONNXFunction', 'file') ~= 2
                warning(['[Q-WiSE] importONNXFunction not available ' ...
                         '(Deep Learning Toolbox missing).']);
                return;
            end

            try
                if exist(obj.fn_name, 'file') ~= 2
                    importONNXFunction(onnx_path, obj.fn_name);
                end
                obj.net_fn = str2func(obj.fn_name);
                obj.ready  = true;
                fprintf('[Q-WiSE] Silero-VAD ONNX loaded (%s).\n', onnx_path);
            catch ME
                warning('QWiSE:VAD:SileroImportFailed', ...
                    '[Q-WiSE] Silero ONNX import failed: %s', ME.message);
                obj.ready = false;
            end
        end

        function [is_speech, score] = step(obj, x)
            if ~obj.ready
                is_speech = false;
                score     = 0;
                return;
            end
            try
                L  = min(numel(x), obj.frame_size);
                xf = single(reshape(x(1:L), 1, L));
                if L < obj.frame_size
                    xf(1, end+1:obj.frame_size) = 0;
                end
                [prob, obj.state] = obj.net_fn(xf, obj.sr, obj.state);
                score     = double(prob(1));
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

        function reset(obj)
            obj.state = zeros(2, 1, 128, 'single');
        end
    end
end
