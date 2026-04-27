classdef vad < handle
%VAD  Q-WiSE VAD dispatcher: ONNX neural backend + energy fallback.
%
%   obj = vad(cfg)
%
%   Selects a backend according to cfg.vad.backend:
%       'silero' : force the neural ONNX backend (falls back if it
%                  cannot be loaded)
%       'energy' : force the statistical fallback
%       'auto'   : try the neural backend, else fall back   <-- default
%
%   Each call to step(x) returns a logical is_speech decision and a
%   soft score in [0,1] that the UI plots as a VAD trace.  A ring
%   buffer of recent scores is kept in obj.history for visualisation.

    properties
        cfg
        backend            % concrete VAD backend instance
        backend_name       % 'qwise-vad' | 'energy'
        history            % [hist_len x 1] ring buffer of scores
        flags              % [hist_len x 1] ring buffer of bool decisions
        hist_len
    end

    properties (Access = private)
        hist_ptr = 1
    end

    methods
        function obj = vad(cfg)
            obj.cfg      = cfg;
            obj.hist_len = max(8, round(cfg.ui.vad_hist_sec * cfg.fs ...
                                        / cfg.frame_size));
            obj.history  = zeros(obj.hist_len, 1);
            obj.flags    = false(obj.hist_len, 1);

            switch lower(cfg.vad.backend)
                case 'silero'
                    s = VADSilero(cfg);
                    if s.ready
                        obj.backend = s;
                        obj.backend_name = 'qwise-vad';
                    else
                        warning(['[Q-WiSE] Neural VAD unavailable; ' ...
                                 'falling back to energy VAD.']);
                        obj.backend = VADEnergy(cfg);
                        obj.backend_name = 'energy';
                    end
                case 'energy'
                    obj.backend = VADEnergy(cfg);
                    obj.backend_name = 'energy';
                otherwise   % 'auto'
                    s = VADSilero(cfg);
                    if s.ready
                        obj.backend = s;
                        obj.backend_name = 'qwise-vad';
                    else
                        obj.backend = VADEnergy(cfg);
                        obj.backend_name = 'energy';
                    end
            end
            fprintf('[Q-WiSE] VAD backend: %s\n', obj.backend_name);
        end

        function [is_speech, score] = step(obj, x)
            [is_speech, score] = obj.backend.step(x);
            obj.history(obj.hist_ptr) = score;
            obj.flags(obj.hist_ptr)   = is_speech;
            obj.hist_ptr = mod(obj.hist_ptr, obj.hist_len) + 1;
        end

        function [scores, flags] = trace(obj)
            % Return history in chronological order (oldest first).
            idx    = mod(obj.hist_ptr - 1 + (0:obj.hist_len-1), obj.hist_len) + 1;
            scores = obj.history(idx);
            flags  = obj.flags(idx);
        end

        function reset(obj)
            obj.history(:) = 0;
            obj.flags(:)   = false;
            obj.hist_ptr   = 1;
            if ismethod(obj.backend, 'reset'), obj.backend.reset(); end
        end
    end
end
