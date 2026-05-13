classdef VADEnergy < handle
%VADENERGY  Statistical fallback VAD (energy + spectral flatness).
%
%   Lightweight, stateless-per-frame VAD that compares short-time energy
%   (dBFS) and spectral flatness against thresholds from cfg.vad.* and
%   smooths the output with a hangover release counter.  Used whenever
%   the Q-WiSE neural VAD backend cannot be loaded.

    properties
        cfg
        ready = true
    end

    properties (Access = private)
        hang_counter = 0
        state_speech = false
        score_ema    = 0
        smooth
    end

    methods
        function obj = VADEnergy(cfg)
            obj.cfg    = cfg.vad;
            obj.smooth = cfg.vad.smoothing;
        end

        function [is_speech, score] = step(obj, x)
            % x: mono frame (column vector)
            rmsv = rms(x) + 1e-12;
            dB   = 20 * log10(rmsv);

            X      = abs(fft(x));
            X      = X(1:floor(numel(X)/2));
            gmean  = exp(mean(log(X + 1e-12)));
            amean  = mean(X) + 1e-12;
            sfm    = gmean / amean;

            raw = double(dB > obj.cfg.energy_threshold && ...
                         sfm < obj.cfg.sfm_threshold);

            % Score EMA for a smoother visualisation trace.
            obj.score_ema = (1-obj.smooth)*obj.score_ema + obj.smooth*raw;
            score = obj.score_ema;

            if raw > 0.5
                obj.state_speech = true;
                obj.hang_counter = obj.cfg.hang_frames;
            elseif obj.hang_counter > 0
                obj.hang_counter = obj.hang_counter - 1;
                obj.state_speech = true;
            else
                obj.state_speech = false;
            end
            is_speech = obj.state_speech;
        end

        function reset(obj)
            obj.hang_counter = 0;
            obj.state_speech = false;
            obj.score_ema    = 0;
        end
    end
end
