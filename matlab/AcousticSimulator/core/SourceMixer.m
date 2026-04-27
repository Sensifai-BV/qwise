classdef SourceMixer < handle
%SOURCEMIXER  Three-source mixer supporting per-channel or physical wiring.
%
%   The Q-WiSE scene has three acoustic sources:
%       1. Human speech    (live MacBook-mic capture)
%       2. Drone-fan noise (wav loop)
%       3. Environment     (wav loop)
%
%   Two wiring modes are offered via cfg.mixer.mode:
%
%   'perChannel'  (demo/default)
%       mic-1  = speech + drone + env   (laptop mic captures the full
%                                        scene — the realistic noisy
%                                        speech the user actually hears)
%       mic-2  = drone   (virtual drone-side reference, noise-only)
%       mic-3  = env     (virtual env-side reference,  noise-only)
%       extra mics replicate mic-1 with per-mic speech TDOA.
%       composite = mic-1 by default (already a complete noisy mix).
%       'sum' and 'mean' remain available for legacy diagnostics.
%
%   'physical'   (research-grade)
%       every mic receives every source with per-source integer-sample
%       TDOA and a 1/d spreading gain clamped at cfg.distance_ref.
%       composite = reference-mic channel.
%
%   Interface:
%       obj  = SourceMixer(cfg, geo)
%       mic  = obj.mix(speech, drone, env)         % [N x n_mics]
%       comp = obj.composite(mic)                  % [N x 1]
%
%   History buffers are carried across calls so TDOAs that exceed one
%   block do not produce clicks.

    properties
        cfg
        geo
        n_mics
        mode              % 'perChannel' | 'physical'
        composite_kind    % 'mic1' | 'sum' | 'mean'
        hist_len          % ring-buffer length in samples
    end

    properties (Access = private)
        hist_speech       % [hist_len x 1]
        hist_drone        % [hist_len x 1]
        hist_env          % [hist_len x 1]
    end

    methods
        function obj = SourceMixer(cfg, geo)
            obj.cfg    = cfg;
            obj.geo    = geo;
            obj.n_mics = cfg.n_mics;

            obj.mode           = 'perChannel';
            obj.composite_kind = 'mic1';
            if isfield(cfg, 'mixer')
                if isfield(cfg.mixer, 'mode'), obj.mode = cfg.mixer.mode; end
                if isfield(cfg.mixer, 'composite')
                    obj.composite_kind = cfg.mixer.composite;
                end
            end

            tau_max = max([max(geo.delays_speech), ...
                           max(geo.delays_drone),  ...
                           max(geo.delays_env)]);
            obj.hist_len = max(tau_max, 1) + cfg.frame_size;

            obj.hist_speech = zeros(obj.hist_len, 1);
            obj.hist_drone  = zeros(obj.hist_len, 1);
            obj.hist_env    = zeros(obj.hist_len, 1);
        end

        function mic = mix(obj, speech, drone, env)
        %MIX  Combine three sources into an n_mics-channel frame.
            [speech, drone, env, N] = obj.coerce_sources_(speech, drone, env);
            switch lower(obj.mode)
                case 'perchannel'
                    mic = obj.mix_per_channel_(speech, drone, env, N);
                otherwise
                    mic = obj.mix_physical_(speech, drone, env, N);
            end

            obj.hist_speech = obj.push_(obj.hist_speech, speech);
            obj.hist_drone  = obj.push_(obj.hist_drone,  drone);
            obj.hist_env    = obj.push_(obj.hist_env,    env);
        end

        function y = composite(obj, mic)
        %COMPOSITE  Reduce the multi-channel mix down to one mono VAD feed.
        %   In perChannel mode mic-1 already carries the realistic noisy
        %   speech, so 'mic1' is the sensible default; summing or meaning
        %   would double-count noise (mic-2 and mic-3 are noise-only).
            if isempty(mic)
                y = zeros(obj.cfg.frame_size, 1);
                return;
            end
            switch lower(obj.composite_kind)
                case 'mean',  y = mean(mic, 2);
                case 'sum',   y = sum(mic, 2);
                otherwise,    y = mic(:, 1);   % default: 'mic1'
            end
        end

        function reset(obj)
            obj.hist_speech(:) = 0;
            obj.hist_drone(:)  = 0;
            obj.hist_env(:)    = 0;
        end
    end

    methods (Access = private)
        function [speech, drone, env, N] = coerce_sources_(~, speech, drone, env)
            speech = speech(:);
            drone  = drone(:);
            env    = env(:);
            N = max([numel(speech), numel(drone), numel(env)]);
            if numel(speech) < N, speech(end+1:N,1) = 0; end
            if numel(drone)  < N, drone(end+1:N,1)  = 0; end
            if numel(env)    < N, env(end+1:N,1)    = 0; end
        end

        function mic = mix_per_channel_(obj, speech, drone, env, N)
        %MIX_PER_CHANNEL_  mic-1 captures the whole scene; mic-2/3 are
        %   noise-only references, which is the configuration the MWF
        %   needs (one noisy ref + dedicated noise observations).
        %       mic-1 = g_d * speech + drone + env  (laptop mic, outdoor)
        %       mic-2 = drone
        %       mic-3 = env
        %   The speech term on mic-1 is attenuated by distance_to_gain
        %   evaluated at the human-to-ref-mic distance, so the simulator
        %   behaves like a real outdoor scene: as the human moves farther
        %   from the drone the captured speech level drops in steps
        %   (see core/distance_to_gain.m for the band table). Drone and
        %   environment channels are NOT touched by the speech-distance
        %   gain — they keep their user-slider levels and remain valid
        %   noise-only references for the MWF.
        %
        %   Extra mics (n_mics > 3) replay the noisy speech with per-mic
        %   speech TDOA so a larger ULA can still be exercised.
            mic = zeros(N, obj.n_mics);
            ref = obj.ref_mic_();
            g_speech = distance_to_gain(obj.geo.dist_speech(ref));
            full = g_speech * speech + drone + env;     % outdoor noisy mic
            if obj.n_mics >= 1
                mic(:, 1) = full;
            end
            if obj.n_mics >= 2
                mic(:, 2) = drone;
            end
            if obj.n_mics >= 3
                mic(:, 3) = env;
            end
            for m = 4:obj.n_mics
                mic(:, m) = obj.tap_(obj.hist_speech, full, ...
                                     obj.geo.delays_speech(m));
            end
        end

        function mic = mix_physical_(obj, speech, drone, env, N)
        %MIX_PHYSICAL_  Each mic receives all 3 sources (TDOA + 1/d gain).
            mic = zeros(N, obj.n_mics);
            for m = 1:obj.n_mics
                s = obj.tap_(obj.hist_speech, speech, obj.geo.delays_speech(m));
                d = obj.tap_(obj.hist_drone,  drone,  obj.geo.delays_drone(m));
                e = obj.tap_(obj.hist_env,    env,    obj.geo.delays_env(m));
                mic(:, m) = obj.geo.gains_speech(m) * s ...
                          + obj.geo.gains_drone(m)  * d ...
                          + obj.geo.gains_env(m)    * e;
            end
        end
    end

    methods (Access = private)
        function ref = ref_mic_(obj)
        %REF_MIC_  Index of the reference (laptop) microphone — the
        %   channel that carries the realistic noisy speech in
        %   perChannel mode. Defaults to 1 if cfg.mwf.ref_mic is absent.
            ref = 1;
            if isfield(obj.cfg, 'mwf') && isfield(obj.cfg.mwf, 'ref_mic')
                ref = obj.cfg.mwf.ref_mic;
            end
        end

        function y = tap_(obj, hist, src, tau)
        %TAP_  Produce a length-N block delayed by TAU samples.
        %   Samples older than N come from the history ring; newer ones
        %   come from the current block.  hist is kept newest-last.
            src = src(:);
            N   = numel(src);
            tau = double(tau);
            if tau <= 0
                y = src;
                return;
            end
            if tau >= N
                % Entire output pulled from history.
                idx = (obj.hist_len - tau + 1):(obj.hist_len - tau + N);
            else
                % Mix: leading samples from history, trailing from src.
                idx_hist = (obj.hist_len - tau + 1):obj.hist_len;
                y = [hist(idx_hist); src(1:N-tau)];
                return;
            end
            y = hist(idx);
        end

        function h = push_(obj, h, src)
        %PUSH_  Advance the history ring, keeping the newest hist_len samples.
            N = numel(src);
            if N >= obj.hist_len
                h = src(end-obj.hist_len+1:end);
            else
                h = [h(N+1:end); src(:)];
            end
        end
    end
end
