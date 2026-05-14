classdef SourceMixer < handle
%SOURCEMIXER  Physical multi-source → N-microphone array simulator.
%
%   Single physical mixing model (mixer_1.m-style): every microphone
%   receives every source. Per-source per-mic propagation applies a
%   fractional sample delay (linear interpolation) and a 1/r spreading
%   gain clamped at cfg.distance_ref:
%
%       mic_m[n] = sum over s ∈ {speech, drone, env}
%                      gain_s(m) · src_s[n - frac_delay_s(m)]
%
%   This wiring matches what the rest of the Q-WiSE pipeline expects:
%     * the VAD consumes a single mono signal (composite — typically the
%       reference mic), and
%     * the MWF consumes the full N-channel block returned by mix().
%
%   Fractional delays that exceed one block are handled with a per-source
%   history ring buffer, so block boundaries do not click.
%
%   Interface:
%       obj  = SourceMixer(cfg, geo)
%       mic  = obj.mix(speech, drone, env)         % [N x n_mics]
%       comp = obj.composite(mic)                  % [N x 1]    (VAD feed)
%       obj.reset()
%
%   The `composite` reduction (default = reference mic) is controlled by
%   cfg.mixer.composite ∈ {'mic1', 'sum', 'mean'}.

    properties
        cfg
        geo
        n_mics
        composite_kind         % 'mic1' | 'sum' | 'mean'
        mode = 'physical'      % retained for UI / status display
        hist_len               % history ring length in samples
    end

    properties (Access = private)
        hist_speech            % [hist_len x 1]
        hist_drone             % [hist_len x 1]
        hist_env               % [hist_len x 1]
        ref_mic_idx
        frac_delays_speech_
        frac_delays_drone_
        frac_delays_env_
        gains_speech_
        gains_drone_
        gains_env_
    end

    methods
        function obj = SourceMixer(cfg, geo)
            obj.cfg    = cfg;
            obj.geo    = geo;
            obj.n_mics = cfg.n_mics;

            obj.composite_kind = 'mic1';
            if isfield(cfg, 'mixer') && isfield(cfg.mixer, 'composite')
                obj.composite_kind = cfg.mixer.composite;
            end

            obj.ref_mic_idx = obj.resolve_ref_mic_();

            % Cache delays and gains as columns once — read every block.
            obj.frac_delays_speech_ = obj.fetch_frac_delays_(geo, 'speech');
            obj.frac_delays_drone_  = obj.fetch_frac_delays_(geo, 'drone');
            obj.frac_delays_env_    = obj.fetch_frac_delays_(geo, 'env');
            obj.gains_speech_       = geo.gains_speech(:);
            obj.gains_drone_        = geo.gains_drone(:);
            obj.gains_env_          = geo.gains_env(:);

            tau_max = max([max(obj.frac_delays_speech_), ...
                           max(obj.frac_delays_drone_),  ...
                           max(obj.frac_delays_env_),    ...
                           0]);
            % Add one extra sample for the linear-interp right tap.
            obj.hist_len = ceil(tau_max) + 1;

            obj.hist_speech = zeros(obj.hist_len, 1);
            obj.hist_drone  = zeros(obj.hist_len, 1);
            obj.hist_env    = zeros(obj.hist_len, 1);
        end

        function mic = mix(obj, speech, drone, env)
        %MIX  Run one block through the physical multi-source mixer.
        %   Inputs are coerced to column vectors and padded with zeros to
        %   the longest of the three. Output is [N x n_mics] where every
        %   mic carries speech + drone + env with its own TDOA + 1/r gain.
            [speech, drone, env, N] = obj.coerce_sources_(speech, drone, env);

            % Build concatenated [history; current] tapes for each source.
            sp_tape = [obj.hist_speech; speech];
            dr_tape = [obj.hist_drone;  drone];
            en_tape = [obj.hist_env;    env];
            base    = obj.hist_len;                 % index of sample-0-of-block - 1

            mic = zeros(N, obj.n_mics);
            n_idx = (1:N).';
            for m = 1:obj.n_mics
                s_q = (base + n_idx) - obj.frac_delays_speech_(m);
                d_q = (base + n_idx) - obj.frac_delays_drone_(m);
                e_q = (base + n_idx) - obj.frac_delays_env_(m);

                s = obj.frac_tap_(sp_tape, s_q);
                d = obj.frac_tap_(dr_tape, d_q);
                e = obj.frac_tap_(en_tape, e_q);

                mic(:, m) = obj.gains_speech_(m) * s + ...
                            obj.gains_drone_(m)  * d + ...
                            obj.gains_env_(m)    * e;
            end

            obj.hist_speech = obj.push_(obj.hist_speech, speech);
            obj.hist_drone  = obj.push_(obj.hist_drone,  drone);
            obj.hist_env    = obj.push_(obj.hist_env,    env);
        end

        function y = composite(obj, mic)
        %COMPOSITE  Reduce the N-channel mix to one mono signal for the VAD.
        %   The default 'mic1' returns the reference microphone (the
        %   channel the VAD treats as its input). 'sum' and 'mean' are
        %   kept for diagnostics.
            if isempty(mic)
                y = zeros(obj.cfg.frame_size, 1);
                return;
            end
            switch lower(obj.composite_kind)
                case 'mean', y = mean(mic, 2);
                case 'sum',  y = sum(mic, 2);
                otherwise                                    % 'mic1' / ref mic
                    ref = min(obj.ref_mic_idx, size(mic, 2));
                    y = mic(:, ref);
            end
        end

        function reset(obj)
            obj.hist_speech(:) = 0;
            obj.hist_drone(:)  = 0;
            obj.hist_env(:)    = 0;
        end

        function ref = ref_mic(obj)
            ref = obj.ref_mic_idx;
        end

        function update_geometry(obj, new_geo)
        %UPDATE_GEOMETRY  Swap in a new geometry struct at runtime.
        %   Re-caches per-source fractional delays and 1/r gains and
        %   resizes the per-source history rings, preserving the tail so
        %   the next mix() call does not click on the boundary.
        %
        %   Used by the GUI scene sliders (human height, slant distance,
        %   azimuths, etc.). The mic count must NOT change at runtime
        %   because that would invalidate downstream buffer shapes.
            if numel(new_geo.gains_speech) ~= obj.n_mics
                error('SourceMixer:update_geometry:NMicChange', ...
                      'Cannot change n_mics at runtime (have %d, got %d).', ...
                      obj.n_mics, numel(new_geo.gains_speech));
            end

            obj.geo                 = new_geo;
            obj.frac_delays_speech_ = obj.fetch_frac_delays_(new_geo, 'speech');
            obj.frac_delays_drone_  = obj.fetch_frac_delays_(new_geo, 'drone');
            obj.frac_delays_env_    = obj.fetch_frac_delays_(new_geo, 'env');
            obj.gains_speech_       = new_geo.gains_speech(:);
            obj.gains_drone_        = new_geo.gains_drone(:);
            obj.gains_env_          = new_geo.gains_env(:);

            tau_max     = max([max(obj.frac_delays_speech_), ...
                               max(obj.frac_delays_drone_),  ...
                               max(obj.frac_delays_env_),    0]);
            new_hist_len = max(1, ceil(tau_max) + 1);

            if new_hist_len > obj.hist_len
                pad = new_hist_len - obj.hist_len;
                obj.hist_speech = [zeros(pad, 1); obj.hist_speech];
                obj.hist_drone  = [zeros(pad, 1); obj.hist_drone];
                obj.hist_env    = [zeros(pad, 1); obj.hist_env];
            elseif new_hist_len < obj.hist_len
                obj.hist_speech = obj.hist_speech(end - new_hist_len + 1:end);
                obj.hist_drone  = obj.hist_drone (end - new_hist_len + 1:end);
                obj.hist_env    = obj.hist_env   (end - new_hist_len + 1:end);
            end
            obj.hist_len = new_hist_len;
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

        function y = frac_tap_(~, tape, q)
        %FRAC_TAP_  Linear-interpolated tap into a [history; current] tape.
        %   q is a column of query indices (1-based). Anything outside the
        %   tape reads as zero (no aliasing from the end). The tape is
        %   tall enough that q is always > 0 here (TDOA referenced from
        %   min == 0 keeps the leading samples in-block).
            L  = numel(tape);
            q0 = floor(q);
            qf = q - q0;
            in = (q0 >= 1) & (q0 + 1 <= L);
            y  = zeros(numel(q), 1);
            % Linear interpolation between the two nearest tape samples.
            idx_a = max(min(q0,     L), 1);
            idx_b = max(min(q0 + 1, L), 1);
            interp = (1 - qf) .* tape(idx_a) + qf .* tape(idx_b);
            y(in) = interp(in);
            % Tail samples that miss the tape stay at zero (silence).
            unused = ~in; %#ok<NASGU>  % kept for future debugging hooks
        end

        function h = push_(obj, h, src)
        %PUSH_  Slide the history ring left, append `src` on the right.
            N = numel(src);
            if N >= obj.hist_len
                h = src(end-obj.hist_len+1:end);
            else
                h = [h(N+1:end); src(:)];
            end
        end

        function tau = fetch_frac_delays_(~, geo, which)
        %FETCH_FRAC_DELAYS_  Read fractional delays for one source; fall
        %   back to the integer field if a caller passed an older geo
        %   struct without frac_delays_*.
            name = ['frac_delays_' which];
            if isfield(geo, name)
                tau = geo.(name)(:);
                return;
            end
            tau = double(geo.(['delays_' which])(:));
        end

        function ref = resolve_ref_mic_(obj)
            ref = 1;
            if isfield(obj.cfg, 'mwf') && isfield(obj.cfg.mwf, 'ref_mic')
                ref = obj.cfg.mwf.ref_mic;
            end
            ref = max(1, min(ref, obj.n_mics));
        end
    end
end
