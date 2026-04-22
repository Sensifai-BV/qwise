classdef mwf < handle
%MWF  Neural-Guided Multi-Channel Wiener Filter (scaffolded / pass-through).
%
%   This is the real-time MWF module for the Q-WiSE framework.  The
%   current implementation is deliberately a PASS-THROUGH (returns the
%   reference mic) but it already carries all of the plumbing needed
%   to drop in a full Rank-1 / SDW-MWF:
%
%     * STFT analysis / synthesis buffers with square-root Hann COLA
%     * Running noise covariance   Rnn[k]     (EMA, updated when !speech)
%     * Running speech covariance  Rss[k]     (EMA, updated when  speech)
%     * Neural-guided gain-map hook get_tf_gain_map()
%
%   Interface:
%       obj = mwf(cfg)
%       y   = obj.step(mic_block, is_speech)   % mic_block: [N x n_mics]
%
%   When cfg.mwf.passthrough is false the scaffold computes an SDW-MWF
%   weight vector per bin and returns the enhanced reference-mic output.

    properties
        cfg
        win         % analysis/synthesis window (sqrt-hann)
        Lwin
        hop
        nbin
        n_mics
        ref_mic
        mu
        eps_reg
        alpha_nn
        alpha_ss
        Rnn         % [nbin x n_mics x n_mics]
        Rss         % [nbin x n_mics x n_mics]
    end

    properties (Access = private)
        in_buf      % [Lwin x n_mics] analysis ring
        ola_buf     % [Lwin x 1]      output overlap-add tail
        out_queue   % queued enhanced samples awaiting return
    end

    methods
        function obj = mwf(cfg)
            obj.cfg      = cfg;
            obj.Lwin     = cfg.mwf.stft_win;
            obj.hop      = cfg.mwf.stft_hop;
            obj.nbin     = obj.Lwin/2 + 1;
            obj.n_mics   = cfg.n_mics;
            obj.ref_mic  = cfg.mwf.ref_mic;
            obj.mu       = cfg.mwf.mu;
            obj.eps_reg  = cfg.mwf.eps_reg;
            obj.alpha_nn = cfg.mwf.alpha_nn;
            obj.alpha_ss = cfg.mwf.alpha_ss;
            obj.win      = sqrt(hann(obj.Lwin, 'periodic'));

            I = eye(obj.n_mics);
            R0 = reshape(I, 1, obj.n_mics, obj.n_mics) * obj.eps_reg;
            obj.Rnn = repmat(R0, obj.nbin, 1, 1);
            obj.Rss = repmat(R0, obj.nbin, 1, 1);

            obj.in_buf    = zeros(obj.Lwin, obj.n_mics);
            obj.ola_buf   = zeros(obj.Lwin, 1);
            obj.out_queue = [];
        end

        function y = step(obj, x, is_speech)
        %STEP  Process one mic block and return an equally-sized output.
        %
        %   x          [N x n_mics]  time-domain mic block
        %   is_speech  logical       VAD decision for this block

            N = size(x, 1);

            % --- Fast path: pure pass-through -----------------------
            if obj.cfg.mwf.passthrough
                y = x(:, obj.ref_mic);
                return;
            end

            % --- Scaffolded STFT-domain path ------------------------
            y = zeros(N, 1);
            L = obj.Lwin;  H = obj.hop;
            cursor = 1;
            while cursor + H - 1 <= N
                seg = x(cursor:cursor+H-1, :);
                obj.in_buf = [obj.in_buf(H+1:end, :); seg];
                % Only spin the analysis when the buffer is primed.
                if any(obj.in_buf(:))
                    Xf = fft(obj.in_buf .* obj.win, L);
                    Xf = Xf(1:obj.nbin, :);
                    obj.update_covariances_(Xf, is_speech);
                    G  = get_tf_gain_map(Xf, obj.cfg);       %#ok<NASGU>
                    W  = obj.compute_sdw_mwf_();
                    Yf = sum(conj(W) .* Xf, 2);
                    Yf_full = [Yf; conj(Yf(end-1:-1:2))];
                    yblk = real(ifft(Yf_full)) .* obj.win;
                    obj.ola_buf = obj.ola_buf + yblk;
                    outH = obj.ola_buf(1:H);
                    obj.ola_buf = [obj.ola_buf(H+1:end); zeros(H,1)];
                else
                    outH = zeros(H, 1);
                end
                y(cursor:cursor+H-1) = outH;
                cursor = cursor + H;
            end
            % Any trailing partial block is passed through from ref mic
            if cursor <= N
                y(cursor:end) = x(cursor:end, obj.ref_mic);
            end
        end

        function reset(obj)
            obj.in_buf(:)  = 0;
            obj.ola_buf(:) = 0;
            obj.out_queue  = [];
            I = eye(obj.n_mics);
            R0 = reshape(I, 1, obj.n_mics, obj.n_mics) * obj.eps_reg;
            obj.Rnn = repmat(R0, obj.nbin, 1, 1);
            obj.Rss = repmat(R0, obj.nbin, 1, 1);
        end
    end

    methods (Access = private)
        function update_covariances_(obj, Xf, is_speech)
            a = ternary(is_speech, obj.alpha_ss, obj.alpha_nn);
            for k = 1:obj.nbin
                xk = Xf(k, :).';
                R  = xk * xk';
                if is_speech
                    obj.Rss(k,:,:) = a*squeeze(obj.Rss(k,:,:)) + (1-a)*R;
                else
                    obj.Rnn(k,:,:) = a*squeeze(obj.Rnn(k,:,:)) + (1-a)*R;
                end
            end
        end

        function W = compute_sdw_mwf_(obj)
            % Rank-1 / SDW-MWF weight per bin targeting the reference mic.
            W = zeros(obj.nbin, obj.n_mics);
            I = eye(obj.n_mics) * obj.eps_reg;
            e = zeros(obj.n_mics, 1); e(obj.ref_mic) = 1;
            for k = 1:obj.nbin
                Rn = squeeze(obj.Rnn(k,:,:)) + I;
                Rs = squeeze(obj.Rss(k,:,:));
                denom = Rs + obj.mu * Rn;
                W(k, :) = (denom \ (Rs * e)).';
            end
        end
    end
end

function out = ternary(cond, a, b)
    if cond, out = a; else, out = b; end
end
