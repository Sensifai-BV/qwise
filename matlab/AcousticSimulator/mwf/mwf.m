classdef mwf < handle
%MWF  Q-WiSE Multi-channel Wiener Filter (batch + streaming).
%
%   The class exposes two complementary entry points:
%
%     y = mwf_obj.process(vad_audio, mic_signals)
%         Batch mode that mirrors the reference Q-WiSE Python pipeline
%         (mwf.py) end-to-end:
%           1. cross-correlate the VAD-extracted speech with the reference
%              mic to align it within the recording
%           2. build a frame-level speech/noise mask
%           3. compute multi-channel STFTs
%           4. estimate Phi_nn from noise frames and Phi_yy from speech
%              frames, derive Phi_ss = Phi_yy - Phi_nn (PSD-enforced)
%           5. compute beamformer weights (GEV / MWF / MVDR — selectable)
%           6. apply beamformer + optional Wiener post-filter + iSTFT
%         Inputs:
%           vad_audio   [Lv x 1]            VAD output (extracted speech) audio
%           mic_signals [Lm x n_mics]       matrix, or 1xN / Nx1 cell of [Lm_i x 1]
%
%     y = mwf_obj.step(mic_block, is_speech)
%         Real-time streaming wrapper used by the GUI loop. Maintains
%         per-bin EMA estimates of Phi_ss and Phi_nn and recomputes the
%         configured beamformer per hop. Pass-through when
%         cfg.mwf.passthrough is true.
%
%   Configuration (cfg.mwf):
%     method            'gev' | 'mwf' | 'mvdr'
%     ref_mic           1-based reference microphone index
%     mu                SDW trade-off (used by 'mwf' only)
%     eps_reg           numerical floor for regularization
%     diag_load_ratio   trace-proportional diagonal loading
%     postfilter        bool, apply Wiener post-filter
%     gain_floor        post-filter floor (anti musical-noise)
%     noise_floor_alpha post-filter noise-tracker EMA
%     pf_smooth_kernel  frequency smoothing kernel size
%     mask_threshold    batch speech-mask RMS threshold
%     mask_context      batch speech-mask dilation radius
%     n_fft, hop        batch STFT geometry
%     stft_win, stft_hop streaming STFT geometry
%     alpha_nn, alpha_ss streaming covariance EMA
%     passthrough       streaming bypass (returns reference mic)

    properties
        cfg
        % --- selection / regularization -----------------------------
        method
        ref_mic
        mu
        eps_reg
        diag_load_ratio
        % --- post-filter --------------------------------------------
        postfilter
        gain_floor
        noise_floor_alpha
        pf_smooth_kernel
        % --- batch mask params --------------------------------------
        mask_threshold
        mask_context
        % --- batch STFT geometry ------------------------------------
        n_fft
        hop
        % --- streaming STFT geometry --------------------------------
        Lwin
        Hstep
        nbin
        n_mics
        alpha_nn
        alpha_ss
        passthrough
        % --- streaming state ----------------------------------------
        win
        Rnn         % [nbin x n_mics x n_mics]
        Rss         % [nbin x n_mics x n_mics]
    end

    properties (Access = private)
        in_buf      % [Lwin x n_mics] analysis ring
        ola_buf     % [Lwin x 1]      output overlap-add tail
    end

    methods
        function obj = mwf(cfg)
            obj.cfg               = cfg;
            obj.method            = lower(cfg.mwf.method);
            obj.ref_mic           = cfg.mwf.ref_mic;
            obj.mu                = cfg.mwf.mu;
            obj.eps_reg           = cfg.mwf.eps_reg;
            obj.diag_load_ratio   = field_or_(cfg.mwf, 'diag_load_ratio', 1e-4);
            obj.postfilter        = field_or_(cfg.mwf, 'postfilter', true);
            obj.gain_floor        = field_or_(cfg.mwf, 'gain_floor', 0.08);
            obj.noise_floor_alpha = field_or_(cfg.mwf, 'noise_floor_alpha', 0.98);
            obj.pf_smooth_kernel  = field_or_(cfg.mwf, 'pf_smooth_kernel', 3);
            obj.mask_threshold    = field_or_(cfg.mwf, 'mask_threshold', 0.01);
            obj.mask_context      = field_or_(cfg.mwf, 'mask_context', 3);
            obj.n_fft             = field_or_(cfg.mwf, 'n_fft', 1024);
            obj.hop               = field_or_(cfg.mwf, 'hop', 256);
            obj.Lwin              = cfg.mwf.stft_win;
            obj.Hstep             = cfg.mwf.stft_hop;
            obj.nbin              = obj.Lwin / 2 + 1;
            obj.n_mics            = cfg.n_mics;
            obj.alpha_nn          = cfg.mwf.alpha_nn;
            obj.alpha_ss          = cfg.mwf.alpha_ss;
            obj.passthrough       = cfg.mwf.passthrough;

            obj.win   = sqrt(hann(obj.Lwin, 'periodic'));
            I0        = reshape(eye(obj.n_mics) * obj.eps_reg, 1, obj.n_mics, obj.n_mics);
            obj.Rnn   = repmat(I0, obj.nbin, 1, 1);
            obj.Rss   = repmat(I0, obj.nbin, 1, 1);
            obj.in_buf  = zeros(obj.Lwin, obj.n_mics);
            obj.ola_buf = zeros(obj.Lwin, 1);

            validate_method_(obj.method);
        end

        % =====================================================================
        %  BATCH PIPELINE (matches Python multichannel_wiener_filter())
        % =====================================================================
        function y = process(obj, vad_audio, mic_signals)
        %PROCESS  Run the full batch MWF pipeline on a complete recording.
        %
        %   y = obj.process(vad_audio, mic_signals)
        %
        %   vad_audio   : [Lv x 1] VAD-extracted speech audio.
        %   mic_signals : matrix [Lm x n_mics] OR cell {n_mics}([Lm_i x 1]).
        %                 If lengths differ they are truncated to the shortest.

            mic_mat = pack_mic_matrix_(mic_signals);
            n_ch    = size(mic_mat, 2);
            if n_ch < 2
                warning('mwf:process:few_channels', ...
                        'MWF expects at least 2 channels for spatial filtering.');
            end

            % ---- align VAD-extracted speech to reference mic ---------------
            ref = mic_mat(:, obj.ref_mic);
            [~, aligned_vad] = mwf_align_vad(vad_audio, ref);

            % ---- multi-channel STFT ----------------------------------------
            stft_list = cell(n_ch, 1);
            for m = 1:n_ch
                stft_list{m} = mwf_stft(mic_mat(:, m), obj.n_fft, obj.hop);
            end
            [n_freq, n_frames] = size(stft_list{1});
            X_multi            = zeros(n_ch, n_freq, n_frames);
            for m = 1:n_ch
                X_multi(m, :, :) = stft_list{m};
            end

            % ---- build speech / noise frame masks --------------------------
            speech_mask = mwf_build_speech_mask(aligned_vad, n_frames, ...
                                                obj.n_fft, obj.hop, ...
                                                obj.mask_threshold, obj.mask_context);
            noise_mask  = ~speech_mask;

            if sum(speech_mask) < 5
                % Retry with a more permissive threshold (mirrors Python safety net).
                speech_mask = mwf_build_speech_mask(aligned_vad, n_frames, ...
                                                    obj.n_fft, obj.hop, ...
                                                    0.003, obj.mask_context);
                noise_mask  = ~speech_mask;
            end
            if sum(noise_mask) < 10
                % Fall back to flanking 15% of frames as noise context.
                n15              = max(10, floor(n_frames / 7));
                noise_mask       = false(n_frames, 1);
                noise_mask(1:min(n15, n_frames)) = true;
                tail_start       = max(1, n_frames - n15 + 1);
                noise_mask(tail_start:n_frames) = true;
                speech_mask      = ~noise_mask;
            end

            % ---- covariance estimation -------------------------------------
            Phi_nn = mwf_estimate_covariance(X_multi, noise_mask,  obj.eps_reg);
            Phi_yy = mwf_estimate_covariance(X_multi, speech_mask, obj.eps_reg);
            Phi_ss = Phi_yy - Phi_nn;
            Phi_ss = enforce_psd_(Phi_ss, n_ch, obj.eps_reg);

            % ---- beamformer weights ---------------------------------------
            W = obj.compute_weights_(Phi_ss, Phi_nn);

            % ---- beamforming + optional post-filter -----------------------
            Y = mwf_apply_beamformer(W, X_multi);
            if obj.postfilter
                Y = mwf_wiener_postfilter(Y, speech_mask, ...
                                          obj.noise_floor_alpha, ...
                                          obj.gain_floor, ...
                                          obj.eps_reg, ...
                                          obj.pf_smooth_kernel);
            end

            % ---- iSTFT + normalize ----------------------------------------
            y = mwf_istft(Y, obj.n_fft, obj.hop);
            y = real(y);
            if numel(y) > size(mic_mat, 1)
                y = y(1:size(mic_mat, 1));
            end
            peak = max(abs(y));
            if peak > 0
                y = y / peak * 0.9;
            end
        end

        % =====================================================================
        %  STREAMING PIPELINE (used by SimulatorUI)
        % =====================================================================
        function y = step(obj, x, is_speech)
        %STEP  Process one mic block and return an equally-sized mono output.
        %
        %   x          [N x n_mics]  time-domain mic block
        %   is_speech  logical       VAD decision for this block
        %   y          [N x 1]       enhanced reference-mic signal

            N = size(x, 1);

            if obj.passthrough
                y = x(:, obj.ref_mic);
                return;
            end

            y      = zeros(N, 1);
            L      = obj.Lwin;
            H      = obj.Hstep;
            cursor = 1;
            while cursor + H - 1 <= N
                seg        = x(cursor:cursor + H - 1, :);
                obj.in_buf = [obj.in_buf(H + 1:end, :); seg];

                if any(obj.in_buf(:))
                    Xf = fft(obj.in_buf .* obj.win, L);
                    Xf = Xf(1:obj.nbin, :);

                    obj.update_covariances_(Xf, is_speech);

                    W  = obj.compute_streaming_weights_();
                    Yf = sum(conj(W) .* Xf, 2);

                    Yfull = [Yf; conj(Yf(end - 1:-1:2))];
                    yblk  = real(ifft(Yfull)) .* obj.win;

                    obj.ola_buf = obj.ola_buf + yblk;
                    outH        = obj.ola_buf(1:H);
                    obj.ola_buf = [obj.ola_buf(H + 1:end); zeros(H, 1)];
                else
                    outH = zeros(H, 1);
                end
                y(cursor:cursor + H - 1) = outH;
                cursor = cursor + H;
            end
            if cursor <= N
                y(cursor:end) = x(cursor:end, obj.ref_mic);
            end
        end

        function reset(obj)
            obj.in_buf(:)  = 0;
            obj.ola_buf(:) = 0;
            I0     = reshape(eye(obj.n_mics) * obj.eps_reg, 1, obj.n_mics, obj.n_mics);
            obj.Rnn = repmat(I0, obj.nbin, 1, 1);
            obj.Rss = repmat(I0, obj.nbin, 1, 1);
        end
    end

    % =====================================================================
    %  INTERNAL HELPERS
    % =====================================================================
    methods (Access = private)
        function W = compute_weights_(obj, Phi_ss, Phi_nn)
            switch obj.method
                case 'mvdr'
                    W = mwf_compute_mvdr_weights(Phi_ss, Phi_nn, obj.ref_mic, ...
                                                 obj.eps_reg, obj.diag_load_ratio);
                case 'gev'
                    W = mwf_compute_gev_weights(Phi_ss, Phi_nn, obj.ref_mic, ...
                                                obj.eps_reg, obj.diag_load_ratio);
                otherwise
                    W = mwf_compute_mwf_weights(Phi_ss, Phi_nn, obj.ref_mic, ...
                                                obj.mu, obj.eps_reg, obj.diag_load_ratio);
            end
        end

        function W = compute_streaming_weights_(obj)
            % Reshape EMA covariances to the [n_freq x n_ch x n_ch] layout
            % expected by the modular weight kernels.
            W = obj.compute_weights_(obj.Rss, obj.Rnn);
        end

        function update_covariances_(obj, Xf, is_speech)
            if is_speech
                a = obj.alpha_ss;
            else
                a = obj.alpha_nn;
            end
            for k = 1:obj.nbin
                xk = Xf(k, :).';
                R  = xk * xk';
                if is_speech
                    obj.Rss(k, :, :) = a * squeeze(obj.Rss(k, :, :)) + (1 - a) * R;
                else
                    obj.Rnn(k, :, :) = a * squeeze(obj.Rnn(k, :, :)) + (1 - a) * R;
                end
            end
        end
    end
end

% =========================================================================
%  FILE-LOCAL UTILITIES
% =========================================================================
function v = field_or_(s, name, default_val)
    if isfield(s, name) && ~isempty(s.(name))
        v = s.(name);
    else
        v = default_val;
    end
end

function validate_method_(m)
    if ~ismember(m, {'gev', 'mwf', 'mvdr'})
        error('mwf:bad_method', ...
              'cfg.mwf.method must be one of: gev | mwf | mvdr (got "%s").', m);
    end
end

function mic_mat = pack_mic_matrix_(mic_signals)
    if iscell(mic_signals)
        n_ch  = numel(mic_signals);
        lens  = cellfun(@numel, mic_signals);
        L     = min(lens);
        mic_mat = zeros(L, n_ch);
        for k = 1:n_ch
            v = mic_signals{k}(:);
            mic_mat(:, k) = v(1:L);
        end
    elseif isnumeric(mic_signals)
        mic_mat = mic_signals;
        if size(mic_mat, 1) < size(mic_mat, 2)
            % treat the [n_ch x L] orientation defensively
            mic_mat = mic_mat.';
        end
    else
        error('mwf:bad_mic_signals', ...
              'mic_signals must be a numeric matrix or a cell of column vectors.');
    end
end

function Phi = enforce_psd_(Phi, n_ch, eps_reg)
    n_freq = size(Phi, 1);
    for f = 1:n_freq
        M = squeeze(Phi(f, :, :));
        M = (M + M') / 2;
        [V, D] = eig(M);
        d      = real(diag(D));
        d(d < 0) = 0;
        M = V * diag(d) * V';
        Phi(f, :, :) = M + eps_reg * eye(n_ch);
    end
end
