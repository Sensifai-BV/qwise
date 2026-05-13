function Y_out = mwf_wiener_postfilter(Y, speech_mask, noise_floor_alpha, gain_floor, eps_reg, smooth_kernel)
%MWF_WIENER_POSTFILTER  Adaptive single-channel Wiener post-filter.
%
%   Y_out = mwf_wiener_postfilter(Y, speech_mask, noise_floor_alpha, gain_floor, eps_reg, smooth_kernel)
%
%   Implements the decision-directed Wiener post-filter from the Q-WiSE
%   reference pipeline:
%
%     1. Seed the noise spectrum from frames flagged as noise.
%     2. For each frame:
%        - update the noise PSD on noise frames (EMA, α = noise_floor_alpha)
%        - compute a-priori SNR = max(P_y - P_n, 0) / P_n
%        - gain = SNR / (SNR + 1), clamped at gain_floor
%     3. Smooth the gain map across frequency to suppress musical noise.
%
%   Inputs:
%     Y                 [n_freq x n_frames]   complex beamformer STFT
%     speech_mask       [n_frames x 1] logical
%     noise_floor_alpha scalar 0..1            EMA smoothing for noise PSD
%     gain_floor        scalar                 minimum gain (anti musical noise)
%     eps_reg           scalar                 numerical floor
%     smooth_kernel     odd integer            frequency smoothing kernel size
%
%   Output:
%     Y_out             [n_freq x n_frames]   gained complex STFT

    if nargin < 3 || isempty(noise_floor_alpha), noise_floor_alpha = 0.98; end
    if nargin < 4 || isempty(gain_floor),        gain_floor        = 0.08; end
    if nargin < 5 || isempty(eps_reg),           eps_reg           = 1e-10; end
    if nargin < 6 || isempty(smooth_kernel),     smooth_kernel     = 3;     end

    [n_freq, n_frames] = size(Y);
    mag   = abs(Y);
    phase = angle(Y);

    speech_mask = logical(speech_mask(:));
    if numel(speech_mask) ~= n_frames
        if numel(speech_mask) > n_frames
            speech_mask = speech_mask(1:n_frames);
        else
            speech_mask = [speech_mask; false(n_frames - numel(speech_mask), 1)];
        end
    end

    noise_idx = find(~speech_mask);
    if ~isempty(noise_idx)
        noise_psd = mean(mag(:, noise_idx).^2, 2);
    else
        noise_psd = 0.1 * mean(mag.^2, 2);
    end
    noise_psd = max(noise_psd, eps_reg);
    noise_est = noise_psd;

    G = ones(n_freq, n_frames);
    for t = 1:n_frames
        P = mag(:, t).^2;
        if ~speech_mask(t)
            noise_est = noise_floor_alpha * noise_est + (1 - noise_floor_alpha) * P;
        end
        snr  = max(P - noise_est, 0) ./ (noise_est + eps_reg);
        gain = snr ./ (snr + 1);
        G(:, t) = max(gain, gain_floor);
    end

    % Frequency smoothing — moving average of length `smooth_kernel`.
    if smooth_kernel >= 3
        kern = ones(smooth_kernel, 1) / smooth_kernel;
        for t = 1:n_frames
            G(:, t) = conv(G(:, t), kern, 'same');
        end
    end

    Y_out = mag .* G .* exp(1j * phase);
end
