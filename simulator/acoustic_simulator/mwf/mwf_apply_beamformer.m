function Y = mwf_apply_beamformer(W, X_multi)
%MWF_APPLY_BEAMFORMER  Apply per-bin beamformer weights to a multi-channel STFT.
%
%   Y = mwf_apply_beamformer(W, X_multi)
%
%       Y(f, t) = W(f)^H * X_multi(:, f, t)
%
%   Inputs:
%     W        [n_freq x n_ch]                complex beamformer weights
%     X_multi  [n_ch x n_freq x n_frames]     complex multi-channel STFT
%
%   Output:
%     Y        [n_freq x n_frames]            single-channel beamformer STFT

    [n_ch, n_freq, n_frames] = size(X_multi);
    Y = zeros(n_freq, n_frames);
    for f = 1:n_freq
        wH      = conj(W(f, :));                         % [1 x n_ch]
        Xf      = reshape(X_multi(:, f, :), n_ch, n_frames);
        Y(f, :) = wH * Xf;
    end
end
