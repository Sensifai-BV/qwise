function Phi = mwf_estimate_covariance(X_multi, mask, eps_reg)
%MWF_ESTIMATE_COVARIANCE  Spatial covariance per frequency bin from multi-channel STFT.
%
%   Phi = mwf_estimate_covariance(X_multi, mask, eps_reg)
%
%   For every frequency bin, accumulates an outer-product sum over the masked
%   frames and normalizes by the number of masked frames. A small multiple of
%   the identity (`eps_reg`) is added for numerical stability — the
%   reference Q-WiSE Python pipeline uses the same diagonal loading.
%
%   Inputs:
%     X_multi  [n_channels x n_freq x n_frames]  complex STFT cube
%     mask     [n_frames x 1] logical            frames to include
%     eps_reg  scalar                            regularization weight
%
%   Output:
%     Phi      [n_freq x n_channels x n_channels]  complex covariance matrices

    if nargin < 3 || isempty(eps_reg), eps_reg = 1e-10; end

    [n_ch, n_freq, ~] = size(X_multi);
    Phi              = zeros(n_freq, n_ch, n_ch);

    idx = find(mask);
    if isempty(idx)
        I0  = eps_reg * eye(n_ch);
        Phi = repmat(reshape(I0, 1, n_ch, n_ch), n_freq, 1, 1);
        return;
    end

    n_valid = numel(idx);
    for f = 1:n_freq
        obs = squeeze(X_multi(:, f, idx));     % [n_ch x n_valid] (or [n_ch x 1])
        if n_valid == 1
            obs = obs(:);
        end
        R = (obs * obs') / n_valid;
        % Sanitize NaN/Inf the same way the Python reference does.
        R(~isfinite(R)) = 0;
        Phi(f, :, :) = R + eps_reg * eye(n_ch);
    end
end
