function Yout = mwf_omlsa_postfilter(Y, gain_floor_db, alpha_dd, alpha_s, ...
                                    alpha_d, win_min, eps_reg)
%MWF_OMLSA_POSTFILTER  OMLSA single-channel residual suppressor (batch STFT).
%
%   Yout = mwf_omlsa_postfilter(Y, gain_floor_db, alpha_dd, alpha_s, ...
%                               alpha_d, win_min, eps_reg)
%
%   Optimally-Modified Log-Spectral-Amplitude (OMLSA) enhancement applied to
%   the beamformer output spectrogram. This is the stage that drives the
%   low-frequency / tonal residual (e.g. drone-rotor harmonics that a small
%   array cannot null spatially) toward zero, while preserving speech, and it
%   tracks the noise PSD CONTINUOUSLY (including under speech) so it removes
%   residual that the VAD-gated Wiener post-filter leaves behind.
%
%   Components (all per-bin scalar math -> ONNX-mappable):
%     * minimum-statistics noise PSD tracker (smoothed power + running minimum)
%     * soft speech-presence probability from the smoothed/min power ratio
%     * decision-directed a priori SNR
%     * Ephraim-Malah log-spectral-amplitude (LSA) gain
%     * OMLSA blend  G = G_LSA^p * Gmin^(1-p)  with a spectral floor
%
%   Inputs:
%     Y             [n_freq x n_frames] complex beamformer STFT
%     gain_floor_db spectral floor in dB (e.g. -30). Lower = more suppression.
%     alpha_dd      decision-directed smoothing            (default 0.92)
%     alpha_s       power smoothing for noise tracking      (default 0.90)
%     alpha_d       noise-PSD update rate                   (default 0.85)
%     win_min       minimum-statistics window (frames)      (default 60)
%     eps_reg       numerical floor                         (default 1e-10)
%
%   Output:
%     Yout          [n_freq x n_frames] enhanced complex STFT
%
%   NOTE (ONNX): the only non-elementary op is the exponential integral E1 in
%   the LSA gain (here MATLAB's expint). For export, replace it with a standard
%   rational/polynomial approximation of E1 — accuracy is non-critical because
%   the gain is bounded by [Gmin, 1].

    if nargin < 2 || isempty(gain_floor_db), gain_floor_db = -30; end
    if nargin < 3 || isempty(alpha_dd),      alpha_dd = 0.92;     end
    if nargin < 4 || isempty(alpha_s),       alpha_s  = 0.90;     end
    if nargin < 5 || isempty(alpha_d),       alpha_d  = 0.85;     end
    if nargin < 6 || isempty(win_min),       win_min  = 60;       end
    if nargin < 7 || isempty(eps_reg),       eps_reg  = 1e-10;    end

    [n_freq, n_frames] = size(Y);
    Gmin = 10^(gain_floor_db / 20);
    P    = abs(Y).^2;

    % --- initial noise PSD from the first few frames -----------------------
    n0   = min(8, n_frames);
    lam  = max(mean(P(:, 1:n0), 2), 1e-10);
    S    = lam;                          % smoothed power
    Gp   = ones(n_freq, 1);              % previous gain
    gam_p = ones(n_freq, 1);             % previous a posteriori SNR
    minbuf = repmat(S, 1, win_min);      % ring buffer for running minimum
    bptr   = 1;

    Yout = zeros(n_freq, n_frames);

    for t = 1:n_frames
        p = P(:, t);

        % smoothed power + running minimum (minimum statistics)
        S            = alpha_s * S + (1 - alpha_s) * p;
        minbuf(:, bptr) = S;
        bptr         = bptr + 1; if bptr > win_min, bptr = 1; end
        Smin         = min(minbuf, [], 2);

        % soft speech-presence probability from S/Smin ratio
        Sr    = S ./ max(Smin, 1e-12);
        ppres = min(max((Sr - 1.0) / 8.0, 0), 1);

        % conditional (soft) noise-PSD update — only where speech is absent
        lam = lam + (1 - ppres) .* (alpha_d * lam + (1 - alpha_d) * p - lam);
        lam = max(lam, 1e-12);

        % SNRs
        gamma = p ./ lam;
        xi    = alpha_dd * (Gp.^2) .* gam_p + (1 - alpha_dd) * max(gamma - 1, 0);
        xi    = max(xi, 1e-3);
        nu    = min(max(xi ./ (1 + xi) .* gamma, 1e-6), 500);

        % LSA gain then OMLSA blend with floor
        G_lsa = xi ./ (1 + xi) .* exp(0.5 * expint(nu));
        G     = (G_lsa .^ ppres) .* (Gmin .^ (1 - ppres));
        G     = min(max(G, Gmin), 1.0);

        Yout(:, t) = G .* Y(:, t);
        Gp    = G;
        gam_p = gamma;
    end
end
