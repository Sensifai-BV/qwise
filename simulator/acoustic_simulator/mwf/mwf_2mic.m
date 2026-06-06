function w = mwf_2mic(phi_s, Phi_nn, d, refMic, epsLoad)
%MWF_2MIC  Closed-form 2-microphone multichannel Wiener filter (one bin).
%
%   w = mwf_2mic(phi_s, Phi_nn, d, refMic)
%   w = mwf_2mic(phi_s, Phi_nn, d, refMic, epsLoad)
%
%   Rank-1-speech-model MWF for a SINGLE time-frequency bin, using the exact
%   closed-form 2x2 inverse of the noise covariance (Sherman-Morrison form):
%
%       w = (phi_s * conj(d(refMic)) / (1 + phi_s * d' * inv(Phi_nn) * d)) ...
%           * inv(Phi_nn) * d
%
%   with
%       inv(Phi_nn) = 1/(a*c - |b|^2) * [c, -b; -conj(b), a]
%       Phi_nn = [a, b; conj(b), c]
%
%   This is the 2-mic special case of the general rank-1 MWF. For the
%   batch/streaming pipeline use mwf_compute_rank1_weights, which estimates
%   (phi_s, d) from Phi_ss per bin and calls this closed form.
%
%   Inputs:
%     phi_s   scalar speech PSD (real, nonnegative)
%     Phi_nn  2x2 noise covariance matrix (Hermitian)
%     d       2x1 complex steering vector
%     refMic  reference microphone index (1 or 2)
%     epsLoad optional diagonal loading scalar (default 0; e.g. 1e-3..1e-6)
%
%   Output:
%     w       2x1 complex MWF weight vector. The enhanced bin is s_hat = w' * y.
%
%   Notes:
%     - One time-frequency bin; call per bin (and per frame) for STFT use.
%     - A small epsLoad is recommended when Phi_nn is ill-conditioned.

    if nargin < 5 || isempty(epsLoad)
        epsLoad = 0;
    end

    % ---- input validation -------------------------------------------------
    if ~isscalar(phi_s) || ~isreal(phi_s) || phi_s < 0
        error('mwf_2mic:phi_s', 'phi_s must be a real nonnegative scalar.');
    end
    if ~isequal(size(Phi_nn), [2, 2])
        error('mwf_2mic:Phi_nn', 'Phi_nn must be a 2x2 matrix.');
    end
    if ~isequal(size(d), [2, 1])
        error('mwf_2mic:d', 'd must be a 2x1 column vector.');
    end
    if ~(isscalar(refMic) && any(refMic == [1, 2]))
        error('mwf_2mic:refMic', 'refMic must be 1 or 2.');
    end

    if epsLoad > 0
        Phi_nn = Phi_nn + epsLoad * eye(2);
    end

    % ---- extract matrix elements -----------------------------------------
    a = Phi_nn(1, 1);
    b = Phi_nn(1, 2);
    c = Phi_nn(2, 2);

    hermErr = norm(Phi_nn - Phi_nn', 'fro');
    if hermErr > 1e-8
        warning('mwf_2mic:hermitian', ...
                'Phi_nn is not exactly Hermitian. Hermitian error = %.3e', hermErr);
    end

    % ---- determinant ------------------------------------------------------
    Delta = a * c - abs(b)^2;
    if abs(Delta) < 1e-12
        error('mwf_2mic:singular', ...
              'Phi_nn is singular or nearly singular. Add diagonal loading.');
    end

    % ---- x = inv(Phi_nn) * d (closed form) -------------------------------
    x1 = (  c * d(1) - b * d(2) ) / Delta;
    x2 = ( -conj(b) * d(1) + a * d(2) ) / Delta;
    x  = [x1; x2];

    % ---- eta = d^H x ------------------------------------------------------
    eta = d' * x;

    % ---- final weights ----------------------------------------------------
    dr    = d(refMic);
    alpha = (phi_s * conj(dr)) / (1 + phi_s * eta);
    w     = alpha * x;
end
