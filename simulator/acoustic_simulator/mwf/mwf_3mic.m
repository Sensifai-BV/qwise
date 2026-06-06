function w = mwf_3mic(phi_s, Phi_nn, d, refMic, epsLoad)
%MWF_3MIC  Closed-form 3-microphone multichannel Wiener filter (one bin).
%
%   w = mwf_3mic(phi_s, Phi_nn, d, refMic)
%   w = mwf_3mic(phi_s, Phi_nn, d, refMic, epsLoad)
%
%   Rank-1-speech-model MWF for a SINGLE time-frequency bin, using the exact
%   adjugate/cofactor 3x3 inverse of the noise covariance:
%
%       w = (phi_s * conj(d(refMic)) / (1 + phi_s * d' * inv(Phi_nn) * d)) ...
%           * inv(Phi_nn) * d
%
%   This is the 3-mic special case of the general rank-1 MWF. For the
%   batch/streaming pipeline use mwf_compute_rank1_weights, which estimates
%   (phi_s, d) from Phi_ss per bin and calls this closed form.
%
%   Inputs:
%     phi_s   scalar speech PSD (real, nonnegative)
%     Phi_nn  3x3 noise covariance matrix (Hermitian)
%     d       3x1 complex steering vector
%     refMic  reference microphone index (1, 2, or 3)
%     epsLoad optional diagonal loading scalar (default 0; e.g. 1e-3..1e-6)
%
%   Output:
%     w       3x1 complex MWF weight vector. The enhanced bin is s_hat = w' * y.
%
%   Note: the adjugate inverse is exact for a nonsingular 3x3 matrix but is
%   less numerically robust than a linear solve. It is used here for the
%   fixed closed-form style (e.g. graph/ONNX export).

    if nargin < 5 || isempty(epsLoad)
        epsLoad = 0;
    end

    % ---- input validation -------------------------------------------------
    if ~isscalar(phi_s) || ~isreal(phi_s) || phi_s < 0
        error('mwf_3mic:phi_s', 'phi_s must be a real nonnegative scalar.');
    end
    if ~isequal(size(Phi_nn), [3, 3])
        error('mwf_3mic:Phi_nn', 'Phi_nn must be a 3x3 matrix.');
    end
    if ~isequal(size(d), [3, 1])
        error('mwf_3mic:d', 'd must be a 3x1 column vector.');
    end
    if ~(isscalar(refMic) && any(refMic == [1, 2, 3]))
        error('mwf_3mic:refMic', 'refMic must be 1, 2, or 3.');
    end

    if epsLoad > 0
        Phi_nn = Phi_nn + epsLoad * eye(3);
    end

    hermErr = norm(Phi_nn - Phi_nn', 'fro');
    if hermErr > 1e-8
        warning('mwf_3mic:hermitian', ...
                'Phi_nn is not exactly Hermitian. Hermitian error = %.3e', hermErr);
    end

    % ---- inverse via adjugate/cofactor formula ---------------------------
    invPhi = inv3x3_adjugate(Phi_nn);

    % ---- x = inv(Phi_nn) * d ---------------------------------------------
    x = invPhi * d;

    % ---- eta = d^H x ------------------------------------------------------
    eta = d' * x;

    % ---- final weights ----------------------------------------------------
    dr    = d(refMic);
    alpha = (phi_s * conj(dr)) / (1 + phi_s * eta);
    w     = alpha * x;
end


function Ainv = inv3x3_adjugate(A)
%INV3X3_ADJUGATE  Exact 3x3 inverse via cofactors / adjugate.

    a11 = A(1,1); a12 = A(1,2); a13 = A(1,3);
    a21 = A(2,1); a22 = A(2,2); a23 = A(2,3);
    a31 = A(3,1); a32 = A(3,2); a33 = A(3,3);

    C11 =  (a22*a33 - a23*a32);
    C12 = -(a21*a33 - a23*a31);
    C13 =  (a21*a32 - a22*a31);

    C21 = -(a12*a33 - a13*a32);
    C22 =  (a11*a33 - a13*a31);
    C23 = -(a11*a32 - a12*a31);

    C31 =  (a12*a23 - a13*a22);
    C32 = -(a11*a23 - a13*a21);
    C33 =  (a11*a22 - a12*a21);

    detA = a11*C11 + a12*C12 + a13*C13;
    if abs(detA) < 1e-12
        error('mwf_3mic:singular', ...
              'Matrix is singular or nearly singular. Add diagonal loading.');
    end

    % adjugate = transpose of cofactor matrix
    adjA = [C11, C21, C31;
            C12, C22, C32;
            C13, C23, C33];

    Ainv = adjA / detA;
end
