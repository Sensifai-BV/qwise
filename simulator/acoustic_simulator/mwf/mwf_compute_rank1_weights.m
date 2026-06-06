function W = mwf_compute_rank1_weights(Phi_ss, Phi_nn, ref_channel, eps_reg, diag_load_ratio)
%MWF_COMPUTE_RANK1_WEIGHTS  Closed-form rank-1 MWF (2-/3-mic special cases).
%
%   W = mwf_compute_rank1_weights(Phi_ss, Phi_nn, ref_channel, eps_reg, diag_load_ratio)
%
%   Per frequency bin, this assumes a rank-1 speech model and computes the
%   multichannel Wiener filter via the Sherman-Morrison reduction:
%
%       w(f) = (phi_s * conj(d_ref) / (1 + phi_s * d^H Phi_nn^{-1} d)) ...
%              * Phi_nn^{-1} d
%
%   where the speech PSD phi_s and the steering vector d are estimated from
%   the dominant eigenpair of Phi_ss(f) (d is phase-normalized so d_ref = 1,
%   making conj(d_ref) = 1). The noise inverse uses the exact closed forms:
%     n_ch == 2 : 2x2 analytic inverse  (see mwf_2mic)
%     n_ch == 3 : 3x3 adjugate inverse  (see mwf_3mic)
%     otherwise : a regularized linear solve (graceful fallback for 1 or >3 mics)
%
%   This kernel plugs into the same pipeline slot as mwf_compute_gev_weights /
%   mwf_compute_mvdr_weights / mwf_compute_mwf_weights and shares their
%   signature and output convention.
%
%   Inputs:
%     Phi_ss          [n_freq x n_ch x n_ch]  speech covariance
%     Phi_nn          [n_freq x n_ch x n_ch]  noise covariance
%     ref_channel     scalar                  1-based reference mic index
%     eps_reg         scalar                  numerical floor
%     diag_load_ratio scalar                  trace-proportional diagonal loading
%
%   Output:
%     W   [n_freq x n_ch]  complex beamformer weights. The beamformer applies
%                          Y(f) = sum(conj(W(f,:)).' .* X(f,:).') = w^H x.

    if nargin < 4 || isempty(eps_reg),          eps_reg = 1e-10; end
    if nargin < 5 || isempty(diag_load_ratio),  diag_load_ratio = 1e-4; end

    [n_freq, n_ch, ~] = size(Phi_ss);
    W                 = zeros(n_freq, n_ch);
    I_ch              = eye(n_ch);

    if ref_channel < 1 || ref_channel > n_ch
        ref_channel = 1;
    end

    for f = 1:n_freq
        Rs = squeeze(Phi_ss(f, :, :));
        Rn = squeeze(Phi_nn(f, :, :));

        Rs = (Rs + Rs') / 2;
        Rn = (Rn + Rn') / 2;

        % --- diagonal loading on the noise covariance -----------------------
        %   With diag_load_ratio == 0 the noise inverse is the RAW analytic
        %   inverse, identical to mwf_2mic / mwf_3mic. A positive ratio applies
        %   trace-proportional loading, identical to mwf_2mic_loaded /
        %   mwf_3mic_loaded (and consistent with the sibling kernels).
        if diag_load_ratio > 0
            load = diag_load_ratio * real(trace(Rn)) / n_ch;
            Rn   = Rn + load * I_ch;
        end

        % --- rank-1 speech model from dominant eigenpair of Phi_ss ----------
        %   Phi_ss = phi_s * d*d^H with d normalized so d_ref = 1. For the
        %   dominant unit eigenvector v (eigenvalue lam): d = v / v_ref and
        %   phi_s = lam * |v_ref|^2  (== speech PSD at the reference mic).
        [V, D]   = eig(Rs);
        [lam, k] = max(real(diag(D)));
        v        = V(:, k);
        vref     = v(ref_channel);
        if abs(vref) > eps_reg
            d     = v / vref;                 % d_ref = 1
            phi_s = max(lam, 0) * abs(vref)^2;
        else
            d     = v;
            phi_s = max(lam, 0);
        end

        % --- x = Phi_nn^{-1} d via the exact closed form --------------------
        switch n_ch
            case 2
                x = inv2x2_times_(Rn, d, eps_reg);
            case 3
                x = inv3x3_times_(Rn, d, eps_reg);
            otherwise
                try
                    x = Rn \ d;
                catch
                    x = pinv(Rn) * d;
                end
        end

        % --- Sherman-Morrison MWF weights -----------------------------------
        eta = d' * x;
        dr  = d(ref_channel);
        den = 1 + phi_s * eta;
        if abs(den) > eps_reg
            w = (phi_s * conj(dr) / den) * x;
        else
            w = zeros(n_ch, 1);
            w(ref_channel) = 1;
        end

        if ~all(isfinite(w))
            w = zeros(n_ch, 1);
            w(ref_channel) = 1;
        end
        W(f, :) = w.';
    end
end

% =========================================================================
%  FILE-LOCAL CLOSED-FORM SOLVES (no per-bin validation / warnings)
% =========================================================================
function x = inv2x2_times_(R, d, eps_reg)
%INV2X2_TIMES_  x = inv(R) * d for a 2x2 Hermitian R, analytic inverse.
    a = R(1,1); b = R(1,2); c = R(2,2);
    Delta = a*c - abs(b)^2;
    if abs(Delta) < eps_reg
        x = pinv(R) * d;
        return;
    end
    x = [ (  c*d(1) - b*d(2) ) / Delta;
          ( -conj(b)*d(1) + a*d(2) ) / Delta ];
end

function x = inv3x3_times_(R, d, eps_reg)
%INV3X3_TIMES_  x = inv(R) * d for a 3x3 R via adjugate/cofactor inverse.
    a11 = R(1,1); a12 = R(1,2); a13 = R(1,3);
    a21 = R(2,1); a22 = R(2,2); a23 = R(2,3);
    a31 = R(3,1); a32 = R(3,2); a33 = R(3,3);

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
    if abs(detA) < eps_reg
        x = pinv(R) * d;
        return;
    end

    adjA = [C11, C21, C31;
            C12, C22, C32;
            C13, C23, C33];
    x = (adjA * d) / detA;
end
