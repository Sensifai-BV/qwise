function W = mwf_compute_rankn_weights(Phi_ss, Phi_nn, ref_channel, eps_reg, ...
                                       mu, power_iters, cg_iters)
%MWF_COMPUTE_RANKN_WEIGHTS  N-microphone rank-1 SDW-MWF, inverse-free & ONNX-ready.
%
%   W = mwf_compute_rankn_weights(Phi_ss, Phi_nn, ref_channel, eps_reg, ...
%                                 mu, power_iters, cg_iters)
%
%   Generalization of the closed-form rank-1 MWF (mwf_2mic / mwf_3mic) to an
%   ARBITRARY number of microphones N >= 2. Per frequency bin:
%
%       w(f) = phi_s * conj(d_ref) * x / (mu + phi_s * d^H x),   x = Phi_nn^{-1} d
%
%   By the Woodbury identity this is EXACTLY the speech-distortion-weighted MWF
%       w(f) = (Phi_ss + mu*Phi_nn)^{-1} Phi_ss e_ref
%   for a rank-1 speech model Phi_ss = phi_s * d d^H. Hence:
%       mu = 1                      -> standard MWF (matches mwf_2mic/mwf_3mic)
%       mu > 1                      -> stronger noise suppression (more denoise)
%
%   Two pieces are computed WITHOUT eig() and WITHOUT a matrix inverse, so the
%   whole kernel maps 1:1 onto ONNX ops (matmul / dot / scalar arithmetic) once
%   complex values are split into (re, im):
%
%     (phi_s, d) : dominant eigenpair of Phi_ss via POWER ITERATION initialized
%                  from the reference column of Phi_ss. For an exact rank-1
%                  Phi_ss that column already equals d, so power_iters = 0
%                  reproduces the original rank-1 algorithm bit-for-bit;
%                  power_iters > 0 refines (phi_s, d) when Phi_ss is only
%                  approximately rank-1 (real estimated covariances).
%     x = Phi_nn^{-1} d : COMPLEX CONJUGATE GRADIENT, exact in <= N steps for a
%                  Hermitian positive-definite Phi_nn. No explicit inverse.
%
%   Inputs:
%     Phi_ss        [n_freq x n_ch x n_ch]  speech covariance
%     Phi_nn        [n_freq x n_ch x n_ch]  noise covariance (Hermitian PD)
%     ref_channel   scalar  1-based reference mic index
%     eps_reg       scalar  numerical floor                       (default 1e-10)
%     mu            scalar  SDW trade-off (>=0)                    (default 1.0)
%     power_iters   scalar  power-iteration refinement steps       (default 0)
%     cg_iters      scalar  CG iterations (>= n_ch is exact)       (default n_ch)
%
%   Output:
%     W   [n_freq x n_ch]  complex weights. Beamformer applies Y = w^H x.
%
%   See also MWF_2MIC, MWF_3MIC, MWF_COMPUTE_RANK1_WEIGHTS.

    if nargin < 4 || isempty(eps_reg),     eps_reg = 1e-10; end
    if nargin < 5 || isempty(mu),          mu = 1.0;        end
    if nargin < 6 || isempty(power_iters), power_iters = 0; end

    [n_freq, n_ch, ~] = size(Phi_ss);
    if nargin < 7 || isempty(cg_iters), cg_iters = n_ch; end
    cg_iters = max(cg_iters, n_ch);

    W = zeros(n_freq, n_ch);
    if ref_channel < 1 || ref_channel > n_ch
        ref_channel = 1;
    end

    for f = 1:n_freq
        Rs = squeeze(Phi_ss(f, :, :));
        Rn = squeeze(Phi_nn(f, :, :));
        Rs = (Rs + Rs') / 2;
        Rn = (Rn + Rn') / 2;

        % --- rank-1 speech model (phi_s, d) from dominant eigenpair ---------
        [phi_s, d] = rank1_model_(Rs, ref_channel, power_iters, eps_reg);

        % --- x = Phi_nn^{-1} d via conjugate gradient (inverse-free) --------
        x = cg_solve_(Rn, d, cg_iters, eps_reg);

        % --- SDW rank-1 weight ----------------------------------------------
        eta = d' * x;
        dr  = d(ref_channel);
        den = mu + phi_s * eta;
        if abs(den) > eps_reg
            w = (phi_s * conj(dr) / den) * x;
        else
            w = zeros(n_ch, 1);  w(ref_channel) = 1;
        end
        if ~all(isfinite(w))
            w = zeros(n_ch, 1);  w(ref_channel) = 1;
        end
        W(f, :) = w.';
    end
end

% =========================================================================
%  FILE-LOCAL KERNELS  (all ONNX-mappable: matmul / dot / scalar ops)
% =========================================================================
function [phi_s, d] = rank1_model_(Rs, ref, n_iter, eps_reg)
%RANK1_MODEL_  Dominant eigenpair of Hermitian Rs by power iteration.
%   d is phase/scale-normalized so d(ref) = 1; phi_s is the speech PSD at the
%   reference mic. n_iter = 0 returns the reference-column estimate, which is
%   exact for a rank-1 Rs.
    n  = size(Rs, 1);
    u  = Rs(:, ref);
    nu = norm(u);
    if nu < eps_reg
        u = zeros(n, 1);  u(ref) = 1;
    else
        u = u / nu;
    end
    for it = 1:n_iter
        y  = Rs * u;
        ny = norm(y);
        if ny < eps_reg, break; end
        u = y / ny;
    end
    lam  = real(u' * Rs * u);          % Rayleigh quotient (||u|| = 1)
    uref = u(ref);
    if abs(uref) > eps_reg
        d     = u / uref;              % d(ref) = 1
        phi_s = max(lam, 0) * abs(uref)^2;
    else
        d     = u;
        phi_s = max(lam, 0);
    end
end

function x = cg_solve_(A, b, K, eps_reg)
%CG_SOLVE_  Solve A x = b for Hermitian positive-definite A by complex
%   conjugate gradient. Exact in <= size(A) iterations (exact arithmetic).
    n  = numel(b);
    x  = zeros(n, 1);
    r  = b;                 % r = b - A*x0, x0 = 0
    p  = r;
    rs = real(r' * r);
    if rs < eps_reg^2
        return;             % b ~ 0  ->  x = 0
    end
    rs0 = rs;
    for k = 1:K
        Ap  = A * p;
        pAp = real(p' * Ap);
        if pAp <= eps_reg
            break;
        end
        a = rs / pAp;
        x = x + a * p;
        r = r - a * Ap;
        rs_new = real(r' * r);
        if rs_new <= eps_reg^2 * rs0
            break;
        end
        p  = r + (rs_new / rs) * p;
        rs = rs_new;
    end
end
