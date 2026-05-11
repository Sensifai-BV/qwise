function W = mwf_compute_mvdr_weights(Phi_ss, Phi_nn, ref_channel, eps_reg, diag_load_ratio)
%MWF_COMPUTE_MVDR_WEIGHTS  MVDR beamformer with eigenvector-derived steering.
%
%   W = mwf_compute_mvdr_weights(Phi_ss, Phi_nn, ref_channel, eps_reg, diag_load_ratio)
%
%   Per-bin MVDR with the steering vector taken from the principal
%   eigenvector of Phi_ss(f):
%       a(f) = principal eigvec(Phi_ss(f))         (normalized so a_ref = 1)
%       W(f) = Phi_nn(f)^{-1} a / (a^H Phi_nn(f)^{-1} a)
%
%   Falls back to a unit vector on the reference channel if the denominator
%   collapses (e.g. zero noise covariance).
%
%   Inputs / outputs : same shapes as mwf_compute_mwf_weights.

    if nargin < 4 || isempty(eps_reg),          eps_reg = 1e-10; end
    if nargin < 5 || isempty(diag_load_ratio),  diag_load_ratio = 1e-4; end

    [n_freq, n_ch, ~] = size(Phi_ss);
    W                 = zeros(n_freq, n_ch);
    I_ch              = eye(n_ch);

    for f = 1:n_freq
        Rs = squeeze(Phi_ss(f, :, :));
        Rn = squeeze(Phi_nn(f, :, :));

        Rs = (Rs + Rs') / 2;
        [V, D] = eig(Rs);
        [~, k] = max(real(diag(D)));
        a      = V(:, k);
        if abs(a(ref_channel)) > eps_reg
            a = a / a(ref_channel);
        end

        load = max(eps_reg, diag_load_ratio * real(trace(Rn)) / n_ch);
        Rn_r = Rn + load * I_ch;

        try
            Rn_inv = Rn_r \ I_ch;
        catch
            Rn_inv = pinv(Rn_r);
        end

        num   = Rn_inv * a;
        denom = a' * num;
        if abs(denom) > eps_reg
            w = num / denom;
        else
            w = zeros(n_ch, 1);
            w(ref_channel) = 1;
        end
        W(f, :) = w.';
    end
end
