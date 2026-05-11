function W = mwf_compute_gev_weights(Phi_ss, Phi_nn, ref_channel, eps_reg, diag_load_ratio)
%MWF_COMPUTE_GEV_WEIGHTS  Generalized-Eigenvalue (Max-SNR) beamformer.
%
%   W = mwf_compute_gev_weights(Phi_ss, Phi_nn, ref_channel, eps_reg, diag_load_ratio)
%
%   For every frequency bin, solves the generalized eigenvalue problem
%       A w = lambda B w        where  A = Phi_ss(f) + ε I, B = Phi_nn(f) + ε I
%   and selects the eigenvector with the largest eigenvalue, i.e. the
%   direction that maximises w^H Phi_ss w / w^H Phi_nn w. The eigenvector is
%   phase-normalized to the reference mic.
%
%   GEV is the most robust beamformer for spatially non-stationary noise
%   (drone rotor blades) — recommended default in the Q-WiSE reference
%   Python pipeline.
%
%   Falls back to B^{-1}A eigendecomposition or e_ref if the GEP solve
%   fails (very rare; usually only at f = 0 for synthetic test inputs).

    if nargin < 4 || isempty(eps_reg),          eps_reg = 1e-10; end
    if nargin < 5 || isempty(diag_load_ratio),  diag_load_ratio = 1e-4; end

    [n_freq, n_ch, ~] = size(Phi_ss);
    W                 = zeros(n_freq, n_ch);
    I_ch              = eye(n_ch);

    for f = 1:n_freq
        Rs = squeeze(Phi_ss(f, :, :));
        Rn = squeeze(Phi_nn(f, :, :));

        load_ss = max(eps_reg, diag_load_ratio * real(trace(Rs)) / n_ch);
        load_nn = max(eps_reg, diag_load_ratio * real(trace(Rn)) / n_ch);
        A = (Rs + Rs') / 2 + load_ss * I_ch;
        B = (Rn + Rn') / 2 + load_nn * I_ch;

        try
            [V, D] = eig(A, B);
            [~, k] = max(real(diag(D)));
            w      = V(:, k);
        catch
            try
                [V, D] = eig(B \ A);
                [~, k] = max(real(diag(D)));
                w      = V(:, k);
            catch
                w = zeros(n_ch, 1);
                w(ref_channel) = 1;
            end
        end
        if ~all(isfinite(w))
            w = zeros(n_ch, 1);
            w(ref_channel) = 1;
        end

        if abs(w(ref_channel)) > eps_reg
            w = w / w(ref_channel);
        end
        W(f, :) = w.';
    end
end
