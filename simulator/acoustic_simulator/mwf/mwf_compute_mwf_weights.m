function W = mwf_compute_mwf_weights(Phi_ss, Phi_nn, ref_channel, mu, eps_reg, diag_load_ratio)
%MWF_COMPUTE_MWF_WEIGHTS  Speech-Distortion-Weighted Multi-channel Wiener Filter.
%
%   W = mwf_compute_mwf_weights(Phi_ss, Phi_nn, ref_channel, mu, eps_reg, diag_load_ratio)
%
%   Closed-form SDW-MWF weight per frequency bin:
%       W(f) = Phi_ss(f) * inv(Phi_ss(f) + mu*Phi_nn(f)) * e_ref
%
%   Inputs:
%     Phi_ss          [n_freq x n_ch x n_ch]  speech covariance
%     Phi_nn          [n_freq x n_ch x n_ch]  noise covariance
%     ref_channel     scalar                  1-based reference mic index
%     mu              scalar                  speech-distortion trade-off
%     eps_reg         scalar                  floor for diagonal loading
%     diag_load_ratio scalar                  trace-proportional loading ratio
%
%   Output:
%     W   [n_freq x n_ch]  complex beamformer weights

    if nargin < 5 || isempty(eps_reg),          eps_reg = 1e-10; end
    if nargin < 6 || isempty(diag_load_ratio),  diag_load_ratio = 1e-4; end

    [n_freq, n_ch, ~] = size(Phi_ss);
    W                 = zeros(n_freq, n_ch);
    e_ref             = zeros(n_ch, 1);
    e_ref(ref_channel) = 1;

    for f = 1:n_freq
        Rs = squeeze(Phi_ss(f, :, :));
        Rn = squeeze(Phi_nn(f, :, :));
        M  = Rs + mu * Rn;

        load = max(eps_reg, diag_load_ratio * real(trace(M)) / n_ch);
        M    = M + load * eye(n_ch);

        try
            Minv = M \ eye(n_ch);
        catch
            Minv = pinv(M);
        end

        W(f, :) = (Rs * Minv * e_ref).';
    end
end
