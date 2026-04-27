function out = expand_channels(raw, geo, n_mics)
%EXPAND_CHANNELS  Build a virtual n-mic array from mono/stereo HW capture.
%
%   out = expand_channels(raw, geo, n_mics)
%
%   Inputs:
%     raw    [N x n_hw]  hardware audio frame (mono for MX2H3)
%     geo                geometry struct from build_geometry()
%     n_mics             desired virtual-array size
%
%   Output:
%     out    [N x n_mics]  per-mic time-delayed replicas.
%
%   If the hardware has >= 2 channels they are wired to the outer mics
%   directly (so a stereo capture stays natural) and the middle mic(s)
%   stay as delayed copies of channel 1.

    N   = size(raw, 1);
    out = zeros(N, n_mics);
    src = raw(:, 1);
    for m = 1:n_mics
        tau = geo.delays(m);
        if tau == 0
            out(:, m) = src;
        elseif tau < N
            out(:, m) = [zeros(tau, 1); src(1:N-tau)];
        end
    end
    n_hw = size(raw, 2);
    if n_hw >= 2 && n_mics >= 2
        out(:, 1)      = raw(:, 1);
        out(:, n_mics) = raw(:, n_hw);
    end
end
