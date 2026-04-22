function G = get_tf_gain_map(X, cfg) %#ok<INUSD>
%GET_TF_GAIN_MAP  Neural-guided time-frequency gain-map placeholder.
%
%   G = get_tf_gain_map(X, cfg)
%
%   In the final Q-WiSE pipeline this routine is replaced by the output
%   of a quantized Transformer/Mamba hybrid model that predicts a per-
%   (t,f) real gain in [0,1].  For now we return all-ones so the MWF
%   pass-through degrades gracefully and the downstream wiring
%   (covariance buffers, STFT frames, etc.) can be validated end-to-end.
%
%   Inputs:
%     X    [nbin x n_mics]  current STFT frame (complex)
%     cfg                   full config struct
%
%   Output:
%     G    [nbin x 1]       per-bin gain, values in [0,1]

    nbin = size(X, 1);
    G    = ones(nbin, 1);
end
