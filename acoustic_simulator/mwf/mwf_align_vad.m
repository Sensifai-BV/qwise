function [lag, aligned_vad] = mwf_align_vad(vad_signal, mic_signal)
%MWF_ALIGN_VAD  Locate VAD-extracted speech inside a reference mic signal.
%
%   [lag, aligned_vad] = mwf_align_vad(vad_signal, mic_signal)
%
%   Uses normalized cross-correlation to find the temporal offset at which
%   the VAD-extracted speech best matches the reference mic recording.
%   Returns the lag (in samples) and a full-length signal with the VAD
%   audio placed at the correct position.
%
%   Inputs:
%     vad_signal  [Lv x 1]  VAD-output (extracted speech) audio
%     mic_signal  [Lm x 1]  reference mic capture (must be Lm >= Lv)
%
%   Outputs:
%     lag         scalar    0-based sample offset (0 = aligned to mic start)
%     aligned_vad [Lm x 1]  vad_signal padded with zeros around the lag

    vad_signal = vad_signal(:);
    mic_signal = mic_signal(:);

    if numel(vad_signal) > numel(mic_signal)
        % Symmetry: clip the longer signal so that cross-correlation in 'valid'
        % mode is well-defined.
        vad_signal = vad_signal(1:numel(mic_signal));
    end

    corr      = xcorr(mic_signal, vad_signal);
    % xcorr returns lags from -(N-1)..(N-1). We want non-negative lags only
    % where the VAD fits entirely inside the mic.
    mid       = numel(mic_signal);              % index of lag = 0
    max_lag   = numel(mic_signal) - numel(vad_signal);
    seg       = corr(mid:mid + max_lag);        % [0 .. max_lag]
    [~, k]    = max(abs(seg));
    lag       = k - 1;                          % 0-based offset

    aligned_vad         = zeros(numel(mic_signal), 1);
    copy_end            = min(lag + numel(vad_signal), numel(mic_signal));
    copy_len            = copy_end - lag;
    aligned_vad(lag + 1:lag + copy_len) = vad_signal(1:copy_len);
end
