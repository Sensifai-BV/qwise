function mask = mwf_build_speech_mask(aligned_vad, n_frames, n_fft, hop, threshold, context_frames)
%MWF_BUILD_SPEECH_MASK  Frame-level speech/noise mask from aligned VAD audio.
%
%   mask = mwf_build_speech_mask(aligned_vad, n_frames, n_fft, hop, threshold, context_frames)
%
%   For each STFT analysis frame, computes the RMS energy of the aligned VAD
%   audio inside that frame. Frames whose normalized energy exceeds the
%   threshold are marked as speech; the result is then dilated by
%   `context_frames` to safely include transitions.
%
%   Inputs:
%     aligned_vad     [Lm x 1]  full-length VAD-extracted speech (zeros where silent)
%     n_frames        scalar    number of STFT frames produced by mwf_stft
%     n_fft, hop      scalars   STFT geometry
%     threshold       scalar    fraction of peak RMS that counts as speech
%     context_frames  scalar    dilation radius around each detected frame
%
%   Output:
%     mask  [n_frames x 1] logical  true = speech frame

    if nargin < 5 || isempty(threshold),       threshold = 0.01; end
    if nargin < 6 || isempty(context_frames),  context_frames = 3; end

    aligned_vad = aligned_vad(:);
    pad         = n_fft / 2;
    xp          = [zeros(pad, 1); aligned_vad; zeros(pad + n_fft, 1)];

    energy = zeros(n_frames, 1);
    for t = 1:n_frames
        start = (t - 1) * hop + 1;
        stop  = start + n_fft - 1;
        if stop > numel(xp)
            break;
        end
        frame    = xp(start:stop);
        energy(t) = sqrt(mean(frame.^2));
    end

    peak = max(energy);
    if peak <= 0
        mask = false(n_frames, 1);
        return;
    end
    base_mask = (energy / peak) > threshold;

    % Dilate by context_frames in both directions.
    mask = base_mask;
    hits = find(base_mask);
    for k = 1:numel(hits)
        i = hits(k);
        a = max(1, i - context_frames);
        b = min(n_frames, i + context_frames);
        mask(a:b) = true;
    end
end
