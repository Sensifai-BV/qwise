function X = mwf_stft(x, n_fft, hop)
%MWF_STFT  Short-time Fourier transform for the Q-WiSE batch MWF pipeline.
%
%   X = mwf_stft(x, n_fft, hop)
%
%   Matches the Python reference (scipy.signal.stft, Hann window, asymmetric
%   form, hop = n_fft - noverlap, zero-padded boundary).
%
%   Inputs:
%     x      [n_samples x 1]  mono signal (column)
%     n_fft  scalar           FFT size  (analysis window length)
%     hop    scalar           hop size between frames
%
%   Output:
%     X      [n_freq x n_frames]  one-sided complex STFT (n_freq = n_fft/2+1)

    x = x(:);
    if ~isfinite(n_fft) || n_fft <= 0
        error('mwf_stft:bad_n_fft', 'n_fft must be a positive integer');
    end
    if ~isfinite(hop) || hop <= 0
        error('mwf_stft:bad_hop', 'hop must be a positive integer');
    end

    win = hann(n_fft, 'periodic');

    % Zero-pad both ends so that the first/last hops are centred on x(1)/x(end).
    pad = n_fft / 2;
    xp  = [zeros(pad, 1); x; zeros(pad + n_fft, 1)];

    n_frames = floor((numel(xp) - n_fft) / hop) + 1;
    n_freq   = n_fft / 2 + 1;
    X        = zeros(n_freq, n_frames);

    for t = 1:n_frames
        start = (t - 1) * hop + 1;
        frame = xp(start:start + n_fft - 1) .* win;
        Xf    = fft(frame, n_fft);
        X(:, t) = Xf(1:n_freq);
    end
end
