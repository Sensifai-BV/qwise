function x = mwf_istft(X, n_fft, hop)
%MWF_ISTFT  Inverse short-time Fourier transform (overlap-add synthesis).
%
%   x = mwf_istft(X, n_fft, hop)
%
%   Performs weighted overlap-add reconstruction using the same Hann window
%   used at analysis. Pairs with mwf_stft for perfect reconstruction (up to
%   the boundary padding it added).
%
%   Inputs:
%     X      [n_freq x n_frames]  one-sided complex STFT
%     n_fft  scalar               original FFT size
%     hop    scalar               original hop size
%
%   Output:
%     x      [n_samples x 1]  time-domain mono signal

    win    = hann(n_fft, 'periodic');
    [n_freq, n_frames] = size(X);
    if n_freq ~= n_fft / 2 + 1
        error('mwf_istft:bad_shape', ...
              'STFT bin count (%d) does not match n_fft/2+1 (%d).', ...
              n_freq, n_fft / 2 + 1);
    end

    n_total = (n_frames - 1) * hop + n_fft;
    x       = zeros(n_total, 1);
    wsum    = zeros(n_total, 1);

    for t = 1:n_frames
        Xf = X(:, t);
        Xfull = [Xf; conj(Xf(end - 1:-1:2))];
        frame = real(ifft(Xfull, n_fft));
        start = (t - 1) * hop + 1;
        idx   = start:start + n_fft - 1;
        x(idx)    = x(idx)    + frame .* win;
        wsum(idx) = wsum(idx) + win .* win;
    end

    nz       = wsum > 1e-12;
    x(nz)    = x(nz) ./ wsum(nz);

    % Trim the symmetric zero-padding added by mwf_stft.
    pad = n_fft / 2;
    x   = x(pad + 1:end);
    if numel(x) > n_total - pad
        x = x(1:n_total - pad);
    end
end
