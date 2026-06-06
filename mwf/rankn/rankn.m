function y = rankn(x, fs, opts)
%RANKN  N-microphone rank-1 multichannel Wiener filter speech enhancer.
%
%   y = RANKN(x, fs)
%   y = RANKN(x, fs, opts)
%   RANKN                                  % run a self-contained demo
%
%   RANKN enhances noisy multichannel speech (e.g. a small microphone array on
%   a drone) and returns a single denoised channel. It is a self-contained
%   reference implementation: STFT, voice-activity detection, spatial filtering
%   and post-filtering are all in this one file, with no toolbox dependencies
%   beyond core MATLAB (uses FFT and EXPINT).
%
%   ALGORITHM
%   ---------
%   Per short-time frequency bin the enhancer applies a rank-1 speech-model
%   multichannel Wiener filter (the MWF / SDW-MWF):
%
%       w(f) = phi_s * conj(d_ref) * Phi_nn^{-1} d / (mu + phi_s * d^H Phi_nn^{-1} d)
%
%   where d is the speech steering vector (normalized so its reference-mic
%   entry is 1), phi_s is the speech power at the reference mic, Phi_nn is the
%   noise spatial covariance, and mu >= 1 is a speech-distortion / noise-
%   suppression trade-off (mu = 1 is the plain MWF). By the Woodbury identity
%   this is exactly (Phi_ss + mu*Phi_nn)^{-1} Phi_ss e_ref for a rank-1 speech
%   covariance Phi_ss = phi_s * d d^H, and for N = 2,3 it coincides with the
%   closed-form 2x2 / 3x3 solutions.
%
%   Two design choices make it both N-microphone-general and exportable to
%   fixed compute graphs (e.g. ONNX) — it avoids EIG and avoids forming a
%   matrix inverse:
%
%     * (phi_s, d) is the dominant eigenpair of the speech covariance, obtained
%       by POWER ITERATION seeded with the reference column of Phi_ss. For an
%       ideal rank-1 Phi_ss that column already is d, so zero iterations is the
%       exact reference-column estimate and a few iterations refine it when the
%       estimated covariance is only approximately rank-1.
%     * x = Phi_nn^{-1} d is obtained by complex CONJUGATE GRADIENT, which is
%       exact in at most N steps for a Hermitian positive-definite Phi_nn.
%
%   The spatial filter raises the SNR but, with a physically small array, has
%   little directivity at low frequencies — it cannot null tonal rotor noise
%   that overlaps the speech band. A single-channel OMLSA post-filter (log-
%   spectral-amplitude gain with decision-directed a-priori SNR and minimum-
%   statistics noise tracking) follows the beamformer to drive that residual
%   toward zero while limiting musical noise. It can be switched off via opts.
%
%   INPUTS
%   ------
%     x    [L x M]  multichannel time-domain signal (one column per mic), or
%                   [L x 1] mono (only the OMLSA stage applies in that case).
%     fs   scalar   sample rate; 8000 or 16000 Hz are the calibrated rates.
%     opts struct   optional overrides (all fields optional):
%                     .ref_mic       reference mic, 1-based            (1)
%                     .mu            SDW trade-off, >= 1               (1.0)
%                     .power_iters   power-iteration refinement steps  (3)
%                     .cg_iters      conjugate-gradient iterations     (8)
%                                    (>= number of mics is exact)
%                     .eps_reg       diagonal loading / numeric floor  (1e-5)
%                     .post_omlsa    apply OMLSA post-filter           (true)
%                     .omlsa_floor_db spectral floor in dB            (-30)
%                     .n_fft/.hop    override STFT geometry            (per fs)
%
%   OUTPUT
%   ------
%     y    [L x 1]  enhanced mono signal, peak-normalized to 0.9.
%
%   EXAMPLE
%   -------
%     % 16 kHz, 3-mic noisy recording -> enhanced mono file
%     [x, fs] = audioread('noise_speech.wav');   % x is [L x 3]
%     y = rankn(x, fs);
%     audiowrite('enhanced.wav', y, fs);
%     % soundsc(y, fs);                          % listen
%
%     % stronger suppression on a 5-mic array
%     opts.mu = 2.0; opts.cg_iters = 5;
%     y = rankn(x, fs, opts);
%
%     % see it run end-to-end on a synthetic mixture
%     rankn
%
%   See also FFT, IFFT, EXPINT.

    if nargin == 0
        y = run_demo_();                 % `rankn` with no args runs the demo
        if nargout == 0, clear y; end
        return;
    end
    if nargin < 3, opts = struct(); end

    cfg = defaults_(fs, opts);
    if isvector(x), x = x(:); end
    L = size(x, 1);

    % --- analysis -------------------------------------------------------
    X = stft_multi_(x, cfg.n_fft, cfg.hop);            % [F x T x M] complex
    [speech, noise] = energy_masks_(X, cfg.ref_mic, cfg.vad_speech_pct, cfg.vad_noise_pct);

    % --- spatial rank-N MWF --------------------------------------------
    [Rss, Rnn] = covariances_(X, speech, noise, cfg.eps_reg);
    W = rankn_weights_(Rss, Rnn, cfg.ref_mic, cfg.mu, ...
                       cfg.power_iters, cfg.cg_iters, cfg.eps_reg);
    Y = beamform_(W, X);                               % [F x T] complex

    % --- single-channel post-filter ------------------------------------
    if cfg.post_omlsa
        Y = omlsa_(Y, cfg.omlsa_floor_db, cfg.omlsa_alpha_dd, ...
                   cfg.omlsa_alpha_s, cfg.omlsa_alpha_d, cfg.omlsa_win_min);
    end

    % --- synthesis ------------------------------------------------------
    y = istft_(Y, cfg.n_fft, cfg.hop, L);
    pk = max(abs(y));
    if pk > 0, y = y / pk * 0.9; end
end


% =====================================================================
%  CONFIGURATION
% =====================================================================
function cfg = defaults_(fs, opts)
    switch fs
        case 16000, cfg.n_fft = 512; cfg.hop = 128;
        case 8000,  cfg.n_fft = 256; cfg.hop = 64;
        otherwise
            % Unlisted rate: pick ~32 ms / 8 ms and warn that it is uncalibrated.
            cfg.n_fft = 2^round(log2(0.032 * fs));
            cfg.hop   = round(cfg.n_fft / 4);
            warning('rankn:fs', ...
                ['fs = %g Hz is not a calibrated rate (use 8000 or 16000); ' ...
                 'using n_fft = %d, hop = %d.'], fs, cfg.n_fft, cfg.hop);
    end
    cfg.fs              = fs;
    cfg.ref_mic         = 1;
    cfg.mu              = 1.0;
    cfg.power_iters     = 3;
    cfg.cg_iters        = 8;
    cfg.eps_reg         = 1e-5;     % >= ~1e-6 keeps a single-precision port stable
    cfg.post_omlsa      = true;
    cfg.omlsa_floor_db  = -30;
    cfg.omlsa_alpha_dd  = 0.92;     % decision-directed a-priori-SNR smoothing
    cfg.omlsa_alpha_s   = 0.90;     % power smoothing for noise tracking
    cfg.omlsa_alpha_d   = 0.85;     % noise-PSD update rate
    cfg.omlsa_win_min   = 60;       % minimum-statistics window (frames)
    cfg.vad_speech_pct  = 60;       % frames above this energy pct -> speech
    cfg.vad_noise_pct   = 35;       % frames below this energy pct -> noise

    % apply user overrides
    fn = fieldnames(opts);
    for k = 1:numel(fn)
        cfg.(fn{k}) = opts.(fn{k});
    end
end


% =====================================================================
%  STFT / iSTFT  (sqrt-Hann, 75% overlap, COLA)
% =====================================================================
function w = sqrt_hann_(n)
    m = (0:n-1).';
    w = sqrt(0.5 - 0.5 * cos(2 * pi * m / n));   % periodic Hann, square-rooted
end

function X = stft_multi_(x, n_fft, hop)
    [L, M] = size(x);
    win = sqrt_hann_(n_fft);
    F   = n_fft / 2 + 1;
    T   = 1 + max(0, floor((L - n_fft) / hop));
    X   = zeros(F, T, M);
    for m = 1:M
        for t = 1:T
            seg = x((t-1)*hop + (1:n_fft), m) .* win;
            S   = fft(seg, n_fft);
            X(:, t, m) = S(1:F);
        end
    end
end

function y = istft_(Y, n_fft, hop, L)
    win = sqrt_hann_(n_fft);
    [~, T] = size(Y);
    Lf  = (T - 1) * hop + n_fft;
    out = zeros(Lf, 1);
    den = zeros(Lf, 1);
    for t = 1:T
        full = [Y(:, t); conj(Y(end-1:-1:2, t))];   % Hermitian -> full spectrum
        seg  = real(ifft(full)) .* win;
        idx  = (t-1)*hop + (1:n_fft);
        out(idx) = out(idx) + seg;
        den(idx) = den(idx) + win.^2;
    end
    y = out ./ max(den, 1e-8);
    if nargin > 3 && ~isempty(L), y = y(1:min(L, numel(y))); end
end


% =====================================================================
%  ENERGY VAD  ->  speech / noise frame masks
% =====================================================================
function [speech, noise] = energy_masks_(X, ref, speech_pct, noise_pct)
    P   = squeeze(sum(abs(X(:, :, ref)).^2, 1));     % [T] frame power
    Pdb = 10 * log10(P(:) + 1e-12);
    speech = Pdb >= prctile_(Pdb, speech_pct);
    noise  = Pdb <= prctile_(Pdb, noise_pct);
    if sum(speech) < 3, speech = Pdb >= prctile_(Pdb, 50); end
    if sum(noise)  < 3, noise  = Pdb <= prctile_(Pdb, 50); end
end

function v = prctile_(a, p)
    % Toolbox-free percentile (linear interpolation), p in [0,100].
    a = sort(a(:));
    n = numel(a);
    if n == 1, v = a; return; end
    r = (p/100) * (n - 1) + 1;
    lo = floor(r); hi = ceil(r);
    v = a(lo) + (r - lo) * (a(hi) - a(lo));
end


% =====================================================================
%  SPATIAL COVARIANCES  (masked average -> Hermitian PSD)
% =====================================================================
function [Rss, Rnn] = covariances_(X, speech, noise, eps_reg)
    [F, ~, M] = size(X);
    ms = double(speech(:)); mn = double(noise(:));
    sw = max(sum(ms), 1); nw = max(sum(mn), 1);
    Rss = zeros(F, M, M); Rnn = zeros(F, M, M);
    I = eye(M);
    for f = 1:F
        A = squeeze(X(f, :, :));            % [T x M]
        if size(A, 2) ~= M, A = reshape(A, [], M); end
        Rs = A.' * (ms .* conj(A)) / sw;    % sum_t ms(t) x_t x_t^H
        Rn = A.' * (mn .* conj(A)) / nw;
        Rs = (Rs + Rs') / 2;                % Hermitian symmetrize
        Rn = (Rn + Rn') / 2 + eps_reg * I;  % + diagonal loading
        Rss(f, :, :) = Rs;
        Rnn(f, :, :) = Rn;
    end
end


% =====================================================================
%  RANK-N MWF WEIGHTS  (power iteration + conjugate gradient, per bin)
% =====================================================================
function W = rankn_weights_(Rss, Rnn, ref, mu, power_iters, cg_iters, eps_reg)
    [F, M, ~] = size(Rss);
    W = zeros(F, M);
    e_ref = zeros(M, 1); e_ref(ref) = 1;
    for f = 1:F
        Rs = squeeze(Rss(f, :, :));
        Rn = squeeze(Rnn(f, :, :));

        [phi, d] = rank1_model_(Rs, ref, power_iters, eps_reg);
        x = cg_solve_(Rn, d, cg_iters, eps_reg);

        eta = d' * x;
        den = mu + phi * eta;
        if abs(den) > eps_reg && all(isfinite(x))
            w = (phi * conj(d(ref)) / den) * x;
        else
            w = e_ref;
        end
        W(f, :) = w.';
    end
end

function [phi, d] = rank1_model_(Rs, ref, n_iter, eps_reg)
    % Dominant eigenpair of Hermitian Rs by power iteration (ref-column init).
    % d is scaled so d(ref) = 1; phi is the speech PSD at the reference mic.
    M = size(Rs, 1);
    u = Rs(:, ref);
    nu = norm(u);
    if nu < eps_reg, u = zeros(M, 1); u(ref) = 1; else, u = u / nu; end
    for it = 1:n_iter
        z  = Rs * u;
        nz = norm(z);
        if nz < eps_reg, break; end
        u = z / nz;
    end
    lam  = real(u' * Rs * u);              % Rayleigh quotient (||u|| = 1)
    uref = u(ref);
    if abs(uref) > eps_reg
        d   = u / uref;
        phi = max(lam, 0) * abs(uref)^2;
    else
        d   = u;
        phi = max(lam, 0);
    end
end

function x = cg_solve_(A, b, K, eps_reg)
    % Solve A x = b for Hermitian positive-definite A by complex conjugate
    % gradient. Exact in <= size(A) iterations in exact arithmetic.
    n  = numel(b);
    x  = zeros(n, 1);
    r  = b;
    p  = r;
    rs = real(r' * r);
    if rs < eps_reg^2, return; end
    rs0 = rs;
    for k = 1:K
        Ap  = A * p;
        pAp = real(p' * Ap);
        if pAp <= eps_reg, break; end
        a = rs / pAp;
        x = x + a * p;
        r = r - a * Ap;
        rs_new = real(r' * r);
        if rs_new <= eps_reg^2 * rs0, break; end
        p  = r + (rs_new / rs) * p;
        rs = rs_new;
    end
end

function Y = beamform_(W, X)
    % Y(f,t) = w(f)^H x(f,t)  over all bins/frames.
    [F, T, ~] = size(X);
    Y = zeros(F, T);
    for f = 1:F
        Xf = squeeze(X(f, :, :)).';        % [M x T]
        Y(f, :) = conj(W(f, :)) * Xf;
    end
end


% =====================================================================
%  OMLSA POST-SUPPRESSOR  (LSA gain + decision-directed SNR + min-stats)
% =====================================================================
function Yout = omlsa_(Y, floor_db, a_dd, a_s, a_d, win_min)
    [F, T] = size(Y);
    P    = abs(Y).^2;
    Gmin = 10^(floor_db / 20);
    n0   = min(8, T);
    lam  = max(mean(P(:, 1:n0), 2), 1e-10);    % initial noise PSD
    S    = lam;
    Gp   = ones(F, 1);
    gam_p = ones(F, 1);
    minbuf = repmat(S, 1, win_min);
    bptr   = 1;
    Yout = zeros(F, T);
    for t = 1:T
        p = P(:, t);
        S = a_s * S + (1 - a_s) * p;            % smoothed power
        minbuf(:, bptr) = S;
        bptr = bptr + 1; if bptr > win_min, bptr = 1; end
        Smin = min(minbuf, [], 2);              % minimum statistics

        Sr    = S ./ max(Smin, 1e-12);
        ppres = min(max((Sr - 1) / 8, 0), 1);   % soft speech-presence prob

        lam = max(lam + (1 - ppres) .* (a_d * lam + (1 - a_d) * p - lam), 1e-12);

        gamma = p ./ lam;                       % a posteriori SNR
        xi    = max(a_dd * Gp.^2 .* gam_p + (1 - a_dd) * max(gamma - 1, 0), 1e-3);
        nu    = min(max(xi ./ (1 + xi) .* gamma, 1e-6), 500);
        G_lsa = xi ./ (1 + xi) .* exp(0.5 * expint(nu));   % log-spectral amplitude
        G     = min(max(G_lsa.^ppres .* Gmin.^(1 - ppres), Gmin), 1);

        Yout(:, t) = G .* Y(:, t);
        Gp = G; gam_p = gamma;
    end
end


% =====================================================================
%  SELF-CONTAINED DEMO  (run `rankn` with no arguments)
% =====================================================================
function y = run_demo_()
    fs = 16000; M = 3; secs = 4;
    [x, clean] = synth_mixture_(fs, secs, M, 0);    % 3-mic, 0 dB input SNR
    y = rankn(x, fs);
    fprintf('rankn demo: %d ch, %g Hz, %.1fs\n', M, fs, secs);
    fprintf('  input  SI-SDR (mic1) : %5.1f dB\n', si_sdr_(x(:,1), clean));
    fprintf('  rankn  SI-SDR        : %5.1f dB\n', si_sdr_(y, clean));
    fprintf('  improvement          : %+5.1f dB\n', ...
            si_sdr_(y, clean) - si_sdr_(x(:,1), clean));
    fprintf('  (soundsc(y,%d) to listen)\n', fs);
end

function [mix, ref] = synth_mixture_(fs, secs, M, snr_db)
    % Harmonic "speech" bursts + a drone-rotor harmonic comb, placed on a
    % linear array with per-mic fractional delays.
    L = round(secs * fs);
    t = (0:L-1).' / fs;
    env = zeros(L, 1);
    for c = [0.4 1.2; 1.7 2.5; 3.0 3.7].'
        env(round(c(1)*fs)+1 : round(c(2)*fs)) = 1;
    end
    k = max(1, round(0.025 * fs));
    env = filter(ones(k,1)/k, 1, env);
    sp = zeros(L, 1);
    for h = 1:min(40, floor(fs/(2*140))), sp = sp + (1/h) * sin(2*pi*140*h*t); end
    sp = (sp .* env); sp = sp / (std(sp) + 1e-9);
    rotor = zeros(L, 1);
    for h = 1:7, rotor = rotor + sin(2*pi*133*h*t + 2*pi*rand); end
    rotor = rotor / (std(rotor) + 1e-9);
    ds = linspace(0, 2.0, M);
    dn = linspace(0, 1.0, M) + 0.6;
    g  = 10^(-snr_db/20);
    mix = zeros(L, M);
    for m = 1:M
        mix(:, m) = fracdelay_(sp, ds(m)) + g * fracdelay_(rotor, dn(m));
    end
    ref = fracdelay_(sp, ds(1));
    mix = mix / (max(abs(mix(:))) + 1e-9);
end

function z = fracdelay_(s, d)
    n = (0:numel(s)-1).';
    z = interp1(n, s, n - d, 'linear', 0);
end

function v = si_sdr_(est, ref)
    m = min(numel(est), numel(ref));
    est = est(1:m) - mean(est(1:m));
    ref = ref(1:m) - mean(ref(1:m));
    a   = (est' * ref) / (ref' * ref + 1e-12);
    tgt = a * ref;
    v   = 10 * log10((tgt'*tgt) / ((est-tgt)'*(est-tgt) + 1e-12) + 1e-12);
end
