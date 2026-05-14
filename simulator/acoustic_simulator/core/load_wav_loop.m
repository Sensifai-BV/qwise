function y = load_wav_loop(wav_path, target_fs, loop_sec)
%LOAD_WAV_LOOP  Read a WAV and tile it into a looped mono buffer.
%
%   y = load_wav_loop(path, fs, seconds) returns a [fs*seconds x 1] mono
%   buffer normalised to unit RMS.  Paths that are not absolute are
%   resolved relative to the project root (parent of the 'core' folder).

    if ~(startsWith(wav_path, filesep) || ...
         (numel(wav_path) >= 2 && wav_path(2) == ':'))
        proj_root = fileparts(fileparts(mfilename('fullpath')));
        wav_path  = fullfile(proj_root, wav_path);
    end

    if ~isfile(wav_path)
        fprintf('[Q-WiSE] %s not found — generating pink-noise placeholder\n', ...
                wav_path);
        N = round(loop_sec * target_fs);
        w = randn(N, 1);
        [b, a] = butter(1, 0.015);
        y = filter(b, a, w);
    else
        fprintf('[Q-WiSE] Loading %s\n', wav_path);
        [y, fs_orig] = audioread(wav_path);
        if size(y, 2) > 1
            y = mean(y, 2);
        end
        if fs_orig ~= target_fs
            y = resample(y, target_fs, fs_orig);
        end
        n_rep = ceil(loop_sec * target_fs / length(y));
        y = repmat(y, n_rep, 1);
        y = y(1:round(loop_sec * target_fs));
    end
    y = y / (rms(y) + 1e-12);
end
