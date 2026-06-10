classdef AudioIO < handle
%AUDIOIO  File-based audio manager for the Q-WiSE simulator.
%
%   Loads the looped drone-fan and environment noise buffers, lists and
%   loads the clean-speech samples, and writes recordings of the noisy
%   array + ONNX-enhanced output to disk.
%
%   obj   = AudioIO(cfg)
%   names = obj.list_speech_samples()       % cellstr of *.wav in samples_dir
%   y     = obj.load_speech_sample(name)    % mono [L x 1] @ cfg.fs, normalised
%   c     = obj.next_drone_chunk(N)         % N samples of the drone loop
%   c     = obj.next_env_chunk(N)           % N samples of the env loop
%   dir   = obj.save_recording(mic, clean)  % write mic01..NN.wav + clean.wav

    properties
        cfg
        drone_wav
        env_wav
    end

    properties (Access = private)
        drone_ptr = 1
        env_ptr   = 1
    end

    methods
        function obj = AudioIO(cfg)
            obj.cfg = cfg;
            obj.drone_wav = load_wav_loop(cfg.drone_wav_path, cfg.fs, cfg.loop_sec);
            obj.env_wav   = load_wav_loop(cfg.env_wav_path,   cfg.fs, cfg.loop_sec);
        end

        % ---------------- Clean-speech samples ---------------------
        function names = list_speech_samples(obj)
        %LIST_SPEECH_SAMPLES  Sorted list of *.wav filenames in samples_dir.
            names = {};
            d = obj.cfg.samples_dir;
            if ~isfolder(d), return; end
            w = dir(fullfile(d, '*.wav'));
            if isempty(w), return; end
            [~, ord] = sort({w.name});
            names = {w(ord).name};
        end

        function y = load_speech_sample(obj, name)
        %LOAD_SPEECH_SAMPLE  Read a sample, mono, resample to cfg.fs, and
        %   peak-normalise to 0.9 so it behaves like a well-levelled talker.
            path = name;
            if ~isfile(path)
                path = fullfile(obj.cfg.samples_dir, name);
            end
            [y, fs0] = audioread(path);
            if size(y, 2) > 1
                y = mean(y, 2);
            end
            if fs0 ~= obj.cfg.fs
                y = resample(y, obj.cfg.fs, fs0);
            end
            pk = max(abs(y));
            if pk > 0
                y = y / pk * 0.9;
            end
            y = y(:);
        end

        % ---------------- Looped noise sources ---------------------
        function c = next_drone_chunk(obj, N)
            c = loop_chunk(obj.drone_wav, obj.drone_ptr, N);
            obj.drone_ptr = mod(obj.drone_ptr + N - 1, length(obj.drone_wav)) + 1;
        end

        function c = next_env_chunk(obj, N)
            c = loop_chunk(obj.env_wav, obj.env_ptr, N);
            obj.env_ptr = mod(obj.env_ptr + N - 1, length(obj.env_wav)) + 1;
        end

        function reset_noise(obj)
        %RESET_NOISE  Rewind both noise loops to the start.
            obj.drone_ptr = 1;
            obj.env_ptr   = 1;
        end

        % ---------------- Recording --------------------------------
        function dst = save_recording(obj, mic, clean)
        %SAVE_RECORDING  Write the noisy array + enhanced output to disk.
        %   mic   : [L x N] noisy microphone channels -> mic01.wav..micNN.wav
        %   clean : [L x 1] ONNX-enhanced speech       -> clean.wav
        %   Returns the timestamped folder the files were written into.
            dst = '';
            if isempty(mic), return; end
            base = obj.cfg.record.dir;
            if ~isfolder(base), mkdir(base); end
            stamp = datestr(now, 'yyyymmdd_HHMMSS'); %#ok<DATST,TNOW1>
            dst   = fullfile(base, sprintf('%s_%s', obj.cfg.record.prefix, stamp));
            if ~isfolder(dst), mkdir(dst); end

            for m = 1:size(mic, 2)
                fn = fullfile(dst, sprintf('mic%02d.wav', m));
                audiowrite(fn, obj.clip_(mic(:, m)), obj.cfg.fs, 'BitsPerSample', 16);
            end
            if ~isempty(clean)
                audiowrite(fullfile(dst, 'clean.wav'), obj.clip_(clean(:)), ...
                           obj.cfg.fs, 'BitsPerSample', 16);
            end
            fprintf('[Q-WiSE] Recording saved -> %s\n', dst);
        end

        function release(~)
        %RELEASE  No live devices to free in the file-based pipeline.
        end
    end

    methods (Access = private)
        function y = clip_(~, x)
            y = max(min(x, 1), -1);
        end
    end
end
