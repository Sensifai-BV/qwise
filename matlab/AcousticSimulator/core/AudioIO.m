classdef AudioIO < handle
%AUDIOIO  Real-time microphone reader + speaker writer + looping noise WAVs.
%
%   obj = AudioIO(cfg)
%     Opens a MacBook built-in microphone (auto-detected via keyword list),
%     opens an audioDeviceWriter for low-latency playback, and preloads
%     two looped noise buffers (drone fan + environment ambient).
%
%   raw        = obj.read()                 % one frame from the mic
%                obj.play(x)                % push mono frame to speakers
%   chunk      = obj.next_drone_chunk(N)    % next N samples of drone loop
%   chunk      = obj.next_env_chunk(N)      % next N samples of env loop
%
%   Real-time recording (independent of VAD / MWF):
%                obj.rec_start(path)        % open a wav sink
%                obj.rec_write(x)           % append mono samples
%   path       = obj.rec_stop()             % close and return file path
%   tf         = obj.is_recording()         % true while recording
%
%                obj.release()              % free the audio devices

    properties
        cfg
        reader
        writer
        n_hw_ch = 1
        drone_wav
        env_wav
    end

    properties (Access = private)
        drone_ptr = 1
        env_ptr   = 1
        rec_active_ = false
        rec_buf_    = zeros(0, 1)   % in-memory accumulation
        rec_path_   = ''
        rec_cap_    = 0             % allocated capacity of rec_buf_
        rec_len_    = 0             % valid length in rec_buf_
    end

    methods
        function obj = AudioIO(cfg)
            obj.cfg = cfg;
            [obj.reader, obj.n_hw_ch] = obj.create_reader_();
            obj.writer = audioDeviceWriter('SampleRate', cfg.fs, ...
                                           'BufferSize', cfg.frame_size);
            obj.drone_wav = load_wav_loop(cfg.drone_wav_path, cfg.fs, cfg.loop_sec);
            obj.env_wav   = load_wav_loop(cfg.env_wav_path,   cfg.fs, cfg.loop_sec);
            fprintf('[Q-WiSE] Mic: %d hw channel(s) -> %d virtual channels\n', ...
                    obj.n_hw_ch, cfg.n_mics);
        end

        function raw = read(obj)
            raw = obj.reader();
        end

        function play(obj, x)
            x = max(min(x, 1), -1);
            try
                obj.writer(x);
            catch
            end
        end

        function c = next_drone_chunk(obj, N)
            c = loop_chunk(obj.drone_wav, obj.drone_ptr, N);
            obj.drone_ptr = mod(obj.drone_ptr + N - 1, length(obj.drone_wav)) + 1;
        end

        function c = next_env_chunk(obj, N)
            c = loop_chunk(obj.env_wav, obj.env_ptr, N);
            obj.env_ptr = mod(obj.env_ptr + N - 1, length(obj.env_wav)) + 1;
        end

        % ---------------- Recording --------------------------------
        % Recording uses an in-memory buffer that is flushed to disk
        % exactly once, on rec_stop().  This avoids any partial-header
        % issues seen with streaming system objects (dsp.AudioFileWriter)
        % when the file is inspected mid-recording: the WAV on disk is
        % always either absent or complete.

        function path = rec_start(obj, path)
        %REC_START  Begin accumulating a mono WAV recording.
        %   If PATH is omitted, generates a timestamped file under
        %   cfg.record.dir.  The file is NOT created until rec_stop().
            if nargin < 2 || isempty(path)
                path = obj.default_rec_path_();
            end
            [d, ~, ~] = fileparts(path);
            if ~isempty(d) && exist(d, 'dir') ~= 7
                mkdir(d);
            end
            % Pre-allocate ~30 s of capacity; grown geometrically on need.
            obj.rec_cap_    = 30 * obj.cfg.fs;
            obj.rec_buf_    = zeros(obj.rec_cap_, 1);
            obj.rec_len_    = 0;
            obj.rec_path_   = path;
            obj.rec_active_ = true;
            fprintf('[Q-WiSE] Recording started -> %s\n', path);
        end

        function rec_write(obj, x)
        %REC_WRITE  Append a mono block to the active recording buffer.
            if ~obj.rec_active_, return; end
            x  = max(min(x(:), 1), -1);
            nx = numel(x);
            need = obj.rec_len_ + nx;
            if need > obj.rec_cap_
                new_cap = max(need, 2 * obj.rec_cap_);
                tmp = zeros(new_cap, 1);
                tmp(1:obj.rec_len_) = obj.rec_buf_(1:obj.rec_len_);
                obj.rec_buf_ = tmp;
                obj.rec_cap_ = new_cap;
            end
            obj.rec_buf_(obj.rec_len_+1 : obj.rec_len_+nx) = x;
            obj.rec_len_ = obj.rec_len_ + nx;
        end

        function path = rec_stop(obj)
        %REC_STOP  Flush the in-memory buffer to a 16-bit WAV and return path.
            path = obj.rec_path_;
            if ~obj.rec_active_
                path = '';
                return;
            end
            obj.rec_active_ = false;
            y = obj.rec_buf_(1:obj.rec_len_);
            obj.rec_buf_    = zeros(0, 1);
            obj.rec_cap_    = 0;
            obj.rec_len_    = 0;
            obj.rec_path_   = '';
            try
                if ~isempty(y)
                    audiowrite(path, y, obj.cfg.fs, ...
                        'BitsPerSample', 16);
                    fprintf('[Q-WiSE] Recording stopped -> %s (%.2f s)\n', ...
                        path, numel(y) / obj.cfg.fs);
                else
                    fprintf(['[Q-WiSE] Recording stopped with ' ...
                             'no samples — file not written.\n']);
                    path = '';
                end
            catch ME
                warning('QWiSE:AudioIO:RecFlush', ...
                    '[Q-WiSE] rec flush failed: %s', ME.message);
                path = '';
            end
        end

        function tf = is_recording(obj)
            tf = obj.rec_active_;
        end

        function release(obj)
            obj.rec_stop();
            try release(obj.reader); catch, end
            try release(obj.writer); catch, end
            fprintf('[Q-WiSE] Audio devices released.\n');
        end
    end

    methods (Access = private)
        function [adr, n_ch] = create_reader_(obj)
            c   = obj.cfg;
            dev = find_mac_mic();
            n_ch = probe_channels(dev, c.fs, c.frame_size, c.n_mics);
            try
                if isempty(dev)
                    adr = audioDeviceReader('SampleRate', c.fs, ...
                        'NumChannels', n_ch, 'SamplesPerFrame', c.frame_size);
                else
                    adr = audioDeviceReader('Device', dev, ...
                        'SampleRate', c.fs, ...
                        'NumChannels', n_ch, 'SamplesPerFrame', c.frame_size);
                end
                adr();   % warm-up
            catch ME
                warning('QWiSE:AudioIO:Fallback', ...
                    '[Q-WiSE] %s — falling back to mono default.', ME.message);
                adr = audioDeviceReader('SampleRate', c.fs, 'NumChannels', 1, ...
                    'SamplesPerFrame', c.frame_size);
                n_ch = 1;
            end
        end

        function p = default_rec_path_(obj)
            dir_ = obj.cfg.record.dir;
            if ~isfolder(dir_), mkdir(dir_); end
            stamp = datestr(now, 'yyyymmdd_HHMMSS'); %#ok<DATST,TNOW1>
            p = fullfile(dir_, sprintf('%s_%s.wav', ...
                obj.cfg.record.prefix, stamp));
        end
    end
end
