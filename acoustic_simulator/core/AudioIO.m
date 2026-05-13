classdef AudioIO < handle
%AUDIOIO  Real-time microphone reader + speaker writer + looping noise WAVs.
%
%   obj = AudioIO(cfg)
%     Opens the default input microphone (auto-detected via keyword list),
%     opens an audioDeviceWriter for low-latency playback, and preloads
%     two looped noise buffers (drone fan + environment ambient).
%
%   raw        = obj.read()                 % one frame from the mic
%                obj.play(x)                % push mono frame to speakers
%   chunk      = obj.next_drone_chunk(N)    % next N samples of drone loop
%   chunk      = obj.next_env_chunk(N)      % next N samples of env loop
%
%   Recording supports two modes, locked in at rec_start time:
%     'mono'  → single mono WAV  (rec_write accepts [N×1])
%     'multi' → one WAV per mic  (rec_write accepts [N×n_mics]); on
%               rec_stop, files are written into a timestamped folder.
%
%                path  = obj.rec_start('mono')   % returns path to .wav
%                base  = obj.rec_start('multi')  % returns the folder
%                obj.rec_write(x)
%   path_or_dir = obj.rec_stop()                % closes; returns location
%   tf          = obj.is_recording()
%   mode        = obj.rec_mode()                % '' | 'mono' | 'multi'
%
%                obj.release()              % free the audio devices

    properties
        cfg
        reader
        writer
        n_hw_ch = 1
        drone_wav
        env_wav
        speech_wav = []         % optional clean-speech loop (empty = use mic)
        speech_wav_path = ''
    end

    properties (Access = private)
        drone_ptr = 1
        env_ptr   = 1
        speech_ptr_ = 1
        rec_active_ = false
        rec_mode_   = ''            % '' | 'mono' | 'multi' | 'session'
        rec_path_   = ''            % file (mono) or folder (multi/session)
        rec_buf_    = zeros(0, 1)   % mono   accumulator
        rec_bufM_   = zeros(0, 0)   % multi  accumulator [samples × n_mics]
        rec_cap_    = 0
        rec_len_    = 0
        rec_n_ch_   = 1
        sess_tracks_ = struct()     % name → struct(buf,len,cap,n_ch)
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

        % ---------------- Clean-speech WAV source ------------------
        % An optional looped clean-speech buffer. When set, the GUI can
        % feed mixer/VAD/MWF from this loop instead of the live mic — the
        % rest of the pipeline does not need to know where the speech
        % came from.

        function load_speech_wav(obj, path)
        %LOAD_SPEECH_WAV  Load a clean-speech WAV, resample to cfg.fs,
        %   mono, and cache it as a looped buffer.
            [y, fs0] = audioread(path);
            if size(y, 2) > 1
                y = mean(y, 2);
            end
            if fs0 ~= obj.cfg.fs
                y = resample(y, obj.cfg.fs, fs0);
            end
            % Peak-normalise to a sane level so the WAV behaves like the
            % live mic (the rest of the pipeline expects |x| <= ~1).
            pk = max(abs(y));
            if pk > 0
                y = y / pk * 0.9;
            end
            obj.speech_wav      = y(:);
            obj.speech_wav_path = path;
            obj.speech_ptr_     = 1;
            fprintf('[Q-WiSE] Loaded clean-speech WAV: %s (%.2f s, %d Hz)\n', ...
                    path, numel(y) / obj.cfg.fs, obj.cfg.fs);
        end

        function clear_speech_wav(obj)
        %CLEAR_SPEECH_WAV  Drop the cached speech loop (revert to mic).
            obj.speech_wav      = [];
            obj.speech_wav_path = '';
            obj.speech_ptr_     = 1;
        end

        function tf = has_speech_wav(obj)
            tf = ~isempty(obj.speech_wav);
        end

        function c = next_speech_chunk(obj, N)
        %NEXT_SPEECH_CHUNK  Next N samples of the clean-speech loop.
        %   Returns zeros if no WAV is loaded (caller should check
        %   has_speech_wav first; the silent fallback is just a safety
        %   net so the timer cannot crash mid-recording).
            if isempty(obj.speech_wav)
                c = zeros(N, 1);
                return;
            end
            c = loop_chunk(obj.speech_wav, obj.speech_ptr_, N);
            obj.speech_ptr_ = mod(obj.speech_ptr_ + N - 1, ...
                                  length(obj.speech_wav)) + 1;
        end

        % ---------------- Recording --------------------------------
        % Recording uses an in-memory buffer flushed to disk on stop.
        % This keeps the WAV header consistent (file is always either
        % absent or complete) and avoids partial-multi-channel writes.

        function dst = rec_start(obj, mode, path)
        %REC_START  Open a recording session.
        %   mode = 'mono'  → records a single mono WAV;
        %   mode = 'multi' → records one mono WAV per mic into a
        %                    timestamped folder under cfg.record.dir.
        %   Returns the resolved path (file for mono, folder for multi).
            if nargin < 2 || isempty(mode), mode = 'mono'; end
            mode = lower(mode);
            if ~ismember(mode, {'mono','multi'})
                error('AudioIO:rec_start:UnknownMode', ...
                      'mode must be ''mono'' or ''multi'' (got %s).', mode);
            end
            if obj.rec_active_
                warning('QWiSE:AudioIO:AlreadyRecording', ...
                        '[Q-WiSE] rec_start called while a recording is active.');
                dst = obj.rec_path_;
                return;
            end

            obj.rec_mode_ = mode;
            obj.rec_len_  = 0;
            obj.rec_cap_  = 30 * obj.cfg.fs;     % ~30 s initial capacity

            switch mode
                case 'mono'
                    if nargin < 3 || isempty(path)
                        path = obj.default_rec_path_('wav');
                    end
                    obj.ensure_dir_(fileparts(path));
                    obj.rec_n_ch_ = 1;
                    obj.rec_buf_  = zeros(obj.rec_cap_, 1);
                    obj.rec_bufM_ = zeros(0, 0);
                case 'multi'
                    if nargin < 3 || isempty(path)
                        path = obj.default_rec_path_('dir');
                    end
                    obj.ensure_dir_(path);
                    obj.rec_n_ch_ = obj.cfg.n_mics;
                    obj.rec_buf_  = zeros(0, 1);
                    obj.rec_bufM_ = zeros(obj.rec_cap_, obj.rec_n_ch_);
            end

            obj.rec_path_   = path;
            obj.rec_active_ = true;
            dst             = path;
            fprintf('[Q-WiSE] Recording (%s) started -> %s\n', mode, path);
        end

        function rec_write(obj, x)
        %REC_WRITE  Append a block to the active recording.
        %   In 'mono' mode x is [N×1].
        %   In 'multi' mode x is [N×n_mics] (one column per microphone).
            if ~obj.rec_active_, return; end
            switch obj.rec_mode_
                case 'mono'
                    x  = max(min(x(:), 1), -1);
                    nx = numel(x);
                    obj.grow_mono_(obj.rec_len_ + nx);
                    obj.rec_buf_(obj.rec_len_+1 : obj.rec_len_+nx) = x;
                    obj.rec_len_ = obj.rec_len_ + nx;
                case 'multi'
                    if size(x, 2) ~= obj.rec_n_ch_
                        warning('QWiSE:AudioIO:ChannelMismatch', ...
                            '[Q-WiSE] rec_write got %d channels, expected %d.', ...
                            size(x, 2), obj.rec_n_ch_);
                        return;
                    end
                    xm = max(min(x, 1), -1);
                    nx = size(xm, 1);
                    obj.grow_multi_(obj.rec_len_ + nx);
                    obj.rec_bufM_(obj.rec_len_+1 : obj.rec_len_+nx, :) = xm;
                    obj.rec_len_ = obj.rec_len_ + nx;
            end
        end

        function dst = rec_stop(obj)
        %REC_STOP  Flush buffers to disk and close the session.
        %   Returns the file path (mono) or the folder path (multi/session).
            dst = obj.rec_path_;
            if ~obj.rec_active_
                dst = '';
                return;
            end
            mode = obj.rec_mode_;
            if strcmpi(mode, 'session')
                % Defer to the session-specific finalizer.
                dst = obj.rec_stop_session();
                return;
            end
            obj.rec_active_ = false;

            try
                switch mode
                    case 'mono'
                        y = obj.rec_buf_(1:obj.rec_len_);
                        if ~isempty(y)
                            audiowrite(obj.rec_path_, y, obj.cfg.fs, ...
                                       'BitsPerSample', 16);
                            fprintf('[Q-WiSE] Recording stopped -> %s (%.2f s)\n', ...
                                obj.rec_path_, numel(y) / obj.cfg.fs);
                        else
                            fprintf(['[Q-WiSE] Recording stopped with no ' ...
                                     'samples — nothing written.\n']);
                            dst = '';
                        end
                    case 'multi'
                        Y = obj.rec_bufM_(1:obj.rec_len_, :);
                        if isempty(Y)
                            fprintf(['[Q-WiSE] Multi recording stopped with no ' ...
                                     'samples — nothing written.\n']);
                            dst = '';
                        else
                            for m = 1:obj.rec_n_ch_
                                fn = fullfile(obj.rec_path_, ...
                                              sprintf('mic%02d.wav', m));
                                audiowrite(fn, Y(:, m), obj.cfg.fs, ...
                                           'BitsPerSample', 16);
                            end
                            fprintf(['[Q-WiSE] Multi recording stopped -> %s ' ...
                                     '(%d mic files, %.2f s)\n'], ...
                                obj.rec_path_, obj.rec_n_ch_, ...
                                size(Y,1) / obj.cfg.fs);
                        end
                end
            catch ME
                warning('QWiSE:AudioIO:RecFlush', ...
                    '[Q-WiSE] rec flush failed: %s', ME.message);
                dst = '';
            end

            % Reset state.
            obj.rec_buf_  = zeros(0, 1);
            obj.rec_bufM_ = zeros(0, 0);
            obj.rec_cap_  = 0;
            obj.rec_len_  = 0;
            obj.rec_n_ch_ = 1;
            obj.rec_path_ = '';
            obj.rec_mode_ = '';
        end

        % ---------------- Session recording -----------------------
        % A "session" is a recording folder that can hold an arbitrary
        % set of named tracks. Tracks are buffered in memory and flushed
        % to disk on rec_stop_session(). This is the mode the GUI uses
        % when the user wants more than one output simultaneously
        % (e.g. raw mics + VAD + MWF cleaned speech).

        function dst = rec_start_session(obj, path)
        %REC_START_SESSION  Open a recording session folder.
            if obj.rec_active_
                warning('QWiSE:AudioIO:AlreadyRecording', ...
                        '[Q-WiSE] rec_start_session called while recording.');
                dst = obj.rec_path_;
                return;
            end
            if nargin < 2 || isempty(path)
                path = obj.default_rec_path_('dir');
            end
            obj.ensure_dir_(path);
            obj.rec_mode_    = 'session';
            obj.rec_path_    = path;
            obj.sess_tracks_ = struct();
            obj.rec_active_  = true;
            dst              = path;
            fprintf('[Q-WiSE] Recording session started -> %s\n', path);
        end

        function rec_session_write(obj, name, x)
        %REC_SESSION_WRITE  Append samples to a named track inside the
        %   active session. `name` becomes the file stem.  `x` is either
        %   a [N×1] mono block (→ <name>.wav) or a [N×K] multi-channel
        %   block (→ <name>01.wav, <name>02.wav, ..., <name>KK.wav).
            if ~obj.rec_active_ || ~strcmpi(obj.rec_mode_, 'session'), return; end
            if isempty(x), return; end
            if ~isvarname(name)
                warning('QWiSE:AudioIO:BadTrackName', ...
                    '[Q-WiSE] track name "%s" is not a valid identifier.', name);
                return;
            end
            n_ch = size(x, 2);
            nx   = size(x, 1);

            if ~isfield(obj.sess_tracks_, name)
                cap0 = 30 * obj.cfg.fs;
                obj.sess_tracks_.(name) = struct( ...
                    'buf',  zeros(cap0, n_ch), ...
                    'len',  0, ...
                    'cap',  cap0, ...
                    'n_ch', n_ch);
            end

            t = obj.sess_tracks_.(name);
            if n_ch ~= t.n_ch
                warning('QWiSE:AudioIO:TrackChannelMismatch', ...
                    '[Q-WiSE] track %s got %d channels, expected %d.', ...
                    name, n_ch, t.n_ch);
                return;
            end

            need = t.len + nx;
            if need > t.cap
                new_cap = max(need, 2 * t.cap);
                tmp = zeros(new_cap, t.n_ch);
                tmp(1:t.len, :) = t.buf(1:t.len, :);
                t.buf = tmp;
                t.cap = new_cap;
            end
            t.buf(t.len+1:t.len+nx, :) = max(min(x, 1), -1);
            t.len = t.len + nx;
            obj.sess_tracks_.(name) = t;
        end

        function dst = rec_stop_session(obj)
        %REC_STOP_SESSION  Flush all session tracks to <session>/<name>.wav
        %   (or <name>NN.wav for multi-channel tracks) and close.
            dst = obj.rec_path_;
            if ~obj.rec_active_ || ~strcmpi(obj.rec_mode_, 'session')
                dst = '';
                return;
            end
            obj.rec_active_ = false;
            names = fieldnames(obj.sess_tracks_);
            files = {};
            try
                for k = 1:numel(names)
                    nm = names{k};
                    t  = obj.sess_tracks_.(nm);
                    Y  = t.buf(1:t.len, :);
                    if isempty(Y), continue; end
                    if t.n_ch == 1
                        fn = fullfile(dst, [nm '.wav']);
                        audiowrite(fn, Y, obj.cfg.fs, 'BitsPerSample', 16);
                        files{end+1} = fn; %#ok<AGROW>
                    else
                        for m = 1:t.n_ch
                            fn = fullfile(dst, sprintf('%s%02d.wav', nm, m));
                            audiowrite(fn, Y(:, m), obj.cfg.fs, 'BitsPerSample', 16);
                            files{end+1} = fn; %#ok<AGROW>
                        end
                    end
                end
                if isempty(files)
                    fprintf(['[Q-WiSE] Session stopped with no samples — ' ...
                             'nothing written.\n']);
                    dst = '';
                else
                    fprintf('[Q-WiSE] Session stopped -> %s (%d files)\n', ...
                            dst, numel(files));
                end
            catch ME
                warning('QWiSE:AudioIO:SessionFlush', ...
                    '[Q-WiSE] session flush failed: %s', ME.message);
                dst = '';
            end
            obj.sess_tracks_ = struct();
            obj.rec_path_    = '';
            obj.rec_mode_    = '';
        end

        function tf = is_recording(obj)
            tf = obj.rec_active_;
        end

        function m = rec_mode(obj)
            m = obj.rec_mode_;
        end

        function release(obj)
            obj.rec_stop();
            try release(obj.reader); catch, end
            try release(obj.writer); catch, end
            fprintf('[Q-WiSE] Audio devices released.\n');
        end
    end

    methods (Access = private)
        function grow_mono_(obj, need)
            if need > obj.rec_cap_
                new_cap = max(need, 2 * obj.rec_cap_);
                tmp = zeros(new_cap, 1);
                tmp(1:obj.rec_len_) = obj.rec_buf_(1:obj.rec_len_);
                obj.rec_buf_ = tmp;
                obj.rec_cap_ = new_cap;
            end
        end

        function grow_multi_(obj, need)
            if need > obj.rec_cap_
                new_cap = max(need, 2 * obj.rec_cap_);
                tmp = zeros(new_cap, obj.rec_n_ch_);
                tmp(1:obj.rec_len_, :) = obj.rec_bufM_(1:obj.rec_len_, :);
                obj.rec_bufM_ = tmp;
                obj.rec_cap_  = new_cap;
            end
        end

        function ensure_dir_(~, d)
            if ~isempty(d) && exist(d, 'dir') ~= 7
                mkdir(d);
            end
        end

        function [adr, n_ch] = create_reader_(obj)
            c   = obj.cfg;
            dev = find_input_mic();
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

        function p = default_rec_path_(obj, kind)
        %DEFAULT_REC_PATH_  Build a timestamped file (kind='wav') or
        %   folder (kind='dir') path under cfg.record.dir.
            dir_ = obj.cfg.record.dir;
            if ~isfolder(dir_), mkdir(dir_); end
            stamp = datestr(now, 'yyyymmdd_HHMMSS'); %#ok<DATST,TNOW1>
            prefix = obj.cfg.record.prefix;
            if strcmpi(kind, 'dir')
                sub = 'multi';
                if isfield(obj.cfg.record, 'multi_subdir') ...
                        && ~isempty(obj.cfg.record.multi_subdir)
                    sub = obj.cfg.record.multi_subdir;
                end
                p = fullfile(dir_, sprintf('%s_%s_%s', prefix, sub, stamp));
            else
                p = fullfile(dir_, sprintf('%s_%s.wav', prefix, stamp));
            end
        end
    end
end
