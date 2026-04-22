classdef AudioIO < handle
%AUDIOIO  Real-time microphone reader + speaker writer + looping noise WAVs.
%
%   obj = AudioIO(cfg)
%     Opens the preferred MacBook microphone (MX2H3 via keyword match),
%     opens an audioDeviceWriter for low-latency playback, and preloads
%     two looped noise buffers (drone fan + environment ambient).
%
%   raw        = obj.read()                 % one frame from the mic
%                obj.play(x)                % push mono frame to speakers
%   chunk      = obj.next_drone_chunk(N)    % next N samples of drone loop
%   chunk      = obj.next_env_chunk(N)      % next N samples of env loop
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

        function release(obj)
            try release(obj.reader); catch, end
            try release(obj.writer); catch, end
            fprintf('[Q-WiSE] Audio devices released.\n');
        end
    end

    methods (Access = private)
        function [adr, n_ch] = create_reader_(obj)
            c   = obj.cfg;
            dev = find_mac_mic(c.mic_model);
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
    end
end
