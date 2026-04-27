function dev = find_input_mic(preferred)
%FIND_INPUT_MIC  Locate a built-in / preferred input microphone.
%
%   dev = find_input_mic()           keyword search over common device names.
%   dev = find_input_mic('keyword')  tries the user keyword first.
%
%   Returns an empty string when no matching device is found; callers
%   should then fall back to the system default input.

    if nargin < 1, preferred = ''; end
    dev = '';
    try
        devs = audioDeviceReader.getAudioDevices();
        if ~isempty(preferred)
            m = devs(contains(devs, preferred, 'IgnoreCase', true));
            if ~isempty(m), dev = m{1}; return; end
        end
        % These keywords match the device names the OS reports on common
        % platforms — the strings stay verbatim because they are matched
        % against real audio-device descriptors, not user-facing text.
        kw = {'MacBook Pro Microphone', 'Built-in Microphone', ...
              'MacBook Air Microphone', 'Apple Silicon Microphone', ...
              'MacBook', 'Built-in', 'Built In Microphone'};
        for k = 1:numel(kw)
            m = devs(contains(devs, kw{k}, 'IgnoreCase', true));
            if ~isempty(m), dev = m{1}; return; end
        end
    catch
    end
end
