function dev = find_mac_mic(preferred)
%FIND_MAC_MIC  Locate the built-in microphone on an Apple machine.
%
%   dev = find_mac_mic()                uses a default keyword list.
%   dev = find_mac_mic('MacBook Pro')   searches for a user keyword first.
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
        kw = {'MacBook Pro Microphone', 'Built-in Microphone', ...
              'MacBook Air Microphone', 'Apple Silicon Microphone', ...
              'MX2H3'};
        for k = 1:numel(kw)
            m = devs(contains(devs, kw{k}, 'IgnoreCase', true));
            if ~isempty(m), dev = m{1}; return; end
        end
    catch
    end
end
