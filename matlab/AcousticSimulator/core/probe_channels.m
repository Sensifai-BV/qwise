function n = probe_channels(dev, fs, frame, desired)
%PROBE_CHANNELS  Try opening audioDeviceReader with decreasing channel
%count until one succeeds, returning the first viable N.

    for n = [desired, 2, 1]
        try
            if isempty(dev)
                t = audioDeviceReader('SampleRate', fs, ...
                    'NumChannels', n, 'SamplesPerFrame', frame);
            else
                t = audioDeviceReader('Device', dev, 'SampleRate', fs, ...
                    'NumChannels', n, 'SamplesPerFrame', frame);
            end
            t();  release(t);  return;
        catch
        end
    end
    n = 1;
end
