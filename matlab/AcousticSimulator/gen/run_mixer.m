distances = [0.8, 1.0, 1.2];

for d = 1:numel(distances)
    dist = distances(d);
    [y, fs, meta] = mixer_1({'woman_4.0.wav', 'drone.wav', 'env.wav'}, 3, 'DroneDistanceFromMouth', dist, 'DroneGain', 0.3);

    for m = 1:size(y, 2)
        filename = sprintf('/Users/javad/Projects/qwise/matlab/AcousticSimulator/gen/woman/sp02_woman_%.1f_mic%d.wav', dist, m);
        audiowrite(filename, y(:, m), fs);
        fprintf('Wrote %s  (%.2f s, peak %.4f)\n', filename, size(y,1)/fs, max(abs(y(:,m))));
    end
end