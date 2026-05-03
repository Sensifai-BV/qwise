function [micSignals, fs, meta] = mixer_1(sourceFiles, numMics, varargin)
% DIEDGE_ACOUSTIC_SIMULATOR
% -------------------------------------------------------------------------
% Practical acoustic simulator for the DIEDGE project.
%
% It simulates microphone-array recordings on a drone in a 3D acoustic scene
% with:
%   1) a human mouth / human sound source,
%   2) a drone/body/rotor sound source near the drone,
%   3) an environmental noise source.
%
% The simulator returns the multichannel microphone outputs measured by
% microphones mounted on the drone.
%
% -------------------------------------------------------------------------
% BASIC USAGE
% -------------------------------------------------------------------------
%   [Y, fs, meta] = diedge_acoustic_simulator( ...
%       {'human.wav','drone.wav','environment.wav'}, 4);
%
%   audiowrite('mic_outputs.wav', Y, fs);
%
% -------------------------------------------------------------------------
% EXAMPLE: DRONE POSITION BY ANGLE AND DISTANCE FROM HUMAN MOUTH
% -------------------------------------------------------------------------
%   [Y, fs, meta] = diedge_acoustic_simulator( ...
%       {'human.wav','drone.wav','environment.wav'}, ...
%       6, ...
%       'DroneDistanceFromMouth', 1.5, ...
%       'DroneAzimuthDeg', 35, ...
%       'DroneElevationDeg', 10, ...
%       'MicGeometry', 'circular', ...
%       'PlotGeometry', true);
%
% -------------------------------------------------------------------------
% EXAMPLE: INDOOR SCENE
% -------------------------------------------------------------------------
%   [Y, fs, meta] = diedge_acoustic_simulator( ...
%       {'human.wav','drone.wav','environment.wav'}, ...
%       4, ...
%       'EnvironmentType', 'indoor', ...
%       'IndoorWallDistance', 2.5, ...
%       'PlotGeometry', true);
%
% -------------------------------------------------------------------------
% REQUIRED INPUTS
% -------------------------------------------------------------------------
%   sourceFiles : 1x3 cell array or string array
%       sourceFiles{1} = human speech / human sound
%       sourceFiles{2} = drone noise / drone-related source
%       sourceFiles{3} = environmental noise / ambience
%
%   numMics : integer >= 3
%       Number of microphones mounted on the drone.
%
% -------------------------------------------------------------------------
% IMPORTANT NAME-VALUE OPTIONS
% -------------------------------------------------------------------------
%   'EnvironmentType' : 'outdoor' or 'indoor'
%       Default: 'outdoor'
%       Outdoor means open air with asphalt-like ground reflection.
%       Indoor means a simple rectangular room around the human mouth with
%       four walls 2.5 m away by default.
%
%   'HumanMouthPosition' or 'HumanPosition' : [x y z] in metres
%       Default: [0 0 1.75]
%       Human mouth/source position. The default height is 1.75 m.
%
%   'DroneDistanceFromMouth' : scalar in metres
%       Default: 1.3
%       3D distance from human mouth to drone centre.
%
%   'DroneAzimuthDeg' : scalar in degrees
%       Default: 20
%       Horizontal angle of drone relative to human mouth.
%       0 deg = +x direction, 90 deg = +y direction.
%
%   'DroneElevationDeg' : scalar in degrees
%       Default: 2
%       Vertical angle of drone relative to human mouth.
%       Positive means drone is above the human mouth.
%
%   'DronePosition' : [x y z] in metres or []
%       Default: []
%       If provided, this overrides DroneDistanceFromMouth,
%       DroneAzimuthDeg, and DroneElevationDeg.
%
%   'DroneSourceOffset' : [x y z] in metres
%       Default: [0.25 0 0]
%       Drone-noise source position relative to drone centre.
%
%   'EnvironmentDistanceFromMouth' : scalar in metres
%       Default: 8
%       Environmental noise source distance from human mouth.
%
%   'EnvironmentAzimuthDeg' : scalar in degrees
%       Default: 135
%
%   'EnvironmentElevationDeg' : scalar in degrees
%       Default: 0
%
%   'EnvironmentPosition' : [x y z] in metres or []
%       Default: []
%       If provided, this overrides the environmental distance/angle options.
%
%   'MicSpacing' : microphone spacing in metres
%       Default: 0.10
%
%   'MicGeometry' : 'linear' or 'circular'
%       Default: 'linear'
%
%   'IndoorWallDistance' : wall distance from human mouth in metres
%       Default: 2.5
%       Used only for EnvironmentType = 'indoor'.
%
%   'IndoorWallReflectionCoeff' : scalar in [0,1]
%       Default: 0.55
%
%   'GroundReflectionCoeff' : scalar in [0,1] or []
%       Default: []
%       If empty, the simulator uses 0.45 for outdoor asphalt-like ground
%       and 0.60 for indoor floor.
%
% -------------------------------------------------------------------------
% OUTPUTS
% -------------------------------------------------------------------------
%   micSignals : N x numMics matrix
%       Simulated microphone signals. Each column is one microphone.
%
%   fs : sampling rate in Hz
%
%   meta : struct
%       Geometry, parameters, distances, delays, and gains.
%
% -------------------------------------------------------------------------
% LIMITATIONS
% -------------------------------------------------------------------------
% This is an engineering simulator, not a full acoustic ray tracer. It models
% direct-path propagation, 1/r geometric attenuation, fractional delay,
% simple distance-dependent high-frequency attenuation, ground reflection,
% optional first-order indoor wall reflections, and weak diffuse microphone
% noise. It does not model wind turbulence, drone aerodynamics, full room
% impulse responses, scattering, diffraction, HRTF, or high-order multipath.
% -------------------------------------------------------------------------

% ----------------------------- Parse inputs ------------------------------
if nargin < 2
    error('You must provide sourceFiles and numMics.');
end

if ~(iscell(sourceFiles) || isstring(sourceFiles)) || numel(sourceFiles) ~= 3
    error('sourceFiles must be a 1x3 cell array or string array: {human, drone, environment}.');
end

if ~isscalar(numMics) || numMics < 3 || floor(numMics) ~= numMics
    error('numMics must be an integer >= 3.');
end

p = inputParser;
p.FunctionName = 'diedge_acoustic_simulator';

addParameter(p, 'Fs', [], @(x) isempty(x) || (isscalar(x) && x > 0));
addParameter(p, 'Duration', [], @(x) isempty(x) || (isscalar(x) && x > 0));

addParameter(p, 'EnvironmentType', 'outdoor', @(s) any(strcmpi(s, {'outdoor','indoor'})));

% Human mouth/source position. HumanPosition is kept for backward compatibility.
addParameter(p, 'HumanMouthPosition', [0 0 1.75], @(x) isnumeric(x) && numel(x)==3);
addParameter(p, 'HumanPosition', [], @(x) isempty(x) || (isnumeric(x) && numel(x)==3));

% Drone centre position can be defined either explicitly or through
% distance/azimuth/elevation relative to the human mouth.
addParameter(p, 'DronePosition', [], @(x) isempty(x) || (isnumeric(x) && numel(x)==3));
addParameter(p, 'DroneDistanceFromMouth', 1.3, @(x) isscalar(x) && x > 0);
addParameter(p, 'DroneAzimuthDeg', 20, @(x) isscalar(x));
addParameter(p, 'DroneElevationDeg', 2, @(x) isscalar(x));

% Drone noise source relative to drone centre.
addParameter(p, 'DroneSourceOffset', [0.25 0 0], @(x) isnumeric(x) && numel(x)==3);

% Environmental noise source. By default, 8 m away from the human mouth.
addParameter(p, 'EnvironmentPosition', [], @(x) isempty(x) || (isnumeric(x) && numel(x)==3));
addParameter(p, 'EnvironmentDistanceFromMouth', 8, @(x) isscalar(x) && x > 0);
addParameter(p, 'EnvironmentAzimuthDeg', 135, @(x) isscalar(x));
addParameter(p, 'EnvironmentElevationDeg', 0, @(x) isscalar(x));

addParameter(p, 'MicSpacing', 0.10, @(x) isscalar(x) && x > 0);
addParameter(p, 'MicGeometry', 'linear', @(s) any(strcmpi(s, {'linear','circular'})));

addParameter(p, 'SpeedOfSound', 343, @(x) isscalar(x) && x > 0);

addParameter(p, 'GroundReflection', true, @(x) islogical(x) || isnumeric(x));
addParameter(p, 'GroundReflectionCoeff', [], @(x) isempty(x) || (isscalar(x) && x >= 0 && x <= 1));

addParameter(p, 'IndoorWallDistance', 2.5, @(x) isscalar(x) && x > 0);
addParameter(p, 'IndoorWallReflectionCoeff', 0.55, @(x) isscalar(x) && x >= 0 && x <= 1);

addParameter(p, 'AirAbsorption', true, @(x) islogical(x) || isnumeric(x));

addParameter(p, 'HumanGain', 1.0, @(x) isscalar(x));
addParameter(p, 'DroneGain', 0.8, @(x) isscalar(x));
addParameter(p, 'EnvironmentGain', 0.5, @(x) isscalar(x));
addParameter(p, 'DiffuseNoiseLevel', 0.005, @(x) isscalar(x) && x >= 0);

addParameter(p, 'NormalizeOutput', true, @(x) islogical(x) || isnumeric(x));
addParameter(p, 'PlotGeometry', false, @(x) islogical(x) || isnumeric(x));

parse(p, varargin{:});
opt = p.Results;

opt.EnvironmentType = lower(opt.EnvironmentType);
opt.MicGeometry = lower(opt.MicGeometry);

% Use HumanPosition if supplied, otherwise HumanMouthPosition.
if ~isempty(opt.HumanPosition)
    humanMouthPos = reshape(opt.HumanPosition, 1, 3);
else
    humanMouthPos = reshape(opt.HumanMouthPosition, 1, 3);
end

% Set default reflection according to environment.
if isempty(opt.GroundReflectionCoeff)
    if strcmpi(opt.EnvironmentType, 'indoor')
        opt.GroundReflectionCoeff = 0.60;   % typical hard indoor floor approximation
    else
        opt.GroundReflectionCoeff = 0.45;   % asphalt-like outdoor ground approximation
    end
end

sourceFiles = cellstr(sourceFiles);

% ----------------------------- Load sources ------------------------------
[xHuman, fsHuman] = audioread(sourceFiles{1});
[xDrone, fsDrone] = audioread(sourceFiles{2});
[xEnv, fsEnv]     = audioread(sourceFiles{3});

% Convert stereo/multichannel inputs to mono.
xHuman = mean(xHuman, 2);
xDrone = mean(xDrone, 2);
xEnv   = mean(xEnv, 2);

% Choose target sampling rate.
if isempty(opt.Fs)
    fs = fsHuman;
else
    fs = opt.Fs;
end

% Resample if necessary.
xHuman = localResampleIfNeeded(xHuman, fsHuman, fs);
xDrone = localResampleIfNeeded(xDrone, fsDrone, fs);
xEnv   = localResampleIfNeeded(xEnv, fsEnv, fs);

% Determine output duration.
% The human voice sets the length; drone and env noise are looped to match.
if isempty(opt.Duration)
    nSamples = numel(xHuman);
else
    nSamples = round(opt.Duration * fs);
end

xHuman = localFitLength(xHuman, nSamples);
xDrone = localFitLength(xDrone, nSamples);
xEnv   = localFitLength(xEnv, nSamples);

% Remove DC and normalize each source safely.
xHuman = localNormalizeSource(xHuman);
xDrone = localNormalizeSource(xDrone);
xEnv   = localNormalizeSource(xEnv);

% Apply user gains.
xHuman = opt.HumanGain * xHuman;
xDrone = opt.DroneGain * xDrone;
xEnv   = opt.EnvironmentGain * xEnv;

% -------------------------- Define 3D geometry ---------------------------
if isempty(opt.DronePosition)
    droneCenter = humanMouthPos + localSphericalOffset( ...
        opt.DroneDistanceFromMouth, opt.DroneAzimuthDeg, opt.DroneElevationDeg);
else
    droneCenter = reshape(opt.DronePosition, 1, 3);
end

droneSourcePos = droneCenter + reshape(opt.DroneSourceOffset, 1, 3);

if isempty(opt.EnvironmentPosition)
    environmentPos = humanMouthPos + localSphericalOffset( ...
        opt.EnvironmentDistanceFromMouth, opt.EnvironmentAzimuthDeg, opt.EnvironmentElevationDeg);
else
    environmentPos = reshape(opt.EnvironmentPosition, 1, 3);
end

micPositions = localBuildMicArray(droneCenter, numMics, opt.MicSpacing, opt.MicGeometry);

sourceSignals = {xHuman, xDrone, xEnv};
sourcePositions = [humanMouthPos; droneSourcePos; environmentPos];
sourceNames = {'human_mouth', 'drone_noise_source', 'environment_noise_source'};

% ----------------------- Propagate to microphones -------------------------
micSignals = zeros(nSamples, numMics);

directDelaysSec = zeros(3, numMics);
directDistances = zeros(3, numMics);
directGains = zeros(3, numMics);

groundDelaysSec = zeros(3, numMics);
groundDistances = zeros(3, numMics);
groundGains = zeros(3, numMics);

indoorWallDelaysSec = [];
indoorWallDistances = [];
indoorWallGains = [];

if strcmpi(opt.EnvironmentType, 'indoor')
    indoorWallDelaysSec = zeros(3, numMics, 4);
    indoorWallDistances = zeros(3, numMics, 4);
    indoorWallGains = zeros(3, numMics, 4);
end

for s = 1:3
    src = sourceSignals{s};
    srcPos = sourcePositions(s, :);

    for m = 1:numMics
        micPos = micPositions(m, :);

        % Direct path.
        [yDirect, d, delaySec, gain] = localPropagateSinglePath( ...
            src, srcPos, micPos, fs, opt.SpeedOfSound, 1.0, opt.AirAbsorption);

        micSignals(:, m) = micSignals(:, m) + yDirect(1:nSamples);

        directDelaysSec(s,m) = delaySec;
        directDistances(s,m) = d;
        directGains(s,m) = gain;

        % Ground/floor reflection using image source below z=0.
        if opt.GroundReflection
            imageSrcPos = [srcPos(1), srcPos(2), -srcPos(3)];

            [yGround, dg, delayG, gainG] = localPropagateSinglePath( ...
                src, imageSrcPos, micPos, fs, opt.SpeedOfSound, ...
                opt.GroundReflectionCoeff, opt.AirAbsorption);

            micSignals(:, m) = micSignals(:, m) + yGround(1:nSamples);

            groundDelaysSec(s,m) = delayG;
            groundDistances(s,m) = dg;
            groundGains(s,m) = gainG;
        end

        % Indoor first-order wall reflections.
        if strcmpi(opt.EnvironmentType, 'indoor')
            wallImageSources = localIndoorWallImageSources(srcPos, humanMouthPos, opt.IndoorWallDistance);

            for w = 1:4
                [yWall, dw, delayW, gainW] = localPropagateSinglePath( ...
                    src, wallImageSources(w,:), micPos, fs, opt.SpeedOfSound, ...
                    opt.IndoorWallReflectionCoeff, opt.AirAbsorption);

                micSignals(:, m) = micSignals(:, m) + yWall(1:nSamples);

                indoorWallDelaysSec(s,m,w) = delayW;
                indoorWallDistances(s,m,w) = dw;
                indoorWallGains(s,m,w) = gainW;
            end
        end
    end
end

% -------------------------- Add diffuse noise -----------------------------
% This simulates weak uncorrelated sensor/open-air residual noise.
if opt.DiffuseNoiseLevel > 0
    micSignals = micSignals + opt.DiffuseNoiseLevel * randn(size(micSignals));
end

% -------------------------- Normalize output ------------------------------
if opt.NormalizeOutput
    peakVal = max(abs(micSignals), [], 'all');
    if peakVal > 0
        micSignals = 0.98 * micSignals / peakVal;
    end
end

% ------------------------------- Metadata --------------------------------
meta = struct();
meta.fs = fs;
meta.numMics = numMics;
meta.environmentType = opt.EnvironmentType;
meta.micPositions = micPositions;
meta.sourceNames = sourceNames;
meta.sourcePositions = sourcePositions;
meta.humanMouthPosition = humanMouthPos;
meta.humanPosition = humanMouthPos; % backward-compatible alias
meta.droneCenterPosition = droneCenter;
meta.droneSourcePosition = droneSourcePos;
meta.environmentPosition = environmentPos;
meta.droneDistanceFromMouthActual = norm(droneCenter - humanMouthPos);
meta.environmentDistanceFromMouthActual = norm(environmentPos - humanMouthPos);
meta.directDistances = directDistances;
meta.directDelaysSec = directDelaysSec;
meta.directGains = directGains;
meta.groundDistances = groundDistances;
meta.groundDelaysSec = groundDelaysSec;
meta.groundGains = groundGains;
meta.indoorWallDistances = indoorWallDistances;
meta.indoorWallDelaysSec = indoorWallDelaysSec;
meta.indoorWallGains = indoorWallGains;
meta.options = opt;

% ------------------------------ Optional plot -----------------------------
if opt.PlotGeometry
    localPlotGeometry(sourcePositions, sourceNames, micPositions, droneCenter, humanMouthPos, opt);
end

end

% =========================================================================
% Local helper functions
% =========================================================================

function offset = localSphericalOffset(distance, azimuthDeg, elevationDeg)
    az = deg2rad(azimuthDeg);
    el = deg2rad(elevationDeg);

    offset = distance * [cos(el)*cos(az), cos(el)*sin(az), sin(el)];
end

function [y, distanceMeters, delaySec, gain] = localPropagateSinglePath( ...
    x, sourcePos, micPos, fs, speedOfSound, reflectionCoeff, useAirAbsorption)

    distanceMeters = norm(micPos - sourcePos);
    delaySec = distanceMeters / speedOfSound;

    % 1/r geometric spreading. The 0.1 m lower bound avoids singular gains
    % for very close sources.
    gain = reflectionCoeff / max(distanceMeters, 0.1);

    y = localFractionalDelay(x, delaySec * fs);
    y = gain * y;

    if useAirAbsorption
        y = localSimpleAirAbsorption(y, fs, distanceMeters);
    end
end

function imageSources = localIndoorWallImageSources(srcPos, roomCenter, wallDistance)
    % Four first-order vertical wall image sources.
    % Room walls are centred around the human mouth:
    %   x = roomCenter(1) +/- wallDistance
    %   y = roomCenter(2) +/- wallDistance

    xMin = roomCenter(1) - wallDistance;
    xMax = roomCenter(1) + wallDistance;
    yMin = roomCenter(2) - wallDistance;
    yMax = roomCenter(2) + wallDistance;

    imageSources = zeros(4,3);

    % Reflection against x = xMin.
    imageSources(1,:) = [2*xMin - srcPos(1), srcPos(2), srcPos(3)];

    % Reflection against x = xMax.
    imageSources(2,:) = [2*xMax - srcPos(1), srcPos(2), srcPos(3)];

    % Reflection against y = yMin.
    imageSources(3,:) = [srcPos(1), 2*yMin - srcPos(2), srcPos(3)];

    % Reflection against y = yMax.
    imageSources(4,:) = [srcPos(1), 2*yMax - srcPos(2), srcPos(3)];
end

function y = localResampleIfNeeded(x, fsIn, fsOut)
    if fsIn == fsOut
        y = x;
        return;
    end

    if exist('resample', 'file') == 2
        y = resample(x, fsOut, fsIn);
    else
        % Fallback if Signal Processing Toolbox is unavailable.
        tIn = (0:numel(x)-1).' / fsIn;
        duration = tIn(end);
        tOut = (0:round(duration*fsOut)).' / fsOut;
        y = interp1(tIn, x, tOut, 'linear', 'extrap');
    end
end

function y = localFitLength(x, nSamples)
    if numel(x) >= nSamples
        y = x(1:nSamples);
    else
        reps = ceil(nSamples / max(numel(x), 1));
        y = repmat(x, reps, 1);
        y = y(1:nSamples);
    end
end

function y = localNormalizeSource(x)
    x = x(:);
    x = x - mean(x);
    peakVal = max(abs(x));
    if peakVal > 0
        y = x / peakVal;
    else
        y = x;
    end
end

function micPositions = localBuildMicArray(centerPos, numMics, spacing, geometry)
    centerPos = reshape(centerPos, 1, 3);

    switch lower(geometry)
        case 'linear'
            % Linear array along x-axis, centred on the drone.
            idx = (0:numMics-1).' - (numMics-1)/2;
            offsets = [idx * spacing, zeros(numMics,1), zeros(numMics,1)];
            micPositions = centerPos + offsets;

        case 'circular'
            % Circular horizontal array. Adjacent chord distance is close to spacing.
            theta = (0:numMics-1).' * 2*pi/numMics;
            radius = spacing / (2*sin(pi/numMics));
            offsets = [radius*cos(theta), radius*sin(theta), zeros(numMics,1)];
            micPositions = centerPos + offsets;

        otherwise
            error('Unsupported microphone geometry.');
    end
end

function y = localFractionalDelay(x, delaySamples)
    % Applies a nonnegative fractional delay using interpolation:
    % y[n] = x[n - delaySamples].
    x = x(:);
    N = numel(x);

    n = (1:N).';
    queryIndex = n - delaySamples;

    y = interp1(n, x, queryIndex, 'linear', 0);
    y = y(:);
end

function y = localSimpleAirAbsorption(x, fs, distanceMeters)
    % Lightweight distance-dependent high-frequency attenuation.
    % This is not a standards-grade ISO 9613-1 implementation. It is a
    % stable engineering approximation for simulation experiments.

    if distanceMeters <= 1
        y = x;
        return;
    end

    cutoffHz = max(2500, min(18000, 18000 - 350 * distanceMeters));
    cutoffHz = min(cutoffHz, 0.45 * fs);

    if cutoffHz >= 0.45 * fs
        y = x;
        return;
    end

    rc = 1 / (2*pi*cutoffHz);
    dt = 1 / fs;
    alpha = dt / (rc + dt);

    y = zeros(size(x));
    y(1) = alpha * x(1);
    for k = 2:numel(x)
        y(k) = y(k-1) + alpha * (x(k) - y(k-1));
    end
end

function localPlotGeometry(sourcePositions, sourceNames, micPositions, droneCenter, humanMouthPos, opt)
    figure;
    hold on;
    grid on;
    axis equal;

    plot3(micPositions(:,1), micPositions(:,2), micPositions(:,3), ...
        'ko', 'MarkerFaceColor', 'k', 'DisplayName', 'Microphones');

    plot3(droneCenter(1), droneCenter(2), droneCenter(3), ...
        'bs', 'MarkerFaceColor', 'b', 'DisplayName', 'Drone centre');

    markers = {'ro','mo','go'};
    for s = 1:size(sourcePositions,1)
        plot3(sourcePositions(s,1), sourcePositions(s,2), sourcePositions(s,3), ...
            markers{s}, 'MarkerSize', 8, 'LineWidth', 1.5, ...
            'DisplayName', sourceNames{s});
        text(sourcePositions(s,1), sourcePositions(s,2), sourcePositions(s,3), ...
            ['  ' sourceNames{s}], 'Interpreter', 'none');
    end

    if strcmpi(opt.EnvironmentType, 'indoor')
        xMin = humanMouthPos(1) - opt.IndoorWallDistance;
        xMax = humanMouthPos(1) + opt.IndoorWallDistance;
        yMin = humanMouthPos(2) - opt.IndoorWallDistance;
        yMax = humanMouthPos(2) + opt.IndoorWallDistance;
        zMin = 0;
        zMax = max(3.0, max([sourcePositions(:,3); micPositions(:,3)]) + 0.5);

        % Draw a simple room footprint and vertical edges.
        plot3([xMin xMax xMax xMin xMin], [yMin yMin yMax yMax yMin], ...
            [zMin zMin zMin zMin zMin], 'k--', 'DisplayName', 'Indoor walls');

        plot3([xMin xMax xMax xMin xMin], [yMin yMin yMax yMax yMin], ...
            [zMax zMax zMax zMax zMax], 'k--', 'HandleVisibility', 'off');

        corners = [xMin yMin; xMax yMin; xMax yMax; xMin yMax];
        for c = 1:4
            plot3([corners(c,1) corners(c,1)], [corners(c,2) corners(c,2)], ...
                [zMin zMax], 'k--', 'HandleVisibility', 'off');
        end
    end

    xlabel('x [m]');
    ylabel('y [m]');
    zlabel('z [m]');
    title(['DIEDGE acoustic simulation geometry - ' opt.EnvironmentType]);
    legend('Location', 'best');
    view(35, 25);
end
