function [micSignals, fs, meta] = diedge_acoustic_simulator(sourceFiles, numMics, varargin)
% DIEDGE_ACOUSTIC_SIMULATOR
% -------------------------------------------------------------------------
% More realistic acoustic simulator for the DIEDGE project.
%
% REQUIRED INPUT/OUTPUT INTERFACE IS UNCHANGED:
%
%   [micSignals, fs, meta] = diedge_acoustic_simulator(sourceFiles, numMics, ...)
%
% Required inputs:
%   sourceFiles : 1x3 cell/string array
%       {human speech file, drone noise file, environmental noise file}
%
%   numMics : integer >= 3
%       number of microphones on the drone.
%
% Outputs:
%   micSignals : N x numMics simulated microphone outputs
%   fs         : sampling frequency
%   meta       : simulation metadata
%
% -------------------------------------------------------------------------
% WHAT IS IMPROVED IN THIS VERSION
% -------------------------------------------------------------------------
% Compared with the earlier version, this file keeps the same external
% interface but internally adds:
%
%   1) Higher-order image-source reflections for indoor rooms.
%   2) Ceiling reflection and floor reflection.
%   3) Frequency-dependent wall/floor/ground absorption approximations.
%   4) Outdoor asphalt-like ground reflection with distance-dependent loss.
%   5) Multi-microphone gain mismatch and timing mismatch.
%   6) Microphone self-noise.
%   7) Wind-like low-frequency turbulent noise.
%   8) Drone vibration/body-coupled low-frequency components.
%   9) Direction-dependent drone noise radiation approximation.
%  10) Optional source/microphone geometry randomization.
%
% This is still not a replacement for real recordings or full acoustic
% wave/ray simulation, but it is much stronger for algorithm development,
% ablation studies, and pretraining data generation.
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
% EXAMPLE: OUTDOOR DEFAULT, DRONE POSITION BY ANGLE AND DISTANCE
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
% EXAMPLE: INDOOR WITH HIGHER-ORDER REFLECTIONS
% -------------------------------------------------------------------------
%   [Y, fs, meta] = diedge_acoustic_simulator( ...
%       {'human.wav','drone.wav','environment.wav'}, ...
%       4, ...
%       'EnvironmentType', 'indoor', ...
%       'IndoorWallDistance', 2.5, ...
%       'ReflectionOrder', 3, ...
%       'PlotGeometry', true);
%
% -------------------------------------------------------------------------
% IMPORTANT NAME-VALUE OPTIONS
% -------------------------------------------------------------------------
% Geometry:
%   'EnvironmentType'              : 'outdoor' or 'indoor', default 'outdoor'
%   'HumanMouthPosition'           : [x y z], default [0 0 1.75]
%   'HumanPosition'                : backward-compatible alias, default []
%   'DronePosition'                : [x y z] explicit override, default []
%   'DroneDistanceFromMouth'       : metres, default 1.3
%   'DroneAzimuthDeg'              : degrees, default 20
%   'DroneElevationDeg'            : degrees, default 2
%   'DroneSourceOffset'            : relative to drone centre, default [0.25 0 0]
%   'EnvironmentPosition'          : [x y z] explicit override, default []
%   'EnvironmentDistanceFromMouth' : metres, default 8
%   'EnvironmentAzimuthDeg'        : degrees, default 135
%   'EnvironmentElevationDeg'      : degrees, default 0
%   'MicSpacing'                   : metres, default 0.10
%   'MicGeometry'                  : 'linear' or 'circular', default 'linear'
%
% Acoustic realism:
%   'ReflectionOrder'              : indoor reflection order, default 3
%   'IndoorWallDistance'           : walls at +/- this distance, default 2.5
%   'RoomHeight'                   : indoor room height, default 3.0
%   'GroundReflection'             : true/false, default true
%   'AirAbsorption'                : true/false, default true
%   'MicImperfections'             : true/false, default true
%   'WindNoise'                    : true/false, default true
%   'DroneVibrationNoise'          : true/false, default true
%   'RandomizeGeometry'            : true/false, default false
%   'RandomSeed'                   : integer or [], default []
%
% Gains/noise:
%   'HumanGain'                    : default 1.0
%   'DroneGain'                    : default 0.8
%   'EnvironmentGain'              : default 0.5
%   'DiffuseNoiseLevel'            : default 0.003
%   'MicSelfNoiseLevel'            : default 0.0015
%   'WindNoiseLevel'               : default 0.015
%   'DroneVibrationLevel'          : default 0.010
%
% -------------------------------------------------------------------------
% HONEST LIMITATION
% -------------------------------------------------------------------------
% This simulator is much more useful than the earlier version, but it is
% still a model. For a top-tier speech-enhancement paper, use this simulator
% for controlled generation/ablation and validate with real drone-mounted
% microphone recordings.
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

addParameter(p, 'HumanMouthPosition', [0 0 1.75], @(x) isnumeric(x) && numel(x)==3);
addParameter(p, 'HumanPosition', [], @(x) isempty(x) || (isnumeric(x) && numel(x)==3));

addParameter(p, 'DronePosition', [], @(x) isempty(x) || (isnumeric(x) && numel(x)==3));
addParameter(p, 'DroneDistanceFromMouth', 1.3, @(x) isscalar(x) && x > 0);
addParameter(p, 'DroneAzimuthDeg', 20, @(x) isscalar(x));
addParameter(p, 'DroneElevationDeg', 2, @(x) isscalar(x));
addParameter(p, 'DroneSourceOffset', [0.25 0 0], @(x) isnumeric(x) && numel(x)==3);

addParameter(p, 'EnvironmentPosition', [], @(x) isempty(x) || (isnumeric(x) && numel(x)==3));
addParameter(p, 'EnvironmentDistanceFromMouth', 8, @(x) isscalar(x) && x > 0);
addParameter(p, 'EnvironmentAzimuthDeg', 135, @(x) isscalar(x));
addParameter(p, 'EnvironmentElevationDeg', 0, @(x) isscalar(x));

addParameter(p, 'MicSpacing', 0.10, @(x) isscalar(x) && x > 0);
addParameter(p, 'MicGeometry', 'linear', @(s) any(strcmpi(s, {'linear','circular'})));

addParameter(p, 'SpeedOfSound', 343, @(x) isscalar(x) && x > 0);

addParameter(p, 'GroundReflection', true, @(x) islogical(x) || isnumeric(x));
addParameter(p, 'GroundReflectionCoeff', [], @(x) isempty(x) || (isscalar(x) && x >= 0 && x <= 1));

addParameter(p, 'ReflectionOrder', 3, @(x) isscalar(x) && x >= 0 && floor(x)==x);
addParameter(p, 'IndoorWallDistance', 2.5, @(x) isscalar(x) && x > 0);
addParameter(p, 'RoomHeight', 3.0, @(x) isscalar(x) && x > 1.5);
addParameter(p, 'IndoorWallReflectionCoeff', 0.62, @(x) isscalar(x) && x >= 0 && x <= 1);
addParameter(p, 'IndoorFloorReflectionCoeff', 0.68, @(x) isscalar(x) && x >= 0 && x <= 1);
addParameter(p, 'IndoorCeilingReflectionCoeff', 0.50, @(x) isscalar(x) && x >= 0 && x <= 1);

addParameter(p, 'AirAbsorption', true, @(x) islogical(x) || isnumeric(x));

addParameter(p, 'HumanGain', 1.0, @(x) isscalar(x));
addParameter(p, 'DroneGain', 0.8, @(x) isscalar(x));
addParameter(p, 'EnvironmentGain', 0.5, @(x) isscalar(x));

addParameter(p, 'DiffuseNoiseLevel', 0.003, @(x) isscalar(x) && x >= 0);
addParameter(p, 'MicSelfNoiseLevel', 0.0015, @(x) isscalar(x) && x >= 0);
addParameter(p, 'WindNoise', true, @(x) islogical(x) || isnumeric(x));
addParameter(p, 'WindNoiseLevel', 0.015, @(x) isscalar(x) && x >= 0);
addParameter(p, 'DroneVibrationNoise', true, @(x) islogical(x) || isnumeric(x));
addParameter(p, 'DroneVibrationLevel', 0.010, @(x) isscalar(x) && x >= 0);

addParameter(p, 'MicImperfections', true, @(x) islogical(x) || isnumeric(x));
addParameter(p, 'MicGainStdDb', 0.7, @(x) isscalar(x) && x >= 0);
addParameter(p, 'MicTimingJitterStdSamples', 0.08, @(x) isscalar(x) && x >= 0);

addParameter(p, 'RandomizeGeometry', false, @(x) islogical(x) || isnumeric(x));
addParameter(p, 'PositionJitterStd', 0.015, @(x) isscalar(x) && x >= 0);
addParameter(p, 'RandomSeed', [], @(x) isempty(x) || isscalar(x));

addParameter(p, 'NormalizeOutput', true, @(x) islogical(x) || isnumeric(x));
addParameter(p, 'PlotGeometry', false, @(x) islogical(x) || isnumeric(x));

parse(p, varargin{:});
opt = p.Results;

opt.EnvironmentType = lower(opt.EnvironmentType);
opt.MicGeometry = lower(opt.MicGeometry);

if ~isempty(opt.RandomSeed)
    rng(opt.RandomSeed);
end

if ~isempty(opt.HumanPosition)
    humanMouthPos = reshape(opt.HumanPosition, 1, 3);
else
    humanMouthPos = reshape(opt.HumanMouthPosition, 1, 3);
end

if isempty(opt.GroundReflectionCoeff)
    if strcmpi(opt.EnvironmentType, 'indoor')
        opt.GroundReflectionCoeff = opt.IndoorFloorReflectionCoeff;
    else
        opt.GroundReflectionCoeff = 0.48; % asphalt-like outdoor approximation
    end
end

sourceFiles = cellstr(sourceFiles);

% ----------------------------- Load sources ------------------------------
[xHuman, fsHuman] = audioread(sourceFiles{1});
[xDrone, fsDrone] = audioread(sourceFiles{2});
[xEnv, fsEnv]     = audioread(sourceFiles{3});

xHuman = mean(xHuman, 2);
xDrone = mean(xDrone, 2);
xEnv   = mean(xEnv, 2);

if isempty(opt.Fs)
    fs = fsHuman;
else
    fs = opt.Fs;
end

xHuman = localResampleIfNeeded(xHuman, fsHuman, fs);
xDrone = localResampleIfNeeded(xDrone, fsDrone, fs);
xEnv   = localResampleIfNeeded(xEnv, fsEnv, fs);

if isempty(opt.Duration)
    nSamples = min([numel(xHuman), numel(xDrone), numel(xEnv)]);
else
    nSamples = round(opt.Duration * fs);
end

xHuman = localFitLength(xHuman, nSamples);
xDrone = localFitLength(xDrone, nSamples);
xEnv   = localFitLength(xEnv, nSamples);

xHuman = localNormalizeSource(xHuman);
xDrone = localNormalizeSource(xDrone);
xEnv   = localNormalizeSource(xEnv);

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

if opt.RandomizeGeometry
    droneCenter = droneCenter + opt.PositionJitterStd * randn(1,3);
    droneSourcePos = droneSourcePos + opt.PositionJitterStd * randn(1,3);
    environmentPos = environmentPos + 2*opt.PositionJitterStd * randn(1,3);
    micPositions = micPositions + opt.PositionJitterStd * randn(size(micPositions));
end

sourceSignals = {xHuman, xDrone, xEnv};
sourcePositions = [humanMouthPos; droneSourcePos; environmentPos];
sourceNames = {'human_mouth', 'drone_noise_source', 'environment_noise_source'};

% ----------------------- Propagate to microphones -------------------------
micSignals = zeros(nSamples, numMics);

directDelaysSec = zeros(3, numMics);
directDistances = zeros(3, numMics);
directGains = zeros(3, numMics);

pathCounter = zeros(3, numMics);
totalPathEnergy = zeros(3, numMics);

for s = 1:3
    src = sourceSignals{s};
    srcPos = sourcePositions(s, :);

    for m = 1:numMics
        micPos = micPositions(m, :);

        % Direct path. Drone source gets direction-dependent radiation.
        dirCoeff = 1.0;
        if s == 2
            dirCoeff = localDroneDirectivity(srcPos, micPos, droneCenter);
        end

        [yDirect, d, delaySec, gain] = localPropagateSinglePath( ...
            src, srcPos, micPos, fs, opt.SpeedOfSound, dirCoeff, ...
            opt.AirAbsorption, 'air');

        micSignals(:, m) = micSignals(:, m) + yDirect(1:nSamples);
        directDistances(s,m) = d;
        directDelaysSec(s,m) = delaySec;
        directGains(s,m) = gain;
        pathCounter(s,m) = pathCounter(s,m) + 1;
        totalPathEnergy(s,m) = totalPathEnergy(s,m) + mean(yDirect.^2);

        % Outdoor/indoor floor or asphalt ground reflection.
        if opt.GroundReflection
            imageSrcPos = [srcPos(1), srcPos(2), -srcPos(3)];

            [yGround, ~, ~, ~] = localPropagateSinglePath( ...
                src, imageSrcPos, micPos, fs, opt.SpeedOfSound, ...
                opt.GroundReflectionCoeff * dirCoeff, opt.AirAbsorption, ...
                localSurfaceType(opt.EnvironmentType, 'floor'));

            micSignals(:, m) = micSignals(:, m) + yGround(1:nSamples);
            pathCounter(s,m) = pathCounter(s,m) + 1;
            totalPathEnergy(s,m) = totalPathEnergy(s,m) + mean(yGround.^2);
        end

        % Indoor higher-order image-source reflections.
        if strcmpi(opt.EnvironmentType, 'indoor') && opt.ReflectionOrder > 0
            room = localBuildRoom(humanMouthPos, opt.IndoorWallDistance, opt.RoomHeight);
            imagePaths = localGenerateIndoorImageSources(srcPos, room, opt.ReflectionOrder);

            for k = 1:numel(imagePaths)
                coeff = imagePaths(k).coeff;
                imagePos = imagePaths(k).pos;
                surfaceKind = imagePaths(k).surfaceKind;

                if s == 2
                    coeff = coeff * localDroneDirectivity(srcPos, micPos, droneCenter);
                end

                [yImg, ~, ~, ~] = localPropagateSinglePath( ...
                    src, imagePos, micPos, fs, opt.SpeedOfSound, coeff, ...
                    opt.AirAbsorption, surfaceKind);

                micSignals(:, m) = micSignals(:, m) + yImg(1:nSamples);
                pathCounter(s,m) = pathCounter(s,m) + 1;
                totalPathEnergy(s,m) = totalPathEnergy(s,m) + mean(yImg.^2);
            end
        end
    end
end

% -------------------- Drone body/vibration coupled noise ------------------
if opt.DroneVibrationNoise && opt.DroneVibrationLevel > 0
    vib = localDroneVibrationSignal(nSamples, fs, xDrone);
    for m = 1:numMics
        coupling = 0.8 + 0.4*rand();
        micSignals(:,m) = micSignals(:,m) + opt.DroneVibrationLevel * coupling * vib;
    end
end

% -------------------------- Wind-like noise -------------------------------
if opt.WindNoise && opt.WindNoiseLevel > 0
    commonWind = localColoredNoise(nSamples, fs, 'wind');
    for m = 1:numMics
        localWind = localColoredNoise(nSamples, fs, 'wind');
        mix = 0.65 * commonWind + 0.35 * localWind;
        micSignals(:,m) = micSignals(:,m) + opt.WindNoiseLevel * mix;
    end
end

% ---------------------- Diffuse and microphone noise ----------------------
if opt.DiffuseNoiseLevel > 0
    diffuseCommon = localColoredNoise(nSamples, fs, 'pinkish');
    for m = 1:numMics
        diffuseLocal = localColoredNoise(nSamples, fs, 'pinkish');
        micSignals(:,m) = micSignals(:,m) + opt.DiffuseNoiseLevel * ...
            (0.4*diffuseCommon + 0.6*diffuseLocal);
    end
end

if opt.MicSelfNoiseLevel > 0
    micSignals = micSignals + opt.MicSelfNoiseLevel * randn(size(micSignals));
end

% -------------------------- Mic imperfections -----------------------------
micGainDb = zeros(1, numMics);
micTimingJitterSamples = zeros(1, numMics);

if opt.MicImperfections
    micGainDb = opt.MicGainStdDb * randn(1, numMics);
    micGains = 10.^(micGainDb/20);

    micTimingJitterSamples = opt.MicTimingJitterStdSamples * randn(1, numMics);

    for m = 1:numMics
        micSignals(:,m) = micGains(m) * localFractionalDelay( ...
            micSignals(:,m), micTimingJitterSamples(m));
    end
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
meta.humanPosition = humanMouthPos;
meta.droneCenterPosition = droneCenter;
meta.droneSourcePosition = droneSourcePos;
meta.environmentPosition = environmentPos;
meta.droneDistanceFromMouthActual = norm(droneCenter - humanMouthPos);
meta.environmentDistanceFromMouthActual = norm(environmentPos - humanMouthPos);
meta.directDistances = directDistances;
meta.directDelaysSec = directDelaysSec;
meta.directGains = directGains;
meta.pathCounter = pathCounter;
meta.totalPathEnergy = totalPathEnergy;
meta.micGainDb = micGainDb;
meta.micTimingJitterSamples = micTimingJitterSamples;
meta.options = opt;

if strcmpi(opt.EnvironmentType, 'indoor')
    meta.room = localBuildRoom(humanMouthPos, opt.IndoorWallDistance, opt.RoomHeight);
else
    meta.room = [];
end

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
    x, sourcePos, micPos, fs, speedOfSound, pathCoeff, useAirAbsorption, surfaceKind)

    distanceMeters = norm(micPos - sourcePos);
    delaySec = distanceMeters / speedOfSound;

    % Slightly softened 1/r spreading to avoid numerical blow-up for
    % extremely close paths.
    gain = pathCoeff / max(distanceMeters, 0.08);

    y = localFractionalDelay(x, delaySec * fs);
    y = gain * y;

    if useAirAbsorption
        y = localAirAbsorptionFilter(y, fs, distanceMeters);
    end

    y = localSurfaceAbsorptionFilter(y, fs, surfaceKind);
end

function coeff = localDroneDirectivity(srcPos, micPos, droneCenter)
    % Approximate drone-noise radiation pattern:
    % stronger in the rotor/body horizontal plane, weaker along vertical axis.
    v = micPos - srcPos;
    if norm(v) < eps
        coeff = 1;
        return;
    end
    v = v / norm(v);
    verticalComponent = abs(v(3));
    horizontalWeight = sqrt(max(0, 1 - verticalComponent^2));
    coeff = 0.65 + 0.55 * horizontalWeight;

    % Make it slightly stronger close to drone body.
    bodyDistance = norm(micPos - droneCenter);
    coeff = coeff * (1 + 0.15 * exp(-bodyDistance/0.3));
end

function surfaceType = localSurfaceType(environmentType, kind)
    if strcmpi(environmentType, 'outdoor')
        surfaceType = 'asphalt';
    else
        surfaceType = kind;
    end
end

function room = localBuildRoom(center, wallDistance, roomHeight)
    room = struct();
    room.xMin = center(1) - wallDistance;
    room.xMax = center(1) + wallDistance;
    room.yMin = center(2) - wallDistance;
    room.yMax = center(2) + wallDistance;
    room.zMin = 0;
    room.zMax = roomHeight;
end

function paths = localGenerateIndoorImageSources(srcPos, room, order)
    % Generates image sources for a rectangular shoebox room. This is an
    % approximate implementation of the image-source principle.
    %
    % Reflection indices nx,ny,nz in [-order, order]. Each nonzero index
    % contributes wall/floor/ceiling attenuation. Direct path is excluded.

    paths = struct('pos', {}, 'coeff', {}, 'surfaceKind', {});

    Lx = room.xMax - room.xMin;
    Ly = room.yMax - room.yMin;
    Lz = room.zMax - room.zMin;

    sx = srcPos(1) - room.xMin;
    sy = srcPos(2) - room.yMin;
    sz = srcPos(3) - room.zMin;

    count = 0;

    for nx = -order:order
        for ny = -order:order
            for nz = -order:order
                reflOrder = abs(nx) + abs(ny) + abs(nz);

                if reflOrder == 0 || reflOrder > order
                    continue;
                end

                imgXLocal = localImageCoord1D(sx, Lx, nx);
                imgYLocal = localImageCoord1D(sy, Ly, ny);
                imgZLocal = localImageCoord1D(sz, Lz, nz);

                imgPos = [room.xMin + imgXLocal, room.yMin + imgYLocal, room.zMin + imgZLocal];

                % Typical approximate reflection coefficients. These are
                % intentionally conservative, not tuned to one specific room.
                wallCoeff = 0.62;
                floorCoeff = 0.68;
                ceilingCoeff = 0.50;

                coeff = (wallCoeff ^ (abs(nx) + abs(ny))) * ...
                        (sqrt(floorCoeff*ceilingCoeff) ^ abs(nz));

                if abs(nz) > 0
                    surfaceKind = 'ceiling_floor';
                elseif abs(nx) + abs(ny) > 0
                    surfaceKind = 'wall';
                else
                    surfaceKind = 'air';
                end

                count = count + 1;
                paths(count).pos = imgPos; %#ok<AGROW>
                paths(count).coeff = coeff; %#ok<AGROW>
                paths(count).surfaceKind = surfaceKind; %#ok<AGROW>
            end
        end
    end
end

function img = localImageCoord1D(s, L, n)
    % Image coordinate in a 1D interval [0,L].
    % This compact formula alternates mirrored and translated images.
    if mod(abs(n), 2) == 0
        img = n*L + s;
    else
        img = (n+sign(n))*L - s;
        if n < 0
            img = n*L - s;
        end
    end
end

function y = localResampleIfNeeded(x, fsIn, fsOut)
    if fsIn == fsOut
        y = x;
        return;
    end

    if exist('resample', 'file') == 2
        y = resample(x, fsOut, fsIn);
    else
        tIn = (0:numel(x)-1).' / fsIn;
        duration = tIn(end);
        tOut = (0:round(duration*fsOut)).' / fsOut;
        y = interp1(tIn, x, tOut, 'linear', 'extrap');
    end
end

function y = localFitLength(x, nSamples)
    if isempty(x)
        y = zeros(nSamples,1);
        return;
    end

    if numel(x) >= nSamples
        y = x(1:nSamples);
    else
        reps = ceil(nSamples / numel(x));
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
            idx = (0:numMics-1).' - (numMics-1)/2;
            offsets = [idx * spacing, zeros(numMics,1), zeros(numMics,1)];
            micPositions = centerPos + offsets;

        case 'circular'
            theta = (0:numMics-1).' * 2*pi/numMics;
            radius = spacing / (2*sin(pi/numMics));
            offsets = [radius*cos(theta), radius*sin(theta), zeros(numMics,1)];
            micPositions = centerPos + offsets;

        otherwise
            error('Unsupported microphone geometry.');
    end
end

function y = localFractionalDelay(x, delaySamples)
    % Applies y[n] = x[n - delaySamples]. Allows small negative jitter.
    x = x(:);
    N = numel(x);
    n = (1:N).';
    queryIndex = n - delaySamples;
    y = interp1(n, x, queryIndex, 'linear', 0);
    y = y(:);
end

function y = localAirAbsorptionFilter(x, fs, distanceMeters)
    % Pragmatic distance-dependent high-frequency attenuation.
    % Not a formal ISO 9613 implementation, but more realistic than leaving
    % far sources spectrally unchanged.

    if distanceMeters <= 1
        y = x;
        return;
    end

    cutoffHz = max(2200, min(18000, 19000 - 420 * distanceMeters));
    cutoffHz = min(cutoffHz, 0.47 * fs);

    if cutoffHz >= 0.47 * fs
        y = x;
        return;
    end

    y = localOnePoleLowpass(x, fs, cutoffHz);
end

function y = localSurfaceAbsorptionFilter(x, fs, surfaceKind)
    % Frequency-dependent surface absorption approximations.
    % Harder surfaces keep more high frequency; walls and ceilings lose more.
    switch lower(surfaceKind)
        case 'asphalt'
            y = localShelfLikeDamping(x, fs, 9000, 0.82);
        case 'floor'
            y = localShelfLikeDamping(x, fs, 8500, 0.86);
        case 'wall'
            y = localShelfLikeDamping(x, fs, 6500, 0.72);
        case 'ceiling_floor'
            y = localShelfLikeDamping(x, fs, 5200, 0.65);
        otherwise
            y = x;
    end
end

function y = localShelfLikeDamping(x, fs, cutoffHz, highFreqScale)
    % Splits approximate low/high bands and attenuates high band.
    cutoffHz = min(cutoffHz, 0.45*fs);
    low = localOnePoleLowpass(x, fs, cutoffHz);
    high = x - low;
    y = low + highFreqScale * high;
end

function y = localOnePoleLowpass(x, fs, cutoffHz)
    rc = 1 / (2*pi*cutoffHz);
    dt = 1 / fs;
    alpha = dt / (rc + dt);

    y = zeros(size(x));
    y(1) = alpha * x(1);
    for k = 2:numel(x)
        y(k) = y(k-1) + alpha * (x(k) - y(k-1));
    end
end

function noise = localColoredNoise(N, fs, type)
    white = randn(N,1);

    switch lower(type)
        case 'wind'
            % Low-frequency turbulent noise with slow amplitude modulation.
            n1 = localOnePoleLowpass(white, fs, 80);
            n2 = localOnePoleLowpass(randn(N,1), fs, 6);
            env = 0.6 + 0.4 * abs(localNormalizeSource(n2));
            noise = env .* n1;

        case 'pinkish'
            % Simple pink-ish noise: lowpass plus some broadband component.
            noise = 0.7 * localOnePoleLowpass(white, fs, 1200) + 0.3 * white;

        otherwise
            noise = white;
    end

    noise = localNormalizeSource(noise);
end

function vib = localDroneVibrationSignal(N, fs, droneAudio)
    % Creates low-frequency body vibration coupled to drone audio envelope.
    t = (0:N-1).' / fs;

    f1 = 35 + 10*rand();
    f2 = 70 + 20*rand();
    f3 = 110 + 25*rand();

    harmonic = sin(2*pi*f1*t + 2*pi*rand()) + ...
               0.55*sin(2*pi*f2*t + 2*pi*rand()) + ...
               0.30*sin(2*pi*f3*t + 2*pi*rand());

    env = abs(localOnePoleLowpass(droneAudio(:), fs, 12));
    env = localFitLength(env, N);
    if max(env) > 0
        env = env / max(env);
    else
        env = ones(N,1);
    end

    vib = env .* harmonic;
    vib = localNormalizeSource(vib);
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
        room = localBuildRoom(humanMouthPos, opt.IndoorWallDistance, opt.RoomHeight);
        xMin = room.xMin; xMax = room.xMax;
        yMin = room.yMin; yMax = room.yMax;
        zMin = room.zMin; zMax = room.zMax;

        plot3([xMin xMax xMax xMin xMin], [yMin yMin yMax yMax yMin], ...
            [zMin zMin zMin zMin zMin], 'k--', 'DisplayName', 'Indoor room');

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
