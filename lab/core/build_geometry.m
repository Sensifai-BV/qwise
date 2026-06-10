function geo = build_geometry(cfg)
%BUILD_GEOMETRY  Compute scene geometry plus per-source TDOAs and gains.
%
%   geo = build_geometry(cfg)
%
%   Returns a geometry struct used by SourceMixer to synthesize N
%   physically realistic microphone signals (mixer_1.m-style):
%
%     * Human speech  — point source at mouth height in front of the drone.
%     * Drone body    — point source at the array centre (rotor / fan noise).
%     * Environment   — independent point source at a configurable
%                       distance/azimuth/elevation from the speaker (kept
%                       separate from the drone so the mixer can give it
%                       its own TDOA + 1/r gain on every microphone).
%
%   For each source s and mic m we compute
%       d_s(m)             (Euclidean distance)
%       frac_delay_s(m)    = (d_s(m) - min(d_s)) / c * fs    [samples, float]
%       delay_s(m)         = round(frac_delay_s(m))          [samples, int]
%       gain_s(m)          = d_ref / max(d_s(m), d_ref)
%
%   Frac delays are what SourceMixer uses (fractional taps via linear
%   interpolation). The integer `delays_*` fields are kept for the
%   diagnostic helpers (print_geometry, draw_scene).

    c     = cfg.c;
    mh    = cfg.mouth_height;
    d_ref = cfg.distance_ref;

    % --- Human mouth position (origin in xy, mouth height in z) -------
    pos_human   = [0.0, 0.0, mh];
    pos_img_src = [0.0, 0.0, -mh];                  % ground image (viz only)

    % --- Drone position via spherical offset from the mouth -----------
    drone_az  = local_drone_azimuth_deg_(cfg);
    pos_drone = pos_human + spherical_offset_( ...
                    cfg.slant_dist, drone_az, cfg.elev_deg);

    % --- Mic array centred on the drone -------------------------------
    pos_mics = build_mic_array_(pos_drone, cfg.n_mics, ...
                                cfg.mic_spacing, mic_geometry_(cfg));

    % --- Environment-noise position (independent point source) --------
    [env_dist, env_az, env_el] = local_env_spherical_(cfg);
    pos_env   = pos_human + spherical_offset_(env_dist, env_az, env_el);

    % --- Per-source per-mic distances ---------------------------------
    d_speech = vecnorm(pos_mics - pos_human, 2, 2);
    d_img    = vecnorm(pos_mics - pos_img_src, 2, 2);
    d_drone  = vecnorm(pos_mics - pos_drone,  2, 2);
    d_env    = vecnorm(pos_mics - pos_env,    2, 2);

    % --- Per-source fractional + integer TDOA  ------------------------
    [delays_speech,      frac_delays_speech] = samp_delays_(d_speech, c, cfg.fs);
    [delays_drone,       frac_delays_drone ] = samp_delays_(d_drone,  c, cfg.fs);
    [delays_env,         frac_delays_env   ] = samp_delays_(d_env,    c, cfg.fs);

    % --- Per-source 1/r spreading gain (clamped at d_ref) -------------
    gains_drone  = d_ref ./ max(d_drone,  d_ref);
    gains_env    = d_ref ./ max(d_env,    d_ref);

    % Speech gain with a tunable distance falloff. The pure physical 1/r
    % law (exponent 1) drops the talker to 0.5 at 2 m / 0.4 at 2.5 m, which
    % — against the fixed-level drone sitting ON the array — buries the
    % speech past ~1.5 m. A gentler exponent keeps the talker quieter with
    % distance (2 m and 2.5 m lower than 1 m and 1.5 m) without collapsing
    % it. cfg.speech_dist_exponent in [0, 1]:
    %     0   -> constant level (no distance effect)
    %     0.5 -> gentle falloff   (default: 1.0/0.84/0.71/0.63 at 1/1.5/2/2.5 m)
    %     1   -> full physical 1/r
    %   Inter-mic gain ratios and all TDOAs are preserved either way.
    alpha = speech_dist_exponent_(cfg);
    gains_speech = (d_ref ./ max(d_speech, d_ref)) .^ alpha;

    % --- Legacy aliases -----------------------------------------------
    delays = delays_speech;            % singular, speech-centric (legacy)

    geo.pos_human         = pos_human;
    geo.pos_img_src       = pos_img_src;
    geo.pos_drone         = pos_drone;
    geo.pos_mics          = pos_mics;
    geo.pos_env           = pos_env;
    geo.pos_env_noise     = pos_env;                  % legacy alias
    geo.drone_agl         = pos_drone(3);
    geo.dist_speech       = d_speech;
    geo.dist_img          = d_img;
    geo.dist_drone        = d_drone;
    geo.dist_env          = d_env;

    geo.delays            = delays;                   % legacy integer
    geo.delays_speech     = delays_speech;            % legacy integer
    geo.delays_drone      = delays_drone;             % legacy integer
    geo.delays_env        = delays_env;               % legacy integer
    geo.frac_delays_speech = frac_delays_speech;      % fractional (samples)
    geo.frac_delays_drone  = frac_delays_drone;
    geo.frac_delays_env    = frac_delays_env;

    geo.gains_speech      = gains_speech;
    geo.gains_drone       = gains_drone;
    geo.gains_env         = gains_env;
    geo.grazing_deg       = rad2deg(asin(mh ./ d_img));
    geo.distance_ref      = d_ref;
end

% --------------------------------------------------------------------
function a = speech_dist_exponent_(cfg)
%SPEECH_DIST_EXPONENT_  Distance-falloff exponent for the speech gain
%   (default 0.5). 0 = constant level, 1 = physical 1/r. Clamped to [0,1].
    a = 0.5;
    if isfield(cfg, 'speech_dist_exponent') && ~isempty(cfg.speech_dist_exponent)
        a = cfg.speech_dist_exponent;
    end
    a = max(0, min(1, a));
end

% --------------------------------------------------------------------
function [tau_int, tau_frac] = samp_delays_(d, c, fs)
%SAMP_DELAYS_  Per-mic sample delay (relative to closest mic so min == 0).
    tau_abs_frac = (d / c) * fs;
    tau_frac     = tau_abs_frac - min(tau_abs_frac);
    tau_int      = round(tau_frac);
end

function off = spherical_offset_(distance, azimuthDeg, elevationDeg)
%SPHERICAL_OFFSET_  Cartesian offset for a spherical (range, az, el) target.
    az = deg2rad(azimuthDeg);
    el = deg2rad(elevationDeg);
    off = distance * [cos(el)*cos(az), cos(el)*sin(az), sin(el)];
end

function pos = build_mic_array_(center, n, spacing, geometry)
%BUILD_MIC_ARRAY_  Build linear or circular array centred on `center`.
    center = reshape(center, 1, 3);
    switch lower(geometry)
        case 'linear'
            % ULA along the x-axis, centred on the drone.
            idx     = (0:n-1).' - (n-1)/2;
            offsets = [idx * spacing, zeros(n,1), zeros(n,1)];
            pos     = center + offsets;
        case 'circular'
            % Equal-chord circular array; adjacent mic distance ≈ spacing.
            theta   = (0:n-1).' * 2*pi/n;
            radius  = spacing / (2*sin(pi/max(n,2)));
            offsets = [radius*cos(theta), radius*sin(theta), zeros(n,1)];
            pos     = center + offsets;
        otherwise
            error('build_geometry:UnknownGeometry', ...
                  'Unsupported mic geometry "%s" (use ''linear'' or ''circular'').', geometry);
    end
end

function g = mic_geometry_(cfg)
    if isfield(cfg, 'mic_geometry') && ~isempty(cfg.mic_geometry)
        g = cfg.mic_geometry;
    else
        g = 'linear';
    end
end

function az = local_drone_azimuth_deg_(cfg)
    az = 0;
    if isfield(cfg, 'drone') && isstruct(cfg.drone) ...
       && isfield(cfg.drone, 'azimuth_deg')
        az = cfg.drone.azimuth_deg;
    end
end

function [dist, az, el] = local_env_spherical_(cfg)
%LOCAL_ENV_SPHERICAL_  Resolve env-source spherical params from cfg.
%   Defaults match mixer_1.m: 8 m, 135 deg azimuth, 0 deg elevation.
    dist = 8.0;
    az   = 135;
    el   = 0;
    if isfield(cfg, 'env') && isstruct(cfg.env)
        if isfield(cfg.env, 'distance_from_mouth')
            dist = cfg.env.distance_from_mouth;
        end
        if isfield(cfg.env, 'azimuth_deg')
            az = cfg.env.azimuth_deg;
        end
        if isfield(cfg.env, 'elevation_deg')
            el = cfg.env.elevation_deg;
        end
    end
end
