function geo = build_geometry(cfg)
%BUILD_GEOMETRY  Compute scene geometry plus per-source TDOAs and gains.
%
%   geo = build_geometry(cfg)
%
%   Returns a geometry struct used by SourceMixer to synthesize three
%   physically realistic microphone signals:
%
%     * Human speech  — point source at shoulder height, ~slant_dist away.
%     * Drone body    — co-located with the array centre (mic fan noise).
%     * Environment   — distant point source representing ambient noise.
%
%   For each source s and mic m we compute
%       tau_s(m)  = round( (d_s(m) - min(d_s)) / c * fs )    [samples]
%       gain_s(m) = d_ref / max(d_s(m), d_ref)               [1/d law]
%
%   Clamping by cfg.distance_ref makes the closest mic / source pair
%   unity-gain rather than exploding when d -> 0, and it keeps the
%   human-at-2.5 m case at gain ≈ 0.40 exactly as the physical model
%   predicts (1 / 2.5 = 0.4).

    c     = cfg.c;
    mh    = cfg.mouth_height;
    d_ref = cfg.distance_ref;

    pos_human   = [0.0, 0.0, mh];
    pos_img_src = [0.0, 0.0, -mh];

    horiz     = cfg.slant_dist * cos(deg2rad(cfg.elev_deg));
    vert_off  = cfg.slant_dist * sin(deg2rad(cfg.elev_deg));
    pos_drone = [horiz, 0.0, mh + vert_off];

    sp       = cfg.mic_spacing;
    off      = sp * (-(cfg.n_mics-1)/2 : (cfg.n_mics-1)/2);
    pos_mics = pos_drone + [off', zeros(cfg.n_mics,1), zeros(cfg.n_mics,1)];

    pos_env  = [4.5, 3.0, 0.3];

    d_speech = vecnorm(pos_mics - pos_human,   2, 2);
    d_img    = vecnorm(pos_mics - pos_img_src, 2, 2);
    d_drone  = vecnorm(pos_mics - pos_drone,   2, 2);
    d_env    = vecnorm(pos_mics - pos_env,     2, 2);

    % --- Per-source TDOA (minimum delay == 0) --------------------
    delays_speech = samp_delay(d_speech, c, cfg.fs);
    delays_drone  = samp_delay(d_drone,  c, cfg.fs);
    delays_env    = samp_delay(d_env,    c, cfg.fs);

    % --- Per-source 1/d spreading gain (clamped at d_ref) --------
    gains_speech = d_ref ./ max(d_speech, d_ref);
    gains_drone  = d_ref ./ max(d_drone,  d_ref);
    gains_env    = d_ref ./ max(d_env,    d_ref);

    % --- Legacy aliases kept for older callers -------------------
    delays = delays_speech;                     % speech-centric TDOA

    geo.pos_human     = pos_human;
    geo.pos_img_src   = pos_img_src;
    geo.pos_drone     = pos_drone;
    geo.pos_mics      = pos_mics;
    geo.pos_env_noise = pos_env;
    geo.drone_agl     = pos_drone(3);
    geo.dist_speech   = d_speech;
    geo.dist_img      = d_img;
    geo.dist_drone    = d_drone;
    geo.dist_env      = d_env;
    geo.delays        = delays;
    geo.delays_speech = delays_speech;
    geo.delays_drone  = delays_drone;
    geo.delays_env    = delays_env;
    geo.gains_speech  = gains_speech;
    geo.gains_drone   = gains_drone;
    geo.gains_env     = gains_env;
    geo.grazing_deg   = rad2deg(asin(mh ./ d_img));
    geo.distance_ref  = d_ref;
end

function tau = samp_delay(d, c, fs)
%SAMP_DELAY  Integer sample delay referenced to closest mic (min -> 0).
    tau_abs = round(d / c * fs);
    tau     = tau_abs - min(tau_abs);
end
