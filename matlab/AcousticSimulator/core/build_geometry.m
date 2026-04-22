function geo = build_geometry(cfg)
%BUILD_GEOMETRY  Compute scene geometry and per-mic TDOA (in samples).
%
%   geo = build_geometry(cfg) returns positions for the speaker, its
%   ground image source, the drone body, every virtual mic in the ULA,
%   and an ambient-noise point source, plus inter-mic delays used by
%   expand_channels() to fabricate a virtual multi-channel capture from
%   a mono/stereo MacBook input.

    c   = cfg.c;
    mh  = cfg.mouth_height;

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
    d_env    = vecnorm(pos_mics - pos_env,      2, 2);

    tau_abs  = round(d_speech / c * cfg.fs);
    delays   = tau_abs - min(tau_abs);

    geo.pos_human     = pos_human;
    geo.pos_img_src   = pos_img_src;
    geo.pos_drone     = pos_drone;
    geo.pos_mics      = pos_mics;
    geo.pos_env_noise = pos_env;
    geo.drone_agl     = pos_drone(3);
    geo.dist_speech   = d_speech;
    geo.dist_img      = d_img;
    geo.dist_env      = d_env;
    geo.delays        = delays;
    geo.grazing_deg   = rad2deg(asin(mh ./ d_img));
end
