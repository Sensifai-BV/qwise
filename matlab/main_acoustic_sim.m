function main_acoustic_sim()
%% Unified Real-Time Acoustic Simulation
%
%  Single file · Single figure
%  Left  : controls  (Start/Stop · Drone WAV · Env WAV · gain sliders)
%  Top-R : 3D acoustic scene (geometry, static)
%  Top-R2: live spectrogram (Mic 1)
%
%  Setup:
%    1. Place WAVs at  noise_files/drone_fan.wav
%                      noise_files/env_ambient.wav
%       (pink-noise placeholders generated if files are missing)
%    2. Run:  main_acoustic_sim

    cfg = get_config();
    geo = build_geometry(cfg);
    print_geometry(geo);

    drone_wav = load_wav_loop(cfg.drone_wav_path, cfg.fs, cfg.loop_sec);
    env_wav   = load_wav_loop(cfg.env_wav_path,   cfg.fs, cfg.loop_sec);
    fprintf('[Q-WiSE] WAV loops ready (%.0f s each)\n', cfg.loop_sec);

    [adr, n_hw_ch] = create_reader(cfg);
    fprintf('[Q-WiSE] Mic: %d hw channel(s) → %d virtual channels\n', ...
            n_hw_ch, cfg.n_mics);

    % ------------------------------------------------------------------
    % Application state stored in figure UserData
    % ------------------------------------------------------------------
    S.running     = false;
    S.drone_on    = false;
    S.env_on      = false;
    S.drone_ptr   = 1;
    S.env_ptr     = 1;
    S.drone_wav   = drone_wav;
    S.env_wav     = env_wav;
    S.drone_gain  = cfg.drone_gain_init;
    S.env_gain    = cfg.env_gain_init;
    S.n_hw_ch     = n_hw_ch;
    S.geo         = geo;
    S.cfg         = cfg;
    S.adr         = adr;

    % ---------------------------------------------------------------
    % FIX 1: Create audioDeviceWriter so mixed WAV can play to speakers
    % ---------------------------------------------------------------
    S.adw         = audioDeviceWriter('SampleRate', cfg.fs, ...
                                      'BufferSize', cfg.frame_size);

    S.timer       = [];          % assigned after figure is created
    nfft          = cfg.frame_size;
    S.spec_ncols  = 90;
    S.spec_buf    = zeros(nfft/2+1, S.spec_ncols);
    S.spec_col    = 1;

    [hfig, H] = build_figure(cfg, geo, S);
    S.H = H;
    hfig.UserData = S;

    % Create timer AFTER figure so the handle is valid in the callback
    t = timer('ExecutionMode', 'fixedRate', ...
              'Period',        max(0.05, cfg.frame_size / cfg.fs), ...
              'BusyMode',      'drop', ...
              'TimerFcn',      @(~,~) rt_loop(hfig), ...
              'StopFcn',       @(~,~) safe_release(hfig));

    Stmp       = hfig.UserData;
    Stmp.timer = t;
    hfig.UserData = Stmp;

    hfig.DeleteFcn = @(~,~) stop_and_delete(t);

    fprintf('[Q-WiSE] Ready — press Start to begin.\n');
    waitfor(hfig);
end


% =========================================================================
%  REAL-TIME LOOP
% =========================================================================
function rt_loop(hfig)
    if ~isvalid(hfig), return; end
    S   = hfig.UserData;
    if ~S.running || isempty(S.adr), return; end

    N = S.cfg.frame_size;
    try
        raw = S.adr();
    catch
        return;
    end

    mic = expand_channels(raw, S.geo, S.cfg.n_mics, S.cfg.fs);

    % === Drone noise mixing ===
    if S.drone_on
        chunk       = loop_chunk(S.drone_wav, S.drone_ptr, N);
        S.drone_ptr = mod(S.drone_ptr + N - 1, length(S.drone_wav)) + 1;
        mic = mic + S.drone_gain * repmat(chunk, 1, S.cfg.n_mics);

        % Debug print (remove later if you want)
        fprintf('Drone: max|chunk|=%.4f  gain=%.2f\n', max(abs(chunk)), S.drone_gain);
    end

    % === Env noise mixing ===
    if S.env_on
        chunk     = loop_chunk(S.env_wav, S.env_ptr, N);
        S.env_ptr = mod(S.env_ptr + N - 1, length(S.env_wav)) + 1;
        mic = mic + S.env_gain * repmat(chunk, 1, S.cfg.n_mics);
    end

    win = hann(N, 'periodic');
    X   = abs(fft(mic(:,1) .* win, N));
    S.spec_buf(:, S.spec_col) = X(1:N/2+1);
    S.spec_col = mod(S.spec_col, S.spec_ncols) + 1;

    % ---------------------------------------------------------------
    % FIX 2: Send mixed audio to speakers via audioDeviceWriter.
    %         Without this call the WAV was mixed but never played.
    % ---------------------------------------------------------------
    out = mean(mic, 2);           % mono mix across all virtual mics
    out = max(min(out, 1), -1);   % hard-clip guard
    try S.adw(out); catch, end   % write to output device

    hfig.UserData = S;
    refresh_display(hfig, mic);
end

% =========================================================================
%  DISPLAY
% =========================================================================
function refresh_display(hfig, mic)
    if ~isvalid(hfig), return; end
    S = hfig.UserData;
    H = S.H;

    for m = 1:S.cfg.n_mics
        pk = max(abs(mic(:,m)));
        if pk > 1e-9
            set(H.hlines(m), 'YData', mic(:,m) / pk);
        end
    end
    set(H.himg, 'CData', 20*log10(S.spec_buf + 1e-12));
    drawnow limitrate;
end


% =========================================================================
%  FIGURE BUILDER
% =========================================================================
function [hfig, H] = build_figure(cfg, geo, S)
    hfig = figure('Name','Q-WiSE  |  Acoustic Simulation + Live Capture', ...
                  'Color',[0.10 0.10 0.10], ...
                  'Position',[50 40 1440 840], ...
                  'MenuBar','none','ToolBar','figure', ...
                  'NumberTitle','off');

    cpw  = 0.148;    % control panel normalised width

    % ---- LEFT: control panel ----
    ctrl = uipanel(hfig, ...
        'BackgroundColor',[0.14 0.14 0.14], ...
        'BorderType','none', ...
        'Units','normalized', ...
        'Position',[0 0 cpw 1]);

    uicontrol(ctrl,'Style','text','String','Q-WiSE', ...
        'FontSize',13,'FontWeight','bold', ...
        'BackgroundColor',[0.14 0.14 0.14],'ForegroundColor',[0.95 0.88 0.55], ...
        'Units','normalized','Position',[0.03 0.94 0.94 0.055]);

    % Start / Stop
    H.btn_start = uicontrol(ctrl,'Style','togglebutton', ...
        'String','▶  Start Capture', ...
        'FontSize',10,'FontWeight','bold', ...
        'BackgroundColor',[0.16 0.60 0.25],'ForegroundColor','w', ...
        'Units','normalized','Position',[0.05 0.865 0.90 0.068], ...
        'Callback',@(src,~) cb_start(src, hfig));

    gui_sep(ctrl, 0.850);

    % Drone noise toggle
    H.btn_drone = uicontrol(ctrl,'Style','togglebutton', ...
        'String','🔇  Drone: OFF', ...
        'FontSize',9,'BackgroundColor',[0.22 0.22 0.22], ...
        'ForegroundColor',[0.60 0.60 0.60], ...
        'Units','normalized','Position',[0.05 0.77 0.90 0.062], ...
        'Callback',@(src,~) cb_toggle(src, hfig, 'drone'));

    H.lbl_dg = uicontrol(ctrl,'Style','text', ...
        'String',sprintf('Drone gain  %.2f',S.drone_gain), ...
        'FontSize',8,'BackgroundColor',[0.14 0.14 0.14], ...
        'ForegroundColor',[0.58 0.58 0.58], ...
        'Units','normalized','Position',[0.05 0.727 0.90 0.034]);
    H.sld_drone = uicontrol(ctrl,'Style','slider', ...
        'Min',0,'Max',1,'Value',S.drone_gain, ...
        'Units','normalized','Position',[0.05 0.693 0.90 0.032], ...
        'Callback',@(src,~) cb_gain(src,hfig,'drone'));

    gui_sep(ctrl, 0.678);

    % Env noise toggle
    H.btn_env = uicontrol(ctrl,'Style','togglebutton', ...
        'String','🔇  Env: OFF', ...
        'FontSize',9,'BackgroundColor',[0.22 0.22 0.22], ...
        'ForegroundColor',[0.60 0.60 0.60], ...
        'Units','normalized','Position',[0.05 0.594 0.90 0.062], ...
        'Callback',@(src,~) cb_toggle(src, hfig, 'env'));

    H.lbl_eg = uicontrol(ctrl,'Style','text', ...
        'String',sprintf('Env gain  %.2f',S.env_gain), ...
        'FontSize',8,'BackgroundColor',[0.14 0.14 0.14], ...
        'ForegroundColor',[0.58 0.58 0.58], ...
        'Units','normalized','Position',[0.05 0.551 0.90 0.034]);
    H.sld_env = uicontrol(ctrl,'Style','slider', ...
        'Min',0,'Max',1,'Value',S.env_gain, ...
        'Units','normalized','Position',[0.05 0.517 0.90 0.032], ...
        'Callback',@(src,~) cb_gain(src,hfig,'env'));

    gui_sep(ctrl, 0.500);

    % Geometry summary
    bpf = round(cfg.drone_rpm * cfg.drone_blades / 60);
    gstr = sprintf( ...
        'Human : %.0f cm\nMouth z: %.2f m\nDrone AGL: %.2f m\nSlant: %.2f m\nAsphalt R: %.2f\nBPF: %d Hz\nMic spacing: %.0f cm', ...
        cfg.human_height*100, cfg.mouth_height, geo.drone_agl, ...
        norm(geo.pos_drone-geo.pos_human), cfg.ground_R, bpf, ...
        cfg.mic_spacing*100);
    uicontrol(ctrl,'Style','text','String',gstr, ...
        'FontSize',8,'BackgroundColor',[0.14 0.14 0.14], ...
        'ForegroundColor',[0.50 0.78 0.50], ...
        'Units','normalized','Position',[0.03 0.27 0.94 0.22], ...
        'HorizontalAlignment','left');

    gui_sep(ctrl, 0.255);

    H.lbl_status = uicontrol(ctrl,'Style','text', ...
        'String',sprintf('● Idle\nfs=%d Hz\nframe=%d smp', ...
                 cfg.fs, cfg.frame_size), ...
        'FontSize',8,'BackgroundColor',[0.14 0.14 0.14], ...
        'ForegroundColor',[0.50 0.50 0.50], ...
        'Units','normalized','Position',[0.03 0.01 0.94 0.23], ...
        'HorizontalAlignment','left');

    % ---- RIGHT: axes ----
    rx  = cpw + 0.008;
    rw  = 1 - rx - 0.010;

    % 3D scene (top-left of right area)
    H.ax_scene = axes(hfig, 'Units','normalized', ...
        'Position',[rx 0.47 rw*0.54 0.50], ...
        'Color',[0.07 0.07 0.07], ...
        'XColor',[0.50 0.50 0.50],'YColor',[0.50 0.50 0.50], ...
        'ZColor',[0.50 0.50 0.50],'GridColor',[0.22 0.22 0.22]);

    % Spectrogram (top-right of right area)
    H.ax_spec = axes(hfig, 'Units','normalized', ...
        'Position',[rx + rw*0.56 0.47 rw*0.43 0.50], ...
        'Color',[0.07 0.07 0.07], ...
        'XColor',[0.50 0.50 0.50],'YColor',[0.50 0.50 0.50], ...
        'GridColor',[0.22 0.22 0.22]);

    % Waveform axes — 3 rows in bottom half
    N   = cfg.frame_size;
    wh  = 0.42 / cfg.n_mics;
    mc  = {[0.00 0.78 0.32],[0.00 0.55 0.95],[1.00 0.52 0.00]};
    H.ax_wav  = gobjects(cfg.n_mics,1);
    H.hlines  = gobjects(cfg.n_mics,1);
    for m = 1:cfg.n_mics
        yp = 0.03 + (cfg.n_mics-m)*wh;
        H.ax_wav(m) = axes(hfig,'Units','normalized', ...
            'Position',[rx yp rw wh-0.010], ...
            'Color',[0.06 0.06 0.06], ...
            'XColor',[0.45 0.45 0.45],'YColor',mc{m}, ...
            'GridColor',[0.20 0.20 0.20]);
        hold(H.ax_wav(m),'on'); grid(H.ax_wav(m),'on');
        ylim(H.ax_wav(m),[-1.05 1.05]);
        xlim(H.ax_wav(m),[1 N]);
        ylabel(H.ax_wav(m), sprintf('Mic %d',m), ...
               'Color',mc{m},'FontSize',9);
        H.hlines(m) = plot(H.ax_wav(m), 1:N, zeros(N,1), ...
                           'Color',mc{m},'LineWidth',0.75);
        if m == 1
            title(H.ax_wav(m),'Live Waveforms  (MacBook 3-mic array)', ...
                  'Color','w','FontSize',9,'FontWeight','bold');
        end
    end
    xlabel(H.ax_wav(cfg.n_mics),'Sample','Color',[0.50 0.50 0.50]);

    % Spectrogram image
    nbin   = N/2+1;
    ncols  = S.spec_ncols;
    freqs  = linspace(0, cfg.fs/2000, nbin);
    H.himg = imagesc(H.ax_spec, 1:ncols, freqs, ...
                     20*log10(S.spec_buf + 1e-12));
    colormap(H.ax_spec,'hot');
    clim(H.ax_spec,[-80 -20]);
    axis(H.ax_spec,'xy');
    xlabel(H.ax_spec,'Frame','Color',[0.50 0.50 0.50]);
    ylabel(H.ax_spec,'Freq (kHz)','Color',[0.50 0.50 0.50]);
    title(H.ax_spec,'Spectrogram — Mic 1','Color','w', ...
          'FontSize',9,'FontWeight','bold');
    cb = colorbar(H.ax_spec);
    cb.Color = [0.60 0.60 0.60];
    cb.Label.String = 'dB';
    cb.Label.Color  = [0.60 0.60 0.60];

    % Static 3D scene
    draw_scene(H.ax_scene, geo, cfg);
end


% =========================================================================
%  STATIC 3D SCENE
% =========================================================================
function draw_scene(ax, geo, cfg)
    hold(ax,'on'); grid(ax,'on'); box(ax,'on');
    set(ax,'DataAspectRatio',[1 1 1],'PlotBoxAspectRatio',[1.3 1.1 1]);
    set(ax,'XLim',[-0.5 5.5],'YLim',[-1.4 4.0],'ZLim',[-1.6 3.8]);

    % Asphalt ground
    [gx,gy] = meshgrid(linspace(-0.4,5.3,8), linspace(-1.2,3.8,8));
    surf(ax, gx, gy, zeros(size(gx)), ...
         'FaceColor',[0.27 0.27 0.27],'FaceAlpha',0.25, ...
         'EdgeColor',[0.42 0.42 0.42],'EdgeAlpha',0.12);
    text(0.2,-1.0,0.02,'Asphalt (R=0.90)','FontSize',7, ...
         'Color',[0.48 0.48 0.48],'Parent',ax);

    % Human
    xh = geo.pos_human(1);  yh = geo.pos_human(2);
    H  = cfg.human_height;
    plot3(ax,[xh xh xh],[yh yh yh],[0 H*0.52 H*0.87], ...
          'Color',[0.30 0.45 0.90],'LineWidth',2.5);
    th = linspace(0,2*pi,36);  rh = 0.08;
    plot3(ax, xh+rh*cos(th), yh+zeros(1,36), H*0.935+rh*sin(th), ...
          'Color',[0.30 0.45 0.90],'LineWidth',2.0);
    scatter3(ax,xh,yh,geo.pos_human(3),65,[0.30 0.55 1.0],'filled','^');
    text(xh+0.09,yh,geo.pos_human(3)+0.09, ...
         sprintf('Mouth %.2fm',geo.pos_human(3)), ...
         'FontSize',7,'Color',[0.40 0.65 1.0],'Parent',ax);
    text(xh-0.08,yh,H+0.14,sprintf('%.0fcm',H*100), ...
         'FontSize',7,'FontWeight','bold','Color',[0.40 0.55 0.90],'Parent',ax);

    % Image source
    scatter3(ax,geo.pos_img_src(1),geo.pos_img_src(2), ...
             geo.pos_img_src(3),40,[0.50 0.50 0.50],'d','LineWidth',1.2);
    plot3(ax,[xh xh],[yh yh],[geo.pos_human(3) geo.pos_img_src(3)], ...
          ':','Color',[0.52 0.52 0.52],'LineWidth',0.9);
    text(geo.pos_img_src(1)+0.08,geo.pos_img_src(2), ...
         geo.pos_img_src(3)-0.13, ...
         sprintf('img z=%.2f',geo.pos_img_src(3)), ...
         'FontSize',7,'Color',[0.50 0.50 0.50],'Parent',ax);

    % Drone 3D body
    draw_drone(ax, geo.pos_drone, geo.pos_mics, cfg);

    % Env noise
    scatter3(ax,geo.pos_env_noise(1),geo.pos_env_noise(2), ...
             geo.pos_env_noise(3),75,[0.65 0.10 0.85],'p','filled');
    text(geo.pos_env_noise(1)+0.09,geo.pos_env_noise(2), ...
         geo.pos_env_noise(3)+0.11,'Env Noise', ...
         'FontSize',7,'Color',[0.65 0.10 0.85],'Parent',ax);

    % Direct path
    d = norm(geo.pos_drone - geo.pos_human);
    plot3(ax,[xh geo.pos_drone(1)],[yh geo.pos_drone(2)], ...
             [geo.pos_human(3) geo.pos_drone(3)], ...
          '--','Color',[0.30 0.55 1.0],'LineWidth',1.5);
    mid = (geo.pos_human+geo.pos_drone)/2;
    text(mid(1)+0.07,mid(2),mid(3)+0.08,sprintf('%.2fm',d), ...
         'FontSize',7,'Color',[0.40 0.65 1.0],'Parent',ax);

    % Reflected path (centre mic)
    img  = geo.pos_img_src;
    mc2  = geo.pos_mics(2,:);
    tsp  = -img(3)/(mc2(3)-img(3));
    sp   = img + tsp*(mc2-img);
    plot3(ax,[xh sp(1)],[yh sp(2)],[geo.pos_human(3) sp(3)], ...
          '-','Color',[0.10 0.72 0.45],'LineWidth',1.2);
    plot3(ax,[sp(1) mc2(1)],[sp(2) mc2(2)],[sp(3) mc2(3)], ...
          '-','Color',[0.10 0.72 0.45],'LineWidth',1.2);
    scatter3(ax,sp(1),sp(2),sp(3),35,[0.10 0.62 0.38],'s','LineWidth',1.2);
    text(sp(1)+0.06,sp(2),sp(3)+0.07, ...
         sprintf('R=%.2f',cfg.ground_R), ...
         'FontSize',7,'Color',[0.10 0.72 0.45],'Parent',ax);

    % Env → drone
    plot3(ax,[geo.pos_env_noise(1) geo.pos_drone(1)], ...
             [geo.pos_env_noise(2) geo.pos_drone(2)], ...
             [geo.pos_env_noise(3) geo.pos_drone(3)], ...
          ':','Color',[0.60 0.10 0.80],'LineWidth',0.9);

    xlabel(ax,'X(m)'); ylabel(ax,'Y(m)'); zlabel(ax,'Z(m)');
    title(ax,'Acoustic Scene  (outdoor / asphalt)', ...
          'Color','w','FontSize',9,'FontWeight','bold');
    view(ax,38,24);
    camproj(ax,'perspective');
end


function draw_drone(ax, pd, pm, cfg)
    % 3D quad-rotor: box body + arms + rotor discs.
    arm_l  = 0.19;  bw = 0.09;  bh = 0.045;
    blift  = 0.075; rr = 0.075;

    mz  = pm(1,3);
    bz0 = mz + blift;
    bz1 = bz0 + bh;
    rz  = bz1 + 0.008;
    dx  = pd(1);  dy = pd(2);

    % Box
    V = [dx-bw dy-bw bz0; dx+bw dy-bw bz0; dx+bw dy+bw bz0; dx-bw dy+bw bz0;
         dx-bw dy-bw bz1; dx+bw dy-bw bz1; dx+bw dy+bw bz1; dx-bw dy+bw bz1];
    F = [1 2 3 4; 5 6 7 8; 1 2 6 5; 3 4 8 7; 2 3 7 6; 1 4 8 5];
    patch(ax,'Vertices',V,'Faces',F, ...
          'FaceColor',[0.13 0.13 0.13],'FaceAlpha',0.88, ...
          'EdgeColor','k','EdgeAlpha',0.85,'LineWidth',0.8);

    % Arms + rotors
    rcx = dx + arm_l*[ 1 -1  0  0];
    rcy = dy + arm_l*[ 0  0  1 -1];
    th  = linspace(0,2*pi,32);
    for k = 1:4
        plot3(ax,[dx rcx(k)],[dy rcy(k)],[bz1 rz], ...
              '-','Color',[0.22 0.22 0.22],'LineWidth',2.3);
        rxc = rcx(k)+rr*cos(th);
        ryc = rcy(k)+rr*sin(th);
        patch(ax,rxc,ryc,rz*ones(1,32),[0.82 0.14 0.14], ...
              'FaceAlpha',0.52,'EdgeColor',[0.55 0.08 0.08],'LineWidth',0.8);
        plot3(ax,[rcx(k)-rr rcx(k)+rr],[rcy(k) rcy(k)],[rz rz], ...
              'k-','LineWidth',1.2);
        plot3(ax,[rcx(k) rcx(k)],[rcy(k)-rr rcy(k)+rr],[rz rz], ...
              'k-','LineWidth',1.2);
    end

    % Mic markers
    mc = {[0.00 0.78 0.32],[0.00 0.55 0.95],[1.00 0.52 0.00]};
    for m = 1:size(pm,1)
        scatter3(ax,pm(m,1),pm(m,2),pm(m,3),60,mc{m},'filled','o');
        text(pm(m,1),pm(m,2),pm(m,3)+0.055, ...
             sprintf('M%d',m),'FontSize',7,'Color',mc{m},'Parent',ax);
    end

    bpf = round(cfg.drone_rpm*cfg.drone_blades/60);
    text(dx+0.11,dy,rz+0.10, ...
         sprintf('BPF %dHz  AGL %.2fm',bpf,pd(3)), ...
         'FontSize',7,'FontWeight','bold','Color',[0.90 0.18 0.18],'Parent',ax);
end


% =========================================================================
%  GUI CALLBACKS
% =========================================================================
function cb_start(src, hfig)
    S = hfig.UserData;
    if isempty(S.timer)
        set(src,'Value',0);
        warndlg('Timer not ready yet. Wait a moment and try again.','Q-WiSE');
        return;
    end
    if src.Value == 1
        S.running = true;
        src.String          = '⏹  Stop Capture';
        src.BackgroundColor = [0.72 0.16 0.16];
        S.H.lbl_status.String = ...
            sprintf('● Capturing\nch: %d  fs: %d\nframe: %d smp', ...
                    S.n_hw_ch, S.cfg.fs, S.cfg.frame_size);
        S.H.lbl_status.ForegroundColor = [0.28 0.88 0.38];
        hfig.UserData = S;
        start(S.timer);
    else
        S.running = false;
        src.String          = '▶  Start Capture';
        src.BackgroundColor = [0.16 0.60 0.25];
        S.H.lbl_status.String = '● Stopped';
        S.H.lbl_status.ForegroundColor = [0.70 0.42 0.18];
        hfig.UserData = S;
        stop(S.timer);
    end
end

function cb_toggle(src, hfig, which)
    S = hfig.UserData;
    on = logical(src.Value);
    if strcmp(which,'drone')
        S.drone_on = on;
        if on
            src.String = '🔊  Drone: ON';
            src.BackgroundColor = [0.68 0.32 0.06];
            src.ForegroundColor = 'w';
        else
            src.String = '🔇  Drone: OFF';
            src.BackgroundColor = [0.22 0.22 0.22];
            src.ForegroundColor = [0.60 0.60 0.60];
        end
    else
        S.env_on = on;
        if on
            src.String = '🔊  Env: ON';
            src.BackgroundColor = [0.36 0.06 0.58];
            src.ForegroundColor = 'w';
        else
            src.String = '🔇  Env: OFF';
            src.BackgroundColor = [0.22 0.22 0.22];
            src.ForegroundColor = [0.60 0.60 0.60];
        end
    end
    hfig.UserData = S;
end

function cb_gain(src, hfig, which)
    S = hfig.UserData;
    if strcmp(which,'drone')
        S.drone_gain = src.Value;
        S.H.lbl_dg.String = sprintf('Drone gain  %.2f', src.Value);
    else
        S.env_gain = src.Value;
        S.H.lbl_eg.String = sprintf('Env gain  %.2f', src.Value);
    end
    hfig.UserData = S;
end


% =========================================================================
%  AUDIO DEVICE
% =========================================================================
function [adr, n_ch] = create_reader(cfg)
    dev  = find_mac_mic();
    n_ch = probe_channels(dev, cfg.fs, cfg.frame_size, cfg.n_mics);
    try
        if isempty(dev)
            adr = audioDeviceReader( ...
                'SampleRate',      cfg.fs, ...
                'NumChannels',     n_ch, ...
                'SamplesPerFrame', cfg.frame_size);
        else
            adr = audioDeviceReader( ...
                'Device',          dev, ...
                'SampleRate',      cfg.fs, ...
                'NumChannels',     n_ch, ...
                'SamplesPerFrame', cfg.frame_size);
        end
        adr();   % warm-up
    catch ME
        warning(1, '[Q-WiSE] %s\nFalling back to default mono.', ME.message);
        adr  = audioDeviceReader('SampleRate',cfg.fs,'NumChannels',1, ...
                                 'SamplesPerFrame',cfg.frame_size);
        n_ch = 1;
    end
end

function dev = find_mac_mic()
    dev = '';
    try
        devs = audioDeviceReader.getAudioDevices();
        kw   = {'MacBook Pro Microphone','Built-in Microphone', ...
                'MacBook Air Microphone','Apple Silicon Microphone'};
        for k = 1:numel(kw)
            m = devs(contains(devs, kw{k}, 'IgnoreCase', true));
            if ~isempty(m)
                dev = m{1};  return;
            end
        end
    catch
    end
end

function n = probe_channels(dev, fs, frame, desired)
    for n = [desired, 2, 1]
        try
            if isempty(dev)
                t = audioDeviceReader('SampleRate',fs,'NumChannels',n,'SamplesPerFrame',frame);
            else
                t = audioDeviceReader('Device',dev,'SampleRate',fs,'NumChannels',n,'SamplesPerFrame',frame);
            end
            t();  release(t);  return;
        catch
        end
    end
    n = 1;
end


% =========================================================================
%  AUDIO HELPERS
% =========================================================================
function [y, fs_orig] = load_wav_loop(wav_path, target_fs, loop_sec)
    % Make path absolute relative to the script location
    if ~startsWith(wav_path, filesep) && ~contains(wav_path, ':')  % not absolute
        script_dir = fileparts(mfilename('fullpath'));
        wav_path = fullfile(script_dir, wav_path);
    end

    if ~isfile(wav_path)
        fprintf('[Q-WiSE] %s not found — generating placeholder\n', wav_path);
        fs_orig = target_fs;
        N = round(loop_sec * target_fs);
        w = randn(N,1);
        [b,a] = butter(1, 0.015);
        y = filter(b,a,w);
    else
        fprintf('[Q-WiSE] Loading %s successfully\n', wav_path);
        [y, fs_orig] = audioread(wav_path);
        if size(y,2) > 1,  y = mean(y,2);  end
        if fs_orig ~= target_fs
            y = resample(y, target_fs, fs_orig);
        end
        n_rep = ceil(loop_sec*target_fs / length(y));
        y     = repmat(y, n_rep, 1);
        y     = y(1:round(loop_sec*target_fs));
    end
    y = y / (rms(y) + 1e-12);
end

function c = loop_chunk(wav, ptr, N)
    L   = length(wav);
    idx = mod(ptr - 1 + (0:N-1)', L) + 1;   % column vector
    c   = wav(idx);
end

function out = expand_channels(raw, geo, n_mics, fs)   %#ok<INUSD>
    N   = size(raw,1);
    out = zeros(N, n_mics);
    src = raw(:,1);
    for m = 1:n_mics
        tau = geo.delays(m);
        if tau == 0
            out(:,m) = src;
        elseif tau < N
            out(:,m) = [zeros(tau,1); src(1:N-tau)];
        end
    end
    n_hw = size(raw,2);
    if n_hw >= 2 && n_mics >= 2
        out(:,1)      = raw(:,1);
        out(:,n_mics) = raw(:,n_hw);
    end
end


% =========================================================================
%  GEOMETRY
% =========================================================================
function geo = build_geometry(cfg)
    c   = cfg.c;
    mh  = cfg.mouth_height;

    pos_human   = [0.0, 0.0, mh];
    pos_img_src = [0.0, 0.0, -mh];

    horiz     = cfg.slant_dist * cos(deg2rad(cfg.elev_deg));
    vert_off  = cfg.slant_dist * sin(deg2rad(cfg.elev_deg));
    pos_drone = [horiz, 0.0, mh + vert_off];

    sp      = cfg.mic_spacing;
    off     = sp * (-(cfg.n_mics-1)/2 : (cfg.n_mics-1)/2);
    pos_mics = pos_drone + [off', zeros(cfg.n_mics,1), zeros(cfg.n_mics,1)];

    pos_env   = [4.5, 3.0, 0.3];

    d_speech  = vecnorm(pos_mics - pos_human,   2, 2);
    d_img     = vecnorm(pos_mics - pos_img_src,  2, 2);
    d_env     = vecnorm(pos_mics - pos_env,       2, 2);

    tau_abs   = round(d_speech / c * cfg.fs);
    delays    = tau_abs - min(tau_abs);

    geo.pos_human    = pos_human;
    geo.pos_img_src  = pos_img_src;
    geo.pos_drone    = pos_drone;
    geo.pos_mics     = pos_mics;
    geo.pos_env_noise= pos_env;
    geo.drone_agl    = pos_drone(3);
    geo.dist_speech  = d_speech;
    geo.dist_img     = d_img;
    geo.dist_env     = d_env;
    geo.delays       = delays;
    geo.grazing_deg  = rad2deg(asin(mh ./ d_img));
end

function print_geometry(geo)
    fprintf('=== Geometry (outdoor / asphalt) ===\n');
    fprintf('  Mouth     : [%.3f %.3f %.3f] m\n', geo.pos_human);
    fprintf('  Image src : [%.3f %.3f %.3f] m\n', geo.pos_img_src);
    fprintf('  Drone     : [%.3f %.3f %.3f] m  AGL=%.2f m\n', ...
            geo.pos_drone, geo.drone_agl);
    for m = 1:size(geo.pos_mics,1)
        fprintf('  Mic%d pos=[%.3f %.3f %.3f]  d=%.3f m  TDOA=%d smp\n', ...
            m, geo.pos_mics(m,1), geo.pos_mics(m,2), geo.pos_mics(m,3), ...
            geo.dist_speech(m), geo.delays(m));
    end
    fprintf('\n');
end


% =========================================================================
%  CLEANUP
% =========================================================================
function safe_release(hfig)
    if ~isvalid(hfig), return; end
    S = hfig.UserData;
    if ~isempty(S.adr)
        try release(S.adr); catch; end
    end
    % ---------------------------------------------------------------
    % FIX 3: Release audioDeviceWriter so the output device is freed
    % ---------------------------------------------------------------
    if isfield(S,'adw') && ~isempty(S.adw)
        try release(S.adw); catch; end
    end
    fprintf('[Q-WiSE] Audio device released.\n');
end

function stop_and_delete(t)
    try
        if isvalid(t); stop(t); delete(t); end
    catch
    end
end

function gui_sep(parent, y)
    uicontrol(parent,'Style','text','String','', ...
        'BackgroundColor',[0.32 0.32 0.32], ...
        'Units','normalized','Position',[0.03 y 0.94 0.004]);
end


% =========================================================================
%  CONFIGURATION
% =========================================================================
function cfg = get_config()
    cfg.fs              = 16000;
    cfg.frame_size      = 1024;
    cfg.c               = 343;
    cfg.n_mics          = 3;

    cfg.human_height    = 1.70;
    cfg.mouth_height    = 0.88 * cfg.human_height;

    cfg.slant_dist      = 2.50;
    cfg.elev_deg        = 30;
    cfg.mic_spacing     = 0.10;

    cfg.drone_rpm       = 8000;
    cfg.drone_blades    = 3;

    cfg.ground_R        = 0.90;
    cfg.alpha_air_dB    = 0.004;

    cfg.drone_wav_path  = fullfile('noise_files','drone_fan.wav');
    cfg.env_wav_path    = fullfile('noise_files','env_ambient.wav');
    cfg.loop_sec        = 120;

    cfg.drone_gain_init = 0.40;
    cfg.env_gain_init   = 0.25;
end