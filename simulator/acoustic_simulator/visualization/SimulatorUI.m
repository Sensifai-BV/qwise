classdef SimulatorUI < handle
%SIMULATORUI  Q-WiSE real-time visualization + control panel.
%
%   ui = SimulatorUI(cfg, geo, audio, vad_obj, mwf_obj)
%   ui.start()
%
%   Renders a single figure with:
%     * Control panel  (Start/Stop, drone toggle+gain, env toggle+gain,
%                       Play VAD Speech, Play MWF, Record Start/Stop, status)
%     * 3D scene       (speaker, drone, mic ULA, ground reflection)
%     * Noisy spectrogram + MWF-enhanced spectrogram  (side by side)
%     * Per-mic live waveforms (n_mics rows)
%     * MWF output waveform
%     * VAD score trace with detection shading
%
%   The RT timer callback:
%     1. reads one live mic block (clean human speech)
%     2. pulls one block of drone-fan and ambient noise loops
%     3. feeds the three sources through SourceMixer, which applies
%        per-source TDOA + 1/d gain to produce the n-mic composite
%     4. runs VAD on the reference mic of the composite
%     5. runs the MWF (pass-through stub for now)
%     6. emits exactly the outputs the user has opted-in to
%        (VAD-gated composite OR MWF, never the raw noisy mix)
%     7. optionally appends the block to an active WAV recording.

    properties
        Cfg
        geo
        audio
        vad
        mwf
        mixer
        fig
        H = struct()

        running         = false
        drone_on        = false
        env_on          = false
        drone_gain
        env_gain
        vad_on          = false   % run VAD stage in the pipeline
        mwf_on          = false   % run MWF stage in the pipeline (requires VAD)
        recording       = false   % is a WAV capture in progress
        rec_kind_       = ''      % short label of the active recording session
        rec_tracks_     = struct('mic', false, 'vad', false, 'mwf', false)
        speech_source   = 'mic'   % 'mic' | 'wav'
    end

    properties (Access = private)
        timerObj
        spec_buf_noisy
        spec_buf_mwf
        spec_col  = 1
        spec_ncols
        vad_trace
        vad_flags
        % --- Latest-recording playback state ------------------------
        playback_panel_           % uipanel that hosts the play buttons
        playback_player_   = []   % active audioplayer
        playback_active_btn_ = [] % togglebutton currently in ON state
        playback_files_    = {}   % cellstr of WAV paths in the latest dir
        playback_dir_      = ''   % the directory those files live in
    end

    methods
        function obj = SimulatorUI(cfg, geo, audio, vad_obj, mwf_obj)
            obj.Cfg   = cfg;
            obj.geo   = geo;
            obj.audio = audio;
            obj.vad   = vad_obj;
            obj.mwf   = mwf_obj;
            obj.mixer = SourceMixer(cfg, geo);

            obj.drone_gain       = cfg.drone_gain_init;
            obj.env_gain         = cfg.env_gain_init;
            obj.vad_on           = false;
            obj.mwf_on           = false;
            obj.spec_ncols       = cfg.ui.spec_ncols;

            N   = cfg.frame_size;
            nb  = N/2 + 1;
            obj.spec_buf_noisy = zeros(nb, obj.spec_ncols);
            obj.spec_buf_mwf   = zeros(nb, obj.spec_ncols);
            obj.vad_trace      = zeros(1, vad_obj.hist_len);
            obj.vad_flags      = false(1, vad_obj.hist_len);

            obj.build_figure_();

            period = max(0.05, cfg.frame_size / cfg.fs);
            obj.timerObj = timer('ExecutionMode','fixedRate', ...
                'Period',   period, ...
                'BusyMode', 'drop', ...
                'TimerFcn', @(~,~) obj.rt_loop_(), ...
                'StopFcn',  @(~,~) obj.safe_release_());

            obj.fig.DeleteFcn = @(~,~) obj.on_delete_();
        end

        function start(obj)
            fprintf('[Q-WiSE] Ready — press Start to begin.\n');
            waitfor(obj.fig);
        end
    end

    % ==================================================================
    %  BUILD FIGURE
    % ==================================================================
    methods (Access = private)
        function build_figure_(obj)
            cfg = obj.Cfg;
            obj.fig = figure( ...
                'Name', 'Q-WiSE Simulation', ...
                'Color', [0.10 0.10 0.10], ...
                'Position', cfg.ui.fig_position, ...
                'MenuBar', 'none', 'ToolBar', 'figure', ...
                'NumberTitle', 'off');

            cpw = 0.152;

            obj.build_control_panel_(cpw);
            obj.build_axes_(cpw);
        end

        function build_control_panel_(obj, cpw)
            cfg = obj.Cfg;
            ctrl = uipanel(obj.fig, ...
                'BackgroundColor', [0.14 0.14 0.14], ...
                'BorderType', 'none', ...
                'Units', 'normalized', ...
                'Position', [0 0 cpw 1]);

            uicontrol(ctrl, 'Style', 'text', 'String', 'Q-WiSE', ...
                'FontSize', 13, 'FontWeight', 'bold', ...
                'BackgroundColor', [0.14 0.14 0.14], ...
                'ForegroundColor', [0.95 0.88 0.55], ...
                'Units', 'normalized', 'Position', [0.03 0.962 0.94 0.034]);

            % --- Start / Stop capture ---
            obj.H.btn_start = uicontrol(ctrl, 'Style', 'togglebutton', ...
                'String', 'Start Capture', ...
                'FontSize', 10, 'FontWeight', 'bold', ...
                'BackgroundColor', [0.16 0.60 0.25], 'ForegroundColor', 'w', ...
                'Units', 'normalized', 'Position', [0.05 0.910 0.90 0.046], ...
                'Callback', @(src,~) obj.cb_start_(src));

            obj.gui_sep_(ctrl, 0.903);

            % --- Speech source (Mic vs WAV loop) ---
            obj.H.btn_source = uicontrol(ctrl, 'Style', 'togglebutton', ...
                'String', 'Source: Mic', ...
                'FontSize', 9, ...
                'BackgroundColor', [0.20 0.40 0.62], ...
                'ForegroundColor', 'w', ...
                'Units', 'normalized', 'Position', [0.05 0.856 0.58 0.040], ...
                'Callback', @(src,~) obj.cb_source_toggle_(src));
            obj.H.btn_browse = uicontrol(ctrl, 'Style', 'pushbutton', ...
                'String', 'WAV...', ...
                'FontSize', 9, ...
                'BackgroundColor', [0.30 0.30 0.30], ...
                'ForegroundColor', 'w', ...
                'Units', 'normalized', 'Position', [0.65 0.856 0.30 0.040], ...
                'Callback', @(~,~) obj.cb_browse_speech_wav_());
            obj.H.lbl_wav = uicontrol(ctrl, 'Style', 'text', ...
                'String', 'No WAV loaded', ...
                'FontSize', 8, ...
                'BackgroundColor', [0.14 0.14 0.14], ...
                'ForegroundColor', [0.55 0.55 0.55], ...
                'Units', 'normalized', 'Position', [0.05 0.828 0.90 0.022], ...
                'HorizontalAlignment', 'left');

            obj.gui_sep_(ctrl, 0.820);

            % --- Drone noise ---
            obj.H.btn_drone = uicontrol(ctrl, 'Style', 'togglebutton', ...
                'String', 'Drone: OFF', ...
                'FontSize', 9, ...
                'BackgroundColor', [0.22 0.22 0.22], ...
                'ForegroundColor', [0.60 0.60 0.60], ...
                'Units', 'normalized', 'Position', [0.05 0.772 0.90 0.040], ...
                'Callback', @(src,~) obj.cb_toggle_(src, 'drone'));
            obj.H.lbl_dg = uicontrol(ctrl, 'Style', 'text', ...
                'String', sprintf('Drone gain  %.2f', obj.drone_gain), ...
                'FontSize', 8, 'BackgroundColor', [0.14 0.14 0.14], ...
                'ForegroundColor', [0.58 0.58 0.58], ...
                'Units', 'normalized', 'Position', [0.05 0.748 0.90 0.020]);
            obj.H.sld_drone = uicontrol(ctrl, 'Style', 'slider', ...
                'Min', 0, 'Max', 1, 'Value', obj.drone_gain, ...
                'Units', 'normalized', 'Position', [0.05 0.726 0.90 0.020], ...
                'Callback', @(src,~) obj.cb_gain_(src, 'drone'));

            obj.gui_sep_(ctrl, 0.718);

            % --- Env noise ---
            obj.H.btn_env = uicontrol(ctrl, 'Style', 'togglebutton', ...
                'String', 'Env: OFF', ...
                'FontSize', 9, ...
                'BackgroundColor', [0.22 0.22 0.22], ...
                'ForegroundColor', [0.60 0.60 0.60], ...
                'Units', 'normalized', 'Position', [0.05 0.670 0.90 0.040], ...
                'Callback', @(src,~) obj.cb_toggle_(src, 'env'));
            obj.H.lbl_eg = uicontrol(ctrl, 'Style', 'text', ...
                'String', sprintf('Env gain  %.2f', obj.env_gain), ...
                'FontSize', 8, 'BackgroundColor', [0.14 0.14 0.14], ...
                'ForegroundColor', [0.58 0.58 0.58], ...
                'Units', 'normalized', 'Position', [0.05 0.646 0.90 0.020]);
            obj.H.sld_env = uicontrol(ctrl, 'Style', 'slider', ...
                'Min', 0, 'Max', 1, 'Value', obj.env_gain, ...
                'Units', 'normalized', 'Position', [0.05 0.624 0.90 0.020], ...
                'Callback', @(src,~) obj.cb_gain_(src, 'env'));

            obj.gui_sep_(ctrl, 0.616);

            % --- Processing toggles (pipeline stages) ---
            uicontrol(ctrl, 'Style', 'text', 'String', 'Processing', ...
                'FontSize', 9, 'FontWeight', 'bold', ...
                'BackgroundColor', [0.14 0.14 0.14], ...
                'ForegroundColor', [0.85 0.85 0.85], ...
                'Units', 'normalized', 'Position', [0.05 0.586 0.90 0.024]);

            obj.H.btn_vad_enable = uicontrol(ctrl, 'Style', 'togglebutton', ...
                'Value', double(obj.vad_on), ...
                'String', 'Enable VAD: OFF', ...
                'FontSize', 9, ...
                'BackgroundColor', [0.22 0.22 0.22], ...
                'ForegroundColor', [0.60 0.60 0.60], ...
                'Units', 'normalized', 'Position', [0.05 0.540 0.90 0.040], ...
                'Callback', @(src,~) obj.cb_vad_enable_(src));

            obj.H.btn_mwf_enable = uicontrol(ctrl, 'Style', 'togglebutton', ...
                'Value', double(obj.mwf_on), ...
                'String', 'MWF: OFF', ...
                'FontSize', 9, ...
                'BackgroundColor', [0.22 0.22 0.22], ...
                'ForegroundColor', [0.60 0.60 0.60], ...
                'Units', 'normalized', 'Position', [0.05 0.494 0.46 0.040], ...
                'Callback', @(src,~) obj.cb_mwf_enable_(src));

            % MWF beamformer algorithm selector (live-switchable).
            [algo_disp, algo_keys] = obj.mwf_method_choices_();
            sel = find(strcmpi(algo_keys, lower(cfg.mwf.method)), 1);
            if isempty(sel), sel = 1; end
            obj.H.pop_mwf_method = uicontrol(ctrl, 'Style', 'popupmenu', ...
                'String', algo_disp, ...
                'Value', sel, ...
                'FontSize', 8, ...
                'BackgroundColor', [0.20 0.20 0.20], ...
                'ForegroundColor', 'w', ...
                'TooltipString', 'MWF beamformer algorithm', ...
                'Units', 'normalized', 'Position', [0.53 0.496 0.42 0.038], ...
                'Callback', @(src,~) obj.cb_mwf_method_(src));

            obj.gui_sep_(ctrl, 0.486);

            % --- Recording ---
            obj.H.btn_record = uicontrol(ctrl, 'Style', 'togglebutton', ...
                'String', 'Record: OFF', ...
                'FontSize', 9, 'FontWeight', 'bold', ...
                'BackgroundColor', [0.22 0.22 0.22], ...
                'ForegroundColor', [0.60 0.60 0.60], ...
                'Units', 'normalized', 'Position', [0.05 0.444 0.90 0.040], ...
                'Callback', @(src,~) obj.cb_record_(src));

            obj.gui_sep_(ctrl, 0.436);

            % --- Scene (live-editable geometry) ---
            uicontrol(ctrl, 'Style', 'text', 'String', 'Scene', ...
                'FontSize', 9, 'FontWeight', 'bold', ...
                'BackgroundColor', [0.14 0.14 0.14], ...
                'ForegroundColor', [0.85 0.85 0.85], ...
                'Units', 'normalized', 'Position', [0.05 0.408 0.90 0.024]);

            obj.build_scene_slider_(ctrl, 'human_h', ...
                'Human height', cfg.human_height*100, 'cm', ...
                100, 220, 0.382, ...
                @(v) obj.cb_scene_change_('human_h', v));

            obj.build_scene_slider_(ctrl, 'slant', ...
                'Slant distance', cfg.slant_dist, 'm', ...
                0.5, 10.0, 0.338, ...
                @(v) obj.cb_scene_change_('slant', v));

            obj.build_scene_slider_(ctrl, 'drone_az', ...
                'Drone azimuth', cfg.drone.azimuth_deg, 'deg', ...
                -180, 180, 0.294, ...
                @(v) obj.cb_scene_change_('drone_az', v));

            obj.build_scene_slider_(ctrl, 'env_dist', ...
                'Env distance', cfg.env.distance_from_mouth, 'm', ...
                1.0, 30.0, 0.250, ...
                @(v) obj.cb_scene_change_('env_dist', v));

            obj.gui_sep_(ctrl, 0.244);

            % --- Compact status / live readouts ---
            obj.H.lbl_status = uicontrol(ctrl, 'Style', 'text', ...
                'String', obj.status_text_('Idle', false), ...
                'FontSize', 8, 'BackgroundColor', [0.14 0.14 0.14], ...
                'ForegroundColor', [0.55 0.78 0.55], ...
                'Units', 'normalized', 'Position', [0.03 0.176 0.94 0.064], ...
                'HorizontalAlignment', 'left');

            obj.gui_sep_(ctrl, 0.170);

            % --- Latest recording playback (compact grid) ---
            uicontrol(ctrl, 'Style', 'text', 'String', 'Latest Recording', ...
                'FontSize', 9, 'FontWeight', 'bold', ...
                'BackgroundColor', [0.14 0.14 0.14], ...
                'ForegroundColor', [0.85 0.85 0.85], ...
                'Units', 'normalized', ...
                'Position', [0.05 0.142 0.55 0.022]);
            obj.H.btn_refresh = uicontrol(ctrl, 'Style', 'pushbutton', ...
                'String', 'Refresh', ...
                'FontSize', 8, ...
                'BackgroundColor', [0.30 0.30 0.30], ...
                'ForegroundColor', 'w', ...
                'Units', 'normalized', ...
                'Position', [0.62 0.142 0.33 0.024], ...
                'Callback', @(~,~) obj.refresh_playback_());
            obj.playback_panel_ = uipanel(ctrl, ...
                'BackgroundColor', [0.10 0.10 0.10], ...
                'BorderType', 'line', ...
                'HighlightColor', [0.25 0.25 0.25], ...
                'Units', 'normalized', ...
                'Position', [0.03 0.005 0.94 0.133]);
            obj.refresh_playback_();
        end

        function build_scene_slider_(obj, parent, key, label, val, units, ...
                                     lo, hi, y_top, cb)
        %BUILD_SCENE_SLIDER_  Two-row scene control: 'Label  value units' on
        %   the top line, slider underneath. `key` is the stem of the
        %   handle name (obj.H.lbl_<key>, obj.H.sld_<key>).
            lbl = uicontrol(parent, 'Style', 'text', ...
                'String', obj.fmt_scene_(label, val, units), ...
                'FontSize', 8, 'BackgroundColor', [0.14 0.14 0.14], ...
                'ForegroundColor', [0.60 0.85 0.60], ...
                'Units', 'normalized', ...
                'Position', [0.05 y_top + 0.022 0.90 0.018], ...
                'HorizontalAlignment', 'left');
            sld = uicontrol(parent, 'Style', 'slider', ...
                'Min', lo, 'Max', hi, 'Value', max(lo, min(hi, val)), ...
                'Units', 'normalized', ...
                'Position', [0.05 y_top 0.90 0.020], ...
                'Callback', @(src,~) cb(src.Value));
            obj.H.(['lbl_' key])     = lbl;
            obj.H.(['sld_' key])     = sld;
            obj.H.(['units_' key])   = units;
            obj.H.(['label_' key])   = label;
        end

        function s = fmt_scene_(~, label, val, units)
            if strcmp(units, 'cm') || strcmp(units, 'deg')
                s = sprintf('%-14s %6.0f %s', label, val, units);
            else
                s = sprintf('%-14s %6.2f %s', label, val, units);
            end
        end

        function s = status_text_(obj, headline, capturing)
        %STATUS_TEXT_  4-line status block — capture state, geometry
        %   summary, ref-mic gains, VAD backend. Kept short so the
        %   status panel doesn't crowd the playback grid below it.
            cfg = obj.Cfg;
            if capturing
                src_lbl = upper(obj.speech_source);
                line2 = sprintf('%s  ch %d  fs %d  N %d', ...
                                src_lbl, obj.audio.n_hw_ch, ...
                                cfg.fs, cfg.frame_size);
            else
                line2 = sprintf('fs %d  N %d  mics %d (%s)', ...
                                cfg.fs, cfg.frame_size, ...
                                cfg.n_mics, cfg.mic_geometry);
            end
            ref = cfg.mwf.ref_mic;
            s = sprintf([ ...
                '%s\n' ...
                '%s\n' ...
                'AGL %.2f m  slant %.2f m  spc %.0f cm\n' ...
                'g[s,d,e]@m%d=%.2f/%.2f/%.2f  VAD %s'], ...
                headline, line2, ...
                obj.geo.drone_agl, ...
                norm(obj.geo.pos_drone - obj.geo.pos_human), ...
                cfg.mic_spacing*100, ...
                ref, obj.geo.gains_speech(ref), ...
                obj.geo.gains_drone(ref), obj.geo.gains_env(ref), ...
                obj.vad.backend_name);
        end

        function build_axes_(obj, cpw)
            cfg = obj.Cfg;
            rx = cpw + 0.008;
            rw = 1 - rx - 0.010;

            % -------- Row 1 : 3D scene + noisy spec + MWF spec --------
            row1_y = 0.555;  row1_h = 0.420;
            scene_w = rw * 0.34;  spec_w  = rw * 0.31;  gap = rw * 0.010;

            obj.H.ax_scene = axes(obj.fig, 'Units', 'normalized', ...
                'Position', [rx row1_y scene_w row1_h], ...
                'Color', [0.07 0.07 0.07], ...
                'XColor', [0.5 0.5 0.5], 'YColor', [0.5 0.5 0.5], ...
                'ZColor', [0.5 0.5 0.5], 'GridColor', [0.22 0.22 0.22]);

            specN_x = rx + scene_w + gap;
            obj.H.ax_spec_noisy = axes(obj.fig, 'Units', 'normalized', ...
                'Position', [specN_x row1_y spec_w row1_h], ...
                'Color', [0.07 0.07 0.07], ...
                'XColor', [0.5 0.5 0.5], 'YColor', [0.5 0.5 0.5], ...
                'GridColor', [0.22 0.22 0.22]);

            specM_x = specN_x + spec_w + gap;
            specM_w = rx + rw - specM_x;
            obj.H.ax_spec_mwf = axes(obj.fig, 'Units', 'normalized', ...
                'Position', [specM_x row1_y specM_w row1_h], ...
                'Color', [0.07 0.07 0.07], ...
                'XColor', [0.5 0.5 0.5], 'YColor', [0.5 0.5 0.5], ...
                'GridColor', [0.22 0.22 0.22]);

            N   = cfg.frame_size;
            nb  = N/2 + 1;
            frq = linspace(0, cfg.fs/2000, nb);

            obj.H.himg_noisy = imagesc(obj.H.ax_spec_noisy, 1:obj.spec_ncols, frq, ...
                                       20*log10(obj.spec_buf_noisy + 1e-12));
            colormap(obj.H.ax_spec_noisy, 'hot');
            clim(obj.H.ax_spec_noisy, [-80 -20]);
            axis(obj.H.ax_spec_noisy, 'xy');
            ylabel(obj.H.ax_spec_noisy, 'Freq (kHz)', 'Color', [0.5 0.5 0.5]);
            title(obj.H.ax_spec_noisy, ...
                  'Spectrogram — Noisy Composite (Speech + Drone + Env)', ...
                  'Color', 'w', 'FontSize', 9, 'FontWeight', 'bold');

            obj.H.himg_mwf = imagesc(obj.H.ax_spec_mwf, 1:obj.spec_ncols, frq, ...
                                     20*log10(obj.spec_buf_mwf + 1e-12));
            colormap(obj.H.ax_spec_mwf, 'hot');
            clim(obj.H.ax_spec_mwf, [-80 -20]);
            axis(obj.H.ax_spec_mwf, 'xy');
            ylabel(obj.H.ax_spec_mwf, 'Freq (kHz)', 'Color', [0.5 0.5 0.5]);
            title(obj.H.ax_spec_mwf, 'Spectrogram — MWF Enhanced', ...
                  'Color', 'w', 'FontSize', 9, 'FontWeight', 'bold');

            % -------- Row 2 : per-mic noisy waveforms --------
            row2_y_top = 0.532;   row2_y_bot = 0.315;
            row2_h    = row2_y_top - row2_y_bot;
            wh        = row2_h / cfg.n_mics;

            mc = {[0.00 0.78 0.32], [0.00 0.55 0.95], [1.00 0.52 0.00], ...
                  [0.95 0.35 0.75], [0.60 0.85 0.20]};
            ref_mic = obj.mixer.ref_mic();
            obj.H.ax_wav = gobjects(cfg.n_mics, 1);
            obj.H.hlines = gobjects(cfg.n_mics, 1);
            for m = 1:cfg.n_mics
                idx = mod(m-1, numel(mc)) + 1;
                yp  = row2_y_bot + (cfg.n_mics - m) * wh;
                obj.H.ax_wav(m) = axes(obj.fig, 'Units', 'normalized', ...
                    'Position', [rx yp rw wh - 0.010], ...
                    'Color', [0.06 0.06 0.06], ...
                    'XColor', [0.45 0.45 0.45], 'YColor', mc{idx}, ...
                    'GridColor', [0.20 0.20 0.20]);
                hold(obj.H.ax_wav(m), 'on');
                grid(obj.H.ax_wav(m), 'on');
                ylim(obj.H.ax_wav(m), [-1.05 1.05]);
                xlim(obj.H.ax_wav(m), [1 N]);
                ylabel(obj.H.ax_wav(m), mic_label_(m, ref_mic), ...
                       'Color', mc{idx}, 'FontSize', 9);
                if m == ref_mic
                    % Green-tint background patch lights up while the VAD
                    % is detecting speech (refresh_display_ toggles its
                    % FaceAlpha each tick). Created BEFORE the line plot
                    % so the waveform draws on top. The tint sits on the
                    % reference-mic row — that channel is what the VAD
                    % actually sees.
                    obj.H.h_mic1_speech = patch(obj.H.ax_wav(m), ...
                        'XData', [1 N N 1], ...
                        'YData', [-1.05 -1.05 1.05 1.05], ...
                        'FaceColor', [0.30 0.80 0.40], ...
                        'FaceAlpha', 0.0, ...
                        'EdgeColor', 'none', ...
                        'HitTest', 'off');
                end
                obj.H.hlines(m) = plot(obj.H.ax_wav(m), 1:N, zeros(N, 1), ...
                    'Color', mc{idx}, 'LineWidth', 0.75);
                if m == 1
                    ttl = sprintf(['Physical %d-Mic Array  (every mic receives ' ...
                                   'speech+drone+env with TDOA + 1/r gain ·  ' ...
                                   'green tint on Mic %d = VAD speech)'], ...
                                  cfg.n_mics, ref_mic);
                    title(obj.H.ax_wav(m), ttl, ...
                        'Color', 'w', 'FontSize', 9, 'FontWeight', 'bold');
                end
            end

            % -------- Row 3 : MWF output waveform --------
            row3_y = 0.190;  row3_h = 0.105;
            obj.H.ax_mwf_wave = axes(obj.fig, 'Units', 'normalized', ...
                'Position', [rx row3_y rw row3_h], ...
                'Color', [0.06 0.06 0.06], ...
                'XColor', [0.45 0.45 0.45], 'YColor', [0.20 0.85 0.90], ...
                'GridColor', [0.20 0.20 0.20]);
            hold(obj.H.ax_mwf_wave, 'on'); grid(obj.H.ax_mwf_wave, 'on');
            ylim(obj.H.ax_mwf_wave, [-1.05 1.05]);
            xlim(obj.H.ax_mwf_wave, [1 N]);
            ylabel(obj.H.ax_mwf_wave, 'MWF out', ...
                'Color', [0.20 0.85 0.90], 'FontSize', 9);
            title(obj.H.ax_mwf_wave, 'MWF Enhanced Output (pass-through stub)', ...
                  'Color', 'w', 'FontSize', 9, 'FontWeight', 'bold');
            obj.H.hline_mwf = plot(obj.H.ax_mwf_wave, 1:N, zeros(N, 1), ...
                'Color', [0.20 0.85 0.90], 'LineWidth', 0.75);

            % -------- Row 4 : VAD trace --------
            row4_y = 0.050;  row4_h = 0.115;
            obj.H.ax_vad = axes(obj.fig, 'Units', 'normalized', ...
                'Position', [rx row4_y rw row4_h], ...
                'Color', [0.06 0.06 0.06], ...
                'XColor', [0.45 0.45 0.45], 'YColor', [0.98 0.75 0.20], ...
                'GridColor', [0.20 0.20 0.20]);
            hold(obj.H.ax_vad, 'on'); grid(obj.H.ax_vad, 'on');
            ylim(obj.H.ax_vad, [-0.05 1.10]);
            xlim(obj.H.ax_vad, [1 obj.vad.hist_len]);
            ylabel(obj.H.ax_vad, 'VAD', 'Color', [0.98 0.75 0.20], 'FontSize', 9);
            xlabel(obj.H.ax_vad, 'Frames (past → now)', 'Color', [0.50 0.50 0.50]);
            title(obj.H.ax_vad, ...
                  sprintf('VAD Trace (%s backend) — shaded = detected speech', ...
                          obj.vad.backend_name), ...
                  'Color', 'w', 'FontSize', 9, 'FontWeight', 'bold');
            obj.H.h_vad_shade = patch(obj.H.ax_vad, ...
                'XData', [1 1 1 1], 'YData', [0 0 0 0], ...
                'FaceColor', [0.30 0.80 0.40], 'FaceAlpha', 0.22, ...
                'EdgeColor', 'none');
            obj.H.h_vad_line = plot(obj.H.ax_vad, ...
                1:obj.vad.hist_len, zeros(1, obj.vad.hist_len), ...
                'Color', [0.98 0.75 0.20], 'LineWidth', 1.4);
            obj.H.h_vad_thr = plot(obj.H.ax_vad, ...
                [1 obj.vad.hist_len], [0.5 0.5], '--', ...
                'Color', [0.55 0.55 0.55], 'LineWidth', 0.9);

            % Static scene (drawn once)
            draw_scene(obj.H.ax_scene, obj.geo, cfg);
        end

        function gui_sep_(~, parent, y)
            uicontrol(parent, 'Style', 'text', 'String', '', ...
                'BackgroundColor', [0.32 0.32 0.32], ...
                'Units', 'normalized', 'Position', [0.03 y 0.94 0.003]);
        end
    end

    % ==================================================================
    %  REAL-TIME LOOP
    % ==================================================================
    methods (Access = private)
        function rt_loop_(obj)
            if ~isvalid(obj.fig), return; end
            if ~obj.running,        return; end
            cfg = obj.Cfg;
            N   = cfg.frame_size;

            try
                raw = obj.audio.read();
            catch
                return;
            end

            % --- Three physical sources feeding the array ---
            if strcmpi(obj.speech_source, 'wav') && obj.audio.has_speech_wav()
                speech = obj.audio.next_speech_chunk(N);    % clean-speech WAV loop
            else
                speech = raw(:, 1);                         % live input mic
            end
            if obj.drone_on
                drone = obj.drone_gain * obj.audio.next_drone_chunk(N);
            else
                drone = zeros(N, 1);
            end
            if obj.env_on
                envn  = obj.env_gain * obj.audio.next_env_chunk(N);
            else
                envn  = zeros(N, 1);
            end

            mic = obj.mixer.mix(speech, drone, envn);

            % --- Composite feed for the VAD (reference mic of the N-mic block) ---
            comp = obj.mixer.composite(mic);

            % --- VAD stage (only when enabled) ---
            if obj.vad_on
                [is_speech, ~] = obj.vad.step(comp);
            else
                is_speech = false;
            end

            % --- MWF stage (only when enabled; requires VAD per cascade rule) ---
            if obj.mwf_on
                y_enh = obj.mwf.step(mic, is_speech);
            else
                y_enh = zeros(N, 1);
            end

            % --- Recording (mode was locked in at rec_start) ---
            if obj.recording
                obj.write_recording_(mic, comp, y_enh, is_speech);
            end

            obj.update_buffers_(comp, y_enh);
            obj.refresh_display_(mic, y_enh, is_speech);
        end

        function update_buffers_(obj, comp, y_enh)
            N   = obj.Cfg.frame_size;
            nb  = N/2 + 1;
            w   = hann(N, 'periodic');
            Xn  = abs(fft(comp  .* w, N));
            Xm  = abs(fft(y_enh .* w, N));
            obj.spec_buf_noisy(:, obj.spec_col) = Xn(1:nb);
            obj.spec_buf_mwf  (:, obj.spec_col) = Xm(1:nb);
            obj.spec_col = mod(obj.spec_col, obj.spec_ncols) + 1;

            [scores, flags] = obj.vad.trace();
            obj.vad_trace = scores(:).';
            obj.vad_flags = flags(:).';
        end

        function refresh_display_(obj, mic, y_enh, is_speech)
            if ~isvalid(obj.fig), return; end
            cfg = obj.Cfg;
            for m = 1:cfg.n_mics
                pk = max(abs(mic(:, m)));
                if pk > 1e-9
                    set(obj.H.hlines(m), 'YData', mic(:, m) / pk);
                else
                    set(obj.H.hlines(m), 'YData', zeros(size(mic, 1), 1));
                end
            end
            pke = max(abs(y_enh));
            if pke > 1e-9
                set(obj.H.hline_mwf, 'YData', y_enh / pke);
            else
                set(obj.H.hline_mwf, 'YData', zeros(numel(y_enh), 1));
            end

            % Green tint behind the mic-1 waveform whenever VAD is
            % enabled and the current frame was flagged as speech.
            if obj.vad_on && is_speech
                set(obj.H.h_mic1_speech, 'FaceAlpha', 0.18);
            else
                set(obj.H.h_mic1_speech, 'FaceAlpha', 0.0);
            end

            set(obj.H.himg_noisy, 'CData', 20*log10(obj.spec_buf_noisy + 1e-12));
            set(obj.H.himg_mwf,   'CData', 20*log10(obj.spec_buf_mwf   + 1e-12));

            set(obj.H.h_vad_line, 'YData', obj.vad_trace);
            obj.update_vad_shade_();
            drawnow limitrate;
        end

        function update_vad_shade_(obj)
            f = obj.vad_flags;
            if isempty(f)
                set(obj.H.h_vad_shade, 'XData', [1 1 1 1], 'YData', [0 0 0 0]);
                return;
            end
            n  = numel(f);
            df = diff([false, f, false]);
            starts = find(df ==  1);
            stops  = find(df == -1) - 1;
            if isempty(starts)
                set(obj.H.h_vad_shade, 'XData', [1 1 1 1], 'YData', [0 0 0 0]);
                return;
            end
            nseg = numel(starts);
            Xs = zeros(4, nseg);
            Ys = zeros(4, nseg);
            for i = 1:nseg
                s = max(1, starts(i));  e = min(n, stops(i));
                Xs(:, i) = [s; e; e; s];
                Ys(:, i) = [0; 0; 1; 1];
            end
            set(obj.H.h_vad_shade, 'XData', Xs, 'YData', Ys);
        end

        function write_recording_(obj, mic, comp, y_enh, is_speech)
        %WRITE_RECORDING_  Route the current tick into the active
        %   recording session. Which tracks exist was decided at
        %   Record-press time and lives in obj.rec_tracks_:
        %
        %     mic = micNN.wav (continuous, N raw mic channels)
        %     vad = vad.wav   (composite, non-speech frames zeroed —
        %                      VAD-detected speech preserved in place)
        %     mwf = mwf.wav   (MWF output, continuous when MWF is active)
        %
        %   Tracks are mixed-and-matched: if MWF was off at Record-press,
        %   the 'mwf' track simply does not exist.
            if obj.rec_tracks_.mic
                obj.audio.rec_session_write('mic', mic);
            end
            if obj.rec_tracks_.vad
                v = comp;
                if ~is_speech
                    v(:) = 0;
                end
                obj.audio.rec_session_write('vad', v);
            end
            if obj.rec_tracks_.mwf
                obj.audio.rec_session_write('mwf', y_enh);
            end
        end
    end

    % ==================================================================
    %  CALLBACKS
    % ==================================================================
    methods (Access = private)
        function cb_start_(obj, src)
            if isempty(obj.timerObj) || ~isvalid(obj.timerObj)
                set(src, 'Value', 0);
                warndlg('Timer not ready yet. Wait a moment and try again.', 'Q-WiSE');
                return;
            end
            if src.Value == 1
                obj.running = true;
                src.String          = 'Stop Capture';
                src.BackgroundColor = [0.72 0.16 0.16];
                obj.H.lbl_status.String = obj.status_text_('Capturing', true);
                obj.H.lbl_status.ForegroundColor = [0.28 0.88 0.38];
                start(obj.timerObj);
            else
                obj.running = false;
                src.String          = 'Start Capture';
                src.BackgroundColor = [0.16 0.60 0.25];
                obj.H.lbl_status.String = obj.status_text_('Stopped', false);
                obj.H.lbl_status.ForegroundColor = [0.70 0.42 0.18];
                stop(obj.timerObj);
            end
        end

        function cb_source_toggle_(obj, src)
        %CB_SOURCE_TOGGLE_  Switch the speech source between the live mic
        %   and a looped clean-speech WAV. If the user flips to WAV but
        %   no file has been loaded yet, auto-open the file picker; if
        %   they cancel, revert to mic.
            want_wav = logical(src.Value);
            if want_wav
                if ~obj.audio.has_speech_wav()
                    if ~obj.cb_browse_speech_wav_()
                        set(src, 'Value', 0);
                        return;
                    end
                end
                obj.speech_source     = 'wav';
                src.String            = 'Source: WAV';
                src.BackgroundColor   = [0.56 0.28 0.62];
                src.ForegroundColor   = 'w';
            else
                obj.speech_source     = 'mic';
                src.String            = 'Source: Mic';
                src.BackgroundColor   = [0.20 0.40 0.62];
                src.ForegroundColor   = 'w';
            end
            obj.refresh_status_();
        end

        function ok = cb_browse_speech_wav_(obj)
        %CB_BROWSE_SPEECH_WAV_  Open a file picker, load the WAV into
        %   the AudioIO speech loop. Returns true on success.
            ok = false;
            [fn, fp] = uigetfile({'*.wav','WAV files (*.wav)'}, ...
                                 'Choose clean-speech WAV');
            if isequal(fn, 0)
                return;
            end
            full = fullfile(fp, fn);
            try
                obj.audio.load_speech_wav(full);
            catch ME
                warndlg(sprintf('Failed to load WAV:\n%s', ME.message), 'Q-WiSE');
                return;
            end
            obj.H.lbl_wav.String = ['WAV: ' fn];
            ok = true;
        end

        function cb_scene_change_(obj, key, val)
        %CB_SCENE_CHANGE_  React to a scene-slider drag. Update the cfg
        %   field, rebuild geometry, push it into the mixer, refresh the
        %   3-D scene plot, and update the on-screen value/status labels.
            switch key
                case 'human_h'
                    obj.Cfg.human_height = val / 100;
                    obj.Cfg.mouth_height = 0.88 * obj.Cfg.human_height;
                case 'slant'
                    obj.Cfg.slant_dist   = val;
                case 'drone_az'
                    obj.Cfg.drone.azimuth_deg = val;
                case 'env_dist'
                    obj.Cfg.env.distance_from_mouth = val;
                otherwise
                    return;
            end

            obj.geo = build_geometry(obj.Cfg);
            obj.mixer.update_geometry(obj.geo);

            % Refresh the per-slider readout line.
            lbl_h = obj.H.(['lbl_' key]);
            lbl_h.String = obj.fmt_scene_( ...
                obj.H.(['label_' key]), val, obj.H.(['units_' key]));

            % Redraw the 3-D scene from scratch.
            if isfield(obj.H, 'ax_scene') && isvalid(obj.H.ax_scene)
                cla(obj.H.ax_scene);
                draw_scene(obj.H.ax_scene, obj.geo, obj.Cfg);
            end

            obj.refresh_status_();
        end

        function refresh_status_(obj)
        %REFRESH_STATUS_  Recompute the status block from current state.
            if ~isfield(obj.H, 'lbl_status') || ~isvalid(obj.H.lbl_status)
                return;
            end
            if obj.running
                obj.H.lbl_status.String = obj.status_text_('Capturing', true);
                obj.H.lbl_status.ForegroundColor = [0.28 0.88 0.38];
            else
                obj.H.lbl_status.String = obj.status_text_('Idle', false);
                obj.H.lbl_status.ForegroundColor = [0.55 0.78 0.55];
            end
        end

        function cb_toggle_(obj, src, which)
            on = logical(src.Value);
            if strcmp(which, 'drone')
                obj.drone_on = on;
                if on
                    src.String = 'Drone: ON';
                    src.BackgroundColor = [0.68 0.32 0.06];
                    src.ForegroundColor = 'w';
                else
                    src.String = 'Drone: OFF';
                    src.BackgroundColor = [0.22 0.22 0.22];
                    src.ForegroundColor = [0.60 0.60 0.60];
                end
            else
                obj.env_on = on;
                if on
                    src.String = 'Env: ON';
                    src.BackgroundColor = [0.36 0.06 0.58];
                    src.ForegroundColor = 'w';
                else
                    src.String = 'Env: OFF';
                    src.BackgroundColor = [0.22 0.22 0.22];
                    src.ForegroundColor = [0.60 0.60 0.60];
                end
            end
        end

        function cb_gain_(obj, src, which)
            if strcmp(which, 'drone')
                obj.drone_gain = src.Value;
                obj.H.lbl_dg.String = sprintf('Drone gain  %.2f', src.Value);
            else
                obj.env_gain = src.Value;
                obj.H.lbl_eg.String = sprintf('Env gain  %.2f', src.Value);
            end
        end

        function cb_vad_enable_(obj, src)
        %CB_VAD_ENABLE_  Toggle the VAD stage. If MWF is currently on and
        %   the user disables VAD, MWF is auto-disabled too — MWF cannot
        %   produce a meaningful output without VAD-driven covariance
        %   updates.
            on = logical(src.Value);
            obj.vad_on = on;
            obj.set_button_state_(src, on, 'Enable VAD', [0.15 0.60 0.30]);
            if ~on && obj.mwf_on
                obj.mwf_on = false;
                obj.set_button_state_(obj.H.btn_mwf_enable, false, ...
                    'MWF', [0.15 0.45 0.70]);
                obj.set_status_warning_( ...
                    'VAD off — MWF auto-disabled (MWF needs VAD).');
                fprintf(['[Q-WiSE] VAD disabled while MWF was on; ' ...
                         'MWF auto-disabled (MWF needs VAD).\n']);
            end
        end

        function cb_mwf_enable_(obj, src)
        %CB_MWF_ENABLE_  Toggle the MWF stage. If VAD is off when the
        %   user enables MWF, VAD is auto-enabled — MWF cannot produce a
        %   meaningful output without VAD-driven covariance updates.
            on = logical(src.Value);
            if on && ~obj.vad_on
                obj.vad_on = true;
                obj.set_button_state_(obj.H.btn_vad_enable, true, ...
                    'Enable VAD', [0.15 0.60 0.30]);
                obj.set_status_warning_( ...
                    'MWF needs VAD — VAD auto-enabled.');
                fprintf(['[Q-WiSE] MWF enabled without VAD; ' ...
                         'VAD auto-enabled (MWF needs VAD).\n']);
            end
            obj.mwf_on = on;
            obj.set_button_state_(src, on, 'MWF', [0.15 0.45 0.70]);
        end

        function [disp_names, keys] = mwf_method_choices_(~)
        %MWF_METHOD_CHOICES_  Display labels and canonical cfg.mwf.method
        %   keys for the algorithm selector, kept in matching order.
            disp_names = {'GEV (Max-SNR)', 'SDW-MWF', 'MVDR', ...
                          'Rank-1 (2/3-mic)', 'Rank-N + OMLSA'};
            keys       = {'gev', 'mwf', 'mvdr', 'rank1', 'rankn'};
        end

        function cb_mwf_method_(obj, src)
        %CB_MWF_METHOD_  Switch the live MWF beamformer algorithm. Takes
        %   effect on the next processed block; existing methods are
        %   unchanged, 'rank1' is the new closed-form 2-/3-mic MWF.
            [~, keys] = obj.mwf_method_choices_();
            key = keys{src.Value};
            obj.mwf.method  = key;          % mwf reads obj.method per block
            obj.Cfg.mwf.method = key;       % keep config in sync
            % Rank-N ships with the OMLSA near-zero denoiser enabled; other
            % methods follow whatever the config requested.
            if strcmp(key, 'rankn')
                obj.mwf.post_omlsa = true;
            elseif isfield(obj.Cfg, 'mwf') && isfield(obj.Cfg.mwf, 'post_omlsa')
                obj.mwf.post_omlsa = logical(obj.Cfg.mwf.post_omlsa);
            end
            omlsa_lbl = 'off';
            if obj.mwf.post_omlsa, omlsa_lbl = 'on'; end
            obj.set_status_warning_(sprintf('MWF algorithm: %s (OMLSA %s)', ...
                                            upper(key), omlsa_lbl));
            fprintf('[Q-WiSE] MWF algorithm set to "%s" (OMLSA %s).\n', key, omlsa_lbl);
        end

        function set_button_state_(~, btn, on, base_label, on_color)
        %SET_BUTTON_STATE_  Apply consistent ON/OFF visual state to a
        %   togglebutton, including the base_label suffix and colour scheme.
            btn.Value = double(on);
            if on
                btn.String          = [base_label ': ON'];
                btn.BackgroundColor = on_color;
                btn.ForegroundColor = 'w';
            else
                btn.String          = [base_label ': OFF'];
                btn.BackgroundColor = [0.22 0.22 0.22];
                btn.ForegroundColor = [0.60 0.60 0.60];
            end
        end

        function set_status_warning_(obj, msg)
        %SET_STATUS_WARNING_  Surface a one-line warning in the status
        %   label using a warm tint. Persists until the next status
        %   update (e.g. start/stop capture).
            if isfield(obj.H, 'lbl_status') && isvalid(obj.H.lbl_status)
                obj.H.lbl_status.String          = msg;
                obj.H.lbl_status.ForegroundColor = [0.95 0.65 0.20];
            end
        end

        function cb_record_(obj, src)
        %CB_RECORD_  Start/stop a multi-track recording session. The set
        %   of tracks is decided here, ONCE, from the current toggles:
        %
        %     VAD on + MWF on  → mic + vad + mwf
        %     VAD on (no MWF)  → mic + vad
        %     both off         → mic only
        %
        %   Every recording produces a timestamped folder under
        %   cfg.record.dir with one WAV per active track (mic uses N
        %   files: mic01.wav, mic02.wav, ...). Toggling VAD/MWF mid-
        %   recording does NOT change the track set — stop and re-start
        %   to reconfigure.
            want_on = logical(src.Value);
            if want_on && ~obj.recording
                tracks = struct( ...
                    'mic', true, ...
                    'vad', obj.vad_on, ...
                    'mwf', obj.vad_on && obj.mwf_on);
                path = obj.audio.rec_start_session();
                if isempty(path)
                    set(src, 'Value', 0);
                    warndlg('Could not open recording session.', 'Q-WiSE');
                    return;
                end
                obj.rec_tracks_ = tracks;
                obj.rec_kind_   = obj.tracks_label_(tracks);
                obj.recording   = true;
                src.String          = sprintf('Record: ● REC (%s)', obj.rec_kind_);
                src.BackgroundColor = [0.78 0.10 0.10];
                src.ForegroundColor = 'w';
                fprintf('[Q-WiSE] Recording tracks: %s -> %s\n', ...
                        obj.rec_kind_, path);
            elseif ~want_on && obj.recording
                path = obj.audio.rec_stop_session();
                obj.recording   = false;
                obj.rec_kind_   = '';
                obj.rec_tracks_ = struct('mic', false, 'vad', false, 'mwf', false);
                src.String          = 'Record: OFF';
                src.BackgroundColor = [0.22 0.22 0.22];
                src.ForegroundColor = [0.60 0.60 0.60];
                if ~isempty(path)
                    fprintf('[Q-WiSE] Recording saved to %s\n', path);
                end
                % Surface the new files in the playback panel.
                obj.refresh_playback_();
            end
        end

        function s = tracks_label_(~, tracks)
        %TRACKS_LABEL_  Short status string of the active track set.
            on = {};
            if tracks.mic, on{end+1} = 'mic'; end
            if tracks.vad, on{end+1} = 'vad'; end
            if tracks.mwf, on{end+1} = 'mwf'; end
            s = strjoin(on, '+');
            if isempty(s), s = 'none'; end
        end

        % ==================================================================
        %  PLAYBACK OF LATEST RECORDING
        % ==================================================================
        function refresh_playback_(obj)
        %REFRESH_PLAYBACK_  Scan cfg.record.dir for the most-recent
        %   recording folder and render one play/stop togglebutton per
        %   WAV inside it. Called on UI construction, after a recording
        %   stops, and from the Refresh button.
            obj.stop_playback_();

            if isempty(obj.playback_panel_) || ~isvalid(obj.playback_panel_)
                return;
            end

            rec_dir = obj.Cfg.record.dir;
            obj.playback_files_ = {};
            obj.playback_dir_   = '';
            delete(obj.playback_panel_.Children);

            if ~isfolder(rec_dir)
                obj.render_playback_empty_('No recordings yet.');
                return;
            end

            % Pick the most recently-modified subfolder. Sessions are
            % always folders ("qwise_multi_YYYYMMDD_HHMMSS") so that's
            % the only kind we list.
            d = dir(rec_dir);
            d = d([d.isdir] & ~startsWith({d.name}, '.'));
            if isempty(d)
                obj.render_playback_empty_('No recording folders.');
                return;
            end
            [~, ord] = sort([d.datenum], 'descend');
            latest = fullfile(rec_dir, d(ord(1)).name);

            wavs = dir(fullfile(latest, '*.wav'));
            if isempty(wavs)
                [~, leaf] = fileparts(latest);
                obj.render_playback_empty_(sprintf('(empty) %s', leaf));
                return;
            end
            [~, ord_w] = sort({wavs.name});
            wavs = wavs(ord_w);

            obj.playback_dir_   = latest;
            obj.playback_files_ = arrayfun( ...
                @(w) fullfile(w.folder, w.name), wavs, 'UniformOutput', false);
            obj.render_playback_files_({wavs.name});
        end

        function render_playback_files_(obj, names)
        %RENDER_PLAYBACK_FILES_  Lay out the WAVs as a compact grid of
        %   small Play/Stop togglebuttons. Labels are shortened
        %   (mic01.wav → M1, mwf.wav → MWF, ...) so each cell stays
        %   readable; hover shows the full filename via tooltip.
            [~, leaf] = fileparts(obj.playback_dir_);
            uicontrol(obj.playback_panel_, 'Style', 'text', ...
                'String', leaf, ...
                'FontSize', 7, 'BackgroundColor', [0.10 0.10 0.10], ...
                'ForegroundColor', [0.55 0.55 0.55], ...
                'Units', 'normalized', ...
                'Position', [0.02 0.86 0.96 0.12], ...
                'HorizontalAlignment', 'left');

            n      = numel(names);
            n_cols = min(3, max(1, n));
            n_rows = max(1, ceil(n / n_cols));

            % Grid area inside the panel (leave room for the folder
            % header at the top and a small bottom margin).
            grid_x0 = 0.02; grid_x1 = 0.98;
            grid_y0 = 0.04; grid_y1 = 0.82;
            cell_w  = (grid_x1 - grid_x0) / n_cols;
            cell_h  = (grid_y1 - grid_y0) / n_rows;
            pad_x   = cell_w * 0.06;
            pad_y   = cell_h * 0.10;

            for k = 1:n
                col = mod(k - 1, n_cols);
                row = floor((k - 1) / n_cols);
                xp  = grid_x0 + col * cell_w + pad_x;
                yp  = grid_y1 - (row + 1) * cell_h + pad_y;
                bw  = cell_w - 2 * pad_x;
                bh  = cell_h - 2 * pad_y;

                short = obj.short_track_label_(names{k});
                uicontrol(obj.playback_panel_, 'Style', 'togglebutton', ...
                    'String',     sprintf('▶ %s', short), ...
                    'FontSize',   8, ...
                    'BackgroundColor', [0.18 0.40 0.22], ...
                    'ForegroundColor', 'w', ...
                    'Units',      'normalized', ...
                    'Position',   [xp yp bw bh], ...
                    'TooltipString', names{k}, ...
                    'UserData',   struct('index', k, ...
                                         'name',  names{k}, ...
                                         'short', short), ...
                    'Callback',   @(s,~) obj.cb_play_toggle_(s));
            end
        end

        function s = short_track_label_(~, name)
        %SHORT_TRACK_LABEL_  micNN.wav → MN, mwf/vad → MWF/VAD.
            [~, stem] = fileparts(name);
            m = regexp(stem, '^mic0*(\d+)$', 'tokens', 'once');
            if ~isempty(m)
                s = sprintf('M%s', m{1});
            else
                s = upper(stem);
            end
        end

        function render_playback_empty_(obj, msg)
            uicontrol(obj.playback_panel_, 'Style', 'text', ...
                'String', msg, ...
                'FontSize', 8, ...
                'BackgroundColor', [0.10 0.10 0.10], ...
                'ForegroundColor', [0.55 0.55 0.55], ...
                'Units', 'normalized', ...
                'Position', [0.02 0.35 0.96 0.30], ...
                'HorizontalAlignment', 'center');
        end

        function cb_play_toggle_(obj, src)
        %CB_PLAY_TOGGLE_  Start playback when toggled ON, stop when OFF.
        %   Only one file plays at a time — toggling another row stops
        %   the previous one.
            ud = src.UserData;
            if logical(src.Value)
                obj.stop_playback_();   % halt anything currently playing
                try
                    [y, fs] = audioread(obj.playback_files_{ud.index});
                    if size(y, 2) > 1
                        y = mean(y, 2);
                    end
                    p = audioplayer(y, fs);
                    p.StopFcn = @(~,~) obj.cb_player_stopped_(src);
                    play(p);
                    obj.playback_player_     = p;
                    obj.playback_active_btn_ = src;
                    src.String          = sprintf('■ %s', ud.short);
                    src.BackgroundColor = [0.78 0.10 0.10];
                catch ME
                    set(src, 'Value', 0);
                    warndlg(sprintf('Could not play %s:\n%s', ud.name, ...
                                    ME.message), 'Q-WiSE');
                end
            else
                obj.stop_playback_();
            end
        end

        function cb_player_stopped_(obj, src)
        %CB_PLAYER_STOPPED_  audioplayer StopFcn — reached EOF or was
        %   stopped explicitly. Reset the button visual.
            if isvalid(src)
                ud = src.UserData;
                src.Value           = 0;
                src.String          = sprintf('▶ %s', ud.short);
                src.BackgroundColor = [0.18 0.40 0.22];
            end
            obj.playback_player_     = [];
            obj.playback_active_btn_ = [];
        end

        function stop_playback_(obj)
        %STOP_PLAYBACK_  Stop any active playback and reset its button.
            if ~isempty(obj.playback_player_)
                try
                    if isvalid(obj.playback_player_) && ...
                       isplaying(obj.playback_player_)
                        stop(obj.playback_player_);
                    end
                catch
                end
            end
            if ~isempty(obj.playback_active_btn_) && ...
                    isvalid(obj.playback_active_btn_)
                ud = obj.playback_active_btn_.UserData;
                obj.playback_active_btn_.Value           = 0;
                obj.playback_active_btn_.String          = sprintf('▶ %s', ud.short);
                obj.playback_active_btn_.BackgroundColor = [0.18 0.40 0.22];
            end
            obj.playback_player_     = [];
            obj.playback_active_btn_ = [];
        end

        function safe_release_(obj)
            try obj.audio.release(); catch, end
        end

        function on_delete_(obj)
            try obj.stop_playback_(); catch, end
            try
                if obj.recording
                    obj.audio.rec_stop();
                    obj.recording = false;
                end
            catch
            end
            try
                if ~isempty(obj.timerObj) && isvalid(obj.timerObj)
                    stop(obj.timerObj);
                    delete(obj.timerObj);
                end
            catch
            end
            try obj.audio.release(); catch, end
        end
    end
end

% ---------------------------------------------------------------------
function s = mic_label_(m, ref_mic)
%MIC_LABEL_  Friendly ylabel for each mic row. Every mic now receives
%   all three sources (speech + drone + env) with its own TDOA + 1/r
%   gain, so we just number them; the reference channel (VAD feed) gets
%   a small marker.
    if m == ref_mic
        s = sprintf('Mic %d  (ref)', m);
    else
        s = sprintf('Mic %d', m);
    end
end
