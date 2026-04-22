classdef SimulatorUI < handle
%SIMULATORUI  Q-WiSE real-time visualization + control panel.
%
%   ui = SimulatorUI(cfg, geo, audio, vad_obj, mwf_obj)
%   ui.start()
%
%   Renders a single figure with:
%     * Control panel  (Start/Stop, drone toggle+gain, env toggle+gain,
%                       playback source switch, status)
%     * 3D scene       (speaker, drone, mic ULA, ground reflection)
%     * Noisy spectrogram + MWF-enhanced spectrogram  (side by side)
%     * Per-mic live waveforms (n_mics rows)
%     * MWF output waveform
%     * VAD score trace with detection shading
%
%   The real-time timer callback reads a mic frame, expands to the
%   virtual ULA, optionally mixes drone + env noise, runs the VAD, runs
%   the MWF (pass-through for now), plays the selected source to
%   speakers, and refreshes every panel.

    properties
        Cfg
        geo
        audio
        vad
        mwf
        fig
        H = struct()

        running         = false
        drone_on        = false
        env_on          = false
        drone_gain
        env_gain
        playback_source        % 'noisy' | 'enhanced'
    end

    properties (Access = private)
        timerObj
        spec_buf_noisy
        spec_buf_mwf
        spec_col  = 1
        spec_ncols
        vad_trace
        vad_flags
    end

    methods
        function obj = SimulatorUI(cfg, geo, audio, vad_obj, mwf_obj)
            obj.Cfg   = cfg;
            obj.geo   = geo;
            obj.audio = audio;
            obj.vad   = vad_obj;
            obj.mwf   = mwf_obj;

            obj.drone_gain       = cfg.drone_gain_init;
            obj.env_gain         = cfg.env_gain_init;
            obj.playback_source  = cfg.playback.source;
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
                'Name', 'Q-WiSE  |  Acoustic Simulation + VAD + MWF', ...
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
                'Units', 'normalized', 'Position', [0.03 0.945 0.94 0.045]);

            % --- Start / Stop ---
            obj.H.btn_start = uicontrol(ctrl, 'Style', 'togglebutton', ...
                'String', 'Start Capture', ...
                'FontSize', 10, 'FontWeight', 'bold', ...
                'BackgroundColor', [0.16 0.60 0.25], 'ForegroundColor', 'w', ...
                'Units', 'normalized', 'Position', [0.05 0.875 0.90 0.060], ...
                'Callback', @(src,~) obj.cb_start_(src));

            obj.gui_sep_(ctrl, 0.862);

            % --- Drone noise ---
            obj.H.btn_drone = uicontrol(ctrl, 'Style', 'togglebutton', ...
                'String', 'Drone: OFF', ...
                'FontSize', 9, ...
                'BackgroundColor', [0.22 0.22 0.22], ...
                'ForegroundColor', [0.60 0.60 0.60], ...
                'Units', 'normalized', 'Position', [0.05 0.795 0.90 0.055], ...
                'Callback', @(src,~) obj.cb_toggle_(src, 'drone'));
            obj.H.lbl_dg = uicontrol(ctrl, 'Style', 'text', ...
                'String', sprintf('Drone gain  %.2f', obj.drone_gain), ...
                'FontSize', 8, 'BackgroundColor', [0.14 0.14 0.14], ...
                'ForegroundColor', [0.58 0.58 0.58], ...
                'Units', 'normalized', 'Position', [0.05 0.759 0.90 0.028]);
            obj.H.sld_drone = uicontrol(ctrl, 'Style', 'slider', ...
                'Min', 0, 'Max', 1, 'Value', obj.drone_gain, ...
                'Units', 'normalized', 'Position', [0.05 0.729 0.90 0.030], ...
                'Callback', @(src,~) obj.cb_gain_(src, 'drone'));

            obj.gui_sep_(ctrl, 0.715);

            % --- Env noise ---
            obj.H.btn_env = uicontrol(ctrl, 'Style', 'togglebutton', ...
                'String', 'Env: OFF', ...
                'FontSize', 9, ...
                'BackgroundColor', [0.22 0.22 0.22], ...
                'ForegroundColor', [0.60 0.60 0.60], ...
                'Units', 'normalized', 'Position', [0.05 0.648 0.90 0.055], ...
                'Callback', @(src,~) obj.cb_toggle_(src, 'env'));
            obj.H.lbl_eg = uicontrol(ctrl, 'Style', 'text', ...
                'String', sprintf('Env gain  %.2f', obj.env_gain), ...
                'FontSize', 8, 'BackgroundColor', [0.14 0.14 0.14], ...
                'ForegroundColor', [0.58 0.58 0.58], ...
                'Units', 'normalized', 'Position', [0.05 0.612 0.90 0.028]);
            obj.H.sld_env = uicontrol(ctrl, 'Style', 'slider', ...
                'Min', 0, 'Max', 1, 'Value', obj.env_gain, ...
                'Units', 'normalized', 'Position', [0.05 0.582 0.90 0.030], ...
                'Callback', @(src,~) obj.cb_gain_(src, 'env'));

            obj.gui_sep_(ctrl, 0.568);

            % --- Playback source toggle ---
            uicontrol(ctrl, 'Style', 'text', 'String', 'Playback', ...
                'FontSize', 9, 'FontWeight', 'bold', ...
                'BackgroundColor', [0.14 0.14 0.14], ...
                'ForegroundColor', [0.85 0.85 0.85], ...
                'Units', 'normalized', 'Position', [0.05 0.530 0.90 0.028]);
            isEnh = strcmpi(obj.playback_source, 'enhanced');
            obj.H.btn_playback = uicontrol(ctrl, 'Style', 'togglebutton', ...
                'Value', double(isEnh), ...
                'String', ternary_str(isEnh, 'Play: Enhanced (MWF)', 'Play: Noisy Mix'), ...
                'FontSize', 9, ...
                'BackgroundColor', ternary_vec(isEnh, [0.15 0.45 0.70], [0.35 0.35 0.35]), ...
                'ForegroundColor', 'w', ...
                'Units', 'normalized', 'Position', [0.05 0.475 0.90 0.050], ...
                'Callback', @(src,~) obj.cb_playback_(src));

            obj.gui_sep_(ctrl, 0.462);

            % --- Scene summary ---
            bpf = round(cfg.drone_rpm * cfg.drone_blades / 60);
            gstr = sprintf([ ...
                'Human      : %.0f cm\n' ...
                'Mouth z    : %.2f m\n' ...
                'Drone AGL  : %.2f m\n' ...
                'Slant      : %.2f m\n' ...
                'Asphalt R  : %.2f\n' ...
                'BPF        : %d Hz\n' ...
                'Mic spacing: %.0f cm\n' ...
                'VAD        : %s'], ...
                cfg.human_height*100, cfg.mouth_height, obj.geo.drone_agl, ...
                norm(obj.geo.pos_drone - obj.geo.pos_human), cfg.ground_R, ...
                bpf, cfg.mic_spacing*100, obj.vad.backend_name);
            uicontrol(ctrl, 'Style', 'text', 'String', gstr, ...
                'FontSize', 8, 'BackgroundColor', [0.14 0.14 0.14], ...
                'ForegroundColor', [0.50 0.78 0.50], ...
                'Units', 'normalized', 'Position', [0.03 0.235 0.94 0.220], ...
                'HorizontalAlignment', 'left');

            obj.gui_sep_(ctrl, 0.222);

            obj.H.lbl_status = uicontrol(ctrl, 'Style', 'text', ...
                'String', sprintf('Idle\nfs=%d Hz\nframe=%d smp', ...
                                  cfg.fs, cfg.frame_size), ...
                'FontSize', 8, 'BackgroundColor', [0.14 0.14 0.14], ...
                'ForegroundColor', [0.50 0.50 0.50], ...
                'Units', 'normalized', 'Position', [0.03 0.01 0.94 0.205], ...
                'HorizontalAlignment', 'left');
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
            title(obj.H.ax_spec_noisy, 'Spectrogram — Noisy (Mic 1)', ...
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
                ylabel(obj.H.ax_wav(m), sprintf('Mic %d', m), ...
                       'Color', mc{idx}, 'FontSize', 9);
                obj.H.hlines(m) = plot(obj.H.ax_wav(m), 1:N, zeros(N, 1), ...
                    'Color', mc{idx}, 'LineWidth', 0.75);
                if m == 1
                    title(obj.H.ax_wav(m), ...
                        sprintf('Live Noisy Waveforms (MacBook %d-mic virtual array)', cfg.n_mics), ...
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

            mic = expand_channels(raw, obj.geo, cfg.n_mics);

            if obj.drone_on
                chunk = obj.audio.next_drone_chunk(N);
                mic = mic + obj.drone_gain * repmat(chunk, 1, cfg.n_mics);
            end
            if obj.env_on
                chunk = obj.audio.next_env_chunk(N);
                mic = mic + obj.env_gain * repmat(chunk, 1, cfg.n_mics);
            end

            % --- VAD on reference mic ---
            ref = mic(:, cfg.mwf.ref_mic);
            [is_speech, vad_score] = obj.vad.step(ref); %#ok<ASGLU>

            % --- MWF (pass-through for now) ---
            y_enh = obj.mwf.step(mic, is_speech);

            % --- Playback ---
            if cfg.playback.enabled
                if strcmpi(obj.playback_source, 'enhanced')
                    out = y_enh;
                else
                    out = mean(mic, 2);
                end
                obj.audio.play(out);
            end

            obj.update_buffers_(mic, y_enh);
            obj.refresh_display_(mic, y_enh);
        end

        function update_buffers_(obj, mic, y_enh)
            N   = obj.Cfg.frame_size;
            nb  = N/2 + 1;
            w   = hann(N, 'periodic');
            Xn  = abs(fft(mic(:, 1) .* w, N));
            Xm  = abs(fft(y_enh      .* w, N));
            obj.spec_buf_noisy(:, obj.spec_col) = Xn(1:nb);
            obj.spec_buf_mwf  (:, obj.spec_col) = Xm(1:nb);
            obj.spec_col = mod(obj.spec_col, obj.spec_ncols) + 1;

            [scores, flags] = obj.vad.trace();
            obj.vad_trace = scores(:).';
            obj.vad_flags = flags(:).';
        end

        function refresh_display_(obj, mic, y_enh)
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

            set(obj.H.himg_noisy, 'CData', 20*log10(obj.spec_buf_noisy + 1e-12));
            set(obj.H.himg_mwf,   'CData', 20*log10(obj.spec_buf_mwf   + 1e-12));

            set(obj.H.h_vad_line, 'YData', obj.vad_trace);
            % shade regions where flags==true (drawn as alternating strips)
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
            % Build a faces/vertices patch
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
                obj.H.lbl_status.String = sprintf( ...
                    'Capturing\nch: %d  fs: %d\nframe: %d smp\nVAD: %s', ...
                    obj.audio.n_hw_ch, obj.Cfg.fs, obj.Cfg.frame_size, ...
                    obj.vad.backend_name);
                obj.H.lbl_status.ForegroundColor = [0.28 0.88 0.38];
                start(obj.timerObj);
            else
                obj.running = false;
                src.String          = 'Start Capture';
                src.BackgroundColor = [0.16 0.60 0.25];
                obj.H.lbl_status.String = 'Stopped';
                obj.H.lbl_status.ForegroundColor = [0.70 0.42 0.18];
                stop(obj.timerObj);
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

        function cb_playback_(obj, src)
            if src.Value == 1
                obj.playback_source = 'enhanced';
                src.String = 'Play: Enhanced (MWF)';
                src.BackgroundColor = [0.15 0.45 0.70];
            else
                obj.playback_source = 'noisy';
                src.String = 'Play: Noisy Mix';
                src.BackgroundColor = [0.35 0.35 0.35];
            end
        end

        function safe_release_(obj)
            try obj.audio.release(); catch, end
        end

        function on_delete_(obj)
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

% -------------------- module helpers --------------------
function s = ternary_str(cond, a, b)
    if cond, s = a; else, s = b; end
end

function v = ternary_vec(cond, a, b)
    if cond, v = a; else, v = b; end
end
