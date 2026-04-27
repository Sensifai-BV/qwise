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
                'Units', 'normalized', 'Position', [0.03 0.955 0.94 0.040]);

            % --- Start / Stop capture ---
            obj.H.btn_start = uicontrol(ctrl, 'Style', 'togglebutton', ...
                'String', 'Start Capture', ...
                'FontSize', 10, 'FontWeight', 'bold', ...
                'BackgroundColor', [0.16 0.60 0.25], 'ForegroundColor', 'w', ...
                'Units', 'normalized', 'Position', [0.05 0.895 0.90 0.050], ...
                'Callback', @(src,~) obj.cb_start_(src));

            obj.gui_sep_(ctrl, 0.886);

            % --- Drone noise ---
            obj.H.btn_drone = uicontrol(ctrl, 'Style', 'togglebutton', ...
                'String', 'Drone: OFF', ...
                'FontSize', 9, ...
                'BackgroundColor', [0.22 0.22 0.22], ...
                'ForegroundColor', [0.60 0.60 0.60], ...
                'Units', 'normalized', 'Position', [0.05 0.828 0.90 0.045], ...
                'Callback', @(src,~) obj.cb_toggle_(src, 'drone'));
            obj.H.lbl_dg = uicontrol(ctrl, 'Style', 'text', ...
                'String', sprintf('Drone gain  %.2f', obj.drone_gain), ...
                'FontSize', 8, 'BackgroundColor', [0.14 0.14 0.14], ...
                'ForegroundColor', [0.58 0.58 0.58], ...
                'Units', 'normalized', 'Position', [0.05 0.798 0.90 0.023]);
            obj.H.sld_drone = uicontrol(ctrl, 'Style', 'slider', ...
                'Min', 0, 'Max', 1, 'Value', obj.drone_gain, ...
                'Units', 'normalized', 'Position', [0.05 0.773 0.90 0.024], ...
                'Callback', @(src,~) obj.cb_gain_(src, 'drone'));

            obj.gui_sep_(ctrl, 0.762);

            % --- Env noise ---
            obj.H.btn_env = uicontrol(ctrl, 'Style', 'togglebutton', ...
                'String', 'Env: OFF', ...
                'FontSize', 9, ...
                'BackgroundColor', [0.22 0.22 0.22], ...
                'ForegroundColor', [0.60 0.60 0.60], ...
                'Units', 'normalized', 'Position', [0.05 0.705 0.90 0.045], ...
                'Callback', @(src,~) obj.cb_toggle_(src, 'env'));
            obj.H.lbl_eg = uicontrol(ctrl, 'Style', 'text', ...
                'String', sprintf('Env gain  %.2f', obj.env_gain), ...
                'FontSize', 8, 'BackgroundColor', [0.14 0.14 0.14], ...
                'ForegroundColor', [0.58 0.58 0.58], ...
                'Units', 'normalized', 'Position', [0.05 0.675 0.90 0.023]);
            obj.H.sld_env = uicontrol(ctrl, 'Style', 'slider', ...
                'Min', 0, 'Max', 1, 'Value', obj.env_gain, ...
                'Units', 'normalized', 'Position', [0.05 0.650 0.90 0.024], ...
                'Callback', @(src,~) obj.cb_gain_(src, 'env'));

            obj.gui_sep_(ctrl, 0.639);

            % --- Processing toggles (pipeline stages) ---
            uicontrol(ctrl, 'Style', 'text', 'String', 'Processing', ...
                'FontSize', 9, 'FontWeight', 'bold', ...
                'BackgroundColor', [0.14 0.14 0.14], ...
                'ForegroundColor', [0.85 0.85 0.85], ...
                'Units', 'normalized', 'Position', [0.05 0.600 0.90 0.028]);

            obj.H.btn_vad_enable = uicontrol(ctrl, 'Style', 'togglebutton', ...
                'Value', double(obj.vad_on), ...
                'String', 'Enable VAD: OFF', ...
                'FontSize', 9, ...
                'BackgroundColor', [0.22 0.22 0.22], ...
                'ForegroundColor', [0.60 0.60 0.60], ...
                'Units', 'normalized', 'Position', [0.05 0.547 0.90 0.045], ...
                'Callback', @(src,~) obj.cb_vad_enable_(src));

            obj.H.btn_mwf_enable = uicontrol(ctrl, 'Style', 'togglebutton', ...
                'Value', double(obj.mwf_on), ...
                'String', 'Enable MWF: OFF', ...
                'FontSize', 9, ...
                'BackgroundColor', [0.22 0.22 0.22], ...
                'ForegroundColor', [0.60 0.60 0.60], ...
                'Units', 'normalized', 'Position', [0.05 0.495 0.90 0.045], ...
                'Callback', @(src,~) obj.cb_mwf_enable_(src));

            obj.gui_sep_(ctrl, 0.483);

            % --- Recording ---
            obj.H.btn_record = uicontrol(ctrl, 'Style', 'togglebutton', ...
                'String', 'Record: OFF', ...
                'FontSize', 9, 'FontWeight', 'bold', ...
                'BackgroundColor', [0.22 0.22 0.22], ...
                'ForegroundColor', [0.60 0.60 0.60], ...
                'Units', 'normalized', 'Position', [0.05 0.428 0.90 0.045], ...
                'Callback', @(src,~) obj.cb_record_(src));

            obj.gui_sep_(ctrl, 0.418);

            % --- Scene summary ---
            bpf = round(cfg.drone_rpm * cfg.drone_blades / 60);
            gstr = sprintf([ ...
                'Wiring     : %s\n' ...
                'Composite  : %s\n' ...
                'Human      : %.0f cm\n' ...
                'Mouth z    : %.2f m\n' ...
                'Drone AGL  : %.2f m\n' ...
                'Slant      : %.2f m\n' ...
                'Asphalt R  : %.2f\n' ...
                'BPF        : %d Hz\n' ...
                'Mic spacing: %.0f cm\n' ...
                'Speech gain: %.2f\n' ...
                'Drone gain : %.2f\n' ...
                'Env gain   : %.2f\n' ...
                'VAD        : %s'], ...
                obj.mixer.mode, obj.mixer.composite_kind, ...
                cfg.human_height*100, cfg.mouth_height, obj.geo.drone_agl, ...
                norm(obj.geo.pos_drone - obj.geo.pos_human), cfg.ground_R, ...
                bpf, cfg.mic_spacing*100, ...
                obj.geo.gains_speech(cfg.mwf.ref_mic), ...
                obj.geo.gains_drone(cfg.mwf.ref_mic), ...
                obj.geo.gains_env(cfg.mwf.ref_mic), ...
                obj.vad.backend_name);
            uicontrol(ctrl, 'Style', 'text', 'String', gstr, ...
                'FontSize', 8, 'BackgroundColor', [0.14 0.14 0.14], ...
                'ForegroundColor', [0.50 0.78 0.50], ...
                'Units', 'normalized', 'Position', [0.03 0.170 0.94 0.242], ...
                'HorizontalAlignment', 'left');

            obj.gui_sep_(ctrl, 0.160);

            obj.H.lbl_status = uicontrol(ctrl, 'Style', 'text', ...
                'String', sprintf('Idle\nfs=%d Hz\nframe=%d smp', ...
                                  cfg.fs, cfg.frame_size), ...
                'FontSize', 8, 'BackgroundColor', [0.14 0.14 0.14], ...
                'ForegroundColor', [0.50 0.50 0.50], ...
                'Units', 'normalized', 'Position', [0.03 0.01 0.94 0.145], ...
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
            per_channel = strcmpi(obj.mixer.mode, 'perChannel');
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
                ylabel(obj.H.ax_wav(m), mic_label_(per_channel, m), ...
                       'Color', mc{idx}, 'FontSize', 9);
                if m == 1
                    % Green-tint background patch lights up while the VAD
                    % is detecting speech (refresh_display_ toggles its
                    % FaceAlpha each tick). Created BEFORE the line plot
                    % so the waveform draws on top.
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
                    if per_channel
                        ttl = ['Per-Mic Sources  (Mic 1 = Speech + Drone + Env  ·  ' ...
                               'Mic 2 = Drone  ·  Mic 3 = Env  ·  green tint = VAD speech)'];
                    else
                        ttl = sprintf(['Physical %d-Mic Array  (every mic receives ' ...
                                       'speech+drone+env with TDOA + 1/d gain ·  ' ...
                                       'green tint on Mic 1 = VAD speech)'], cfg.n_mics);
                    end
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
            speech = raw(:, 1);                             % live input mic
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

            % --- Composite feed for the VAD (mic-1 in perChannel mode) ---
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

            % --- Recording (independent of VAD / MWF) ---
            if obj.recording
                switch lower(cfg.record.source)
                    case 'raw_mic'
                        obj.audio.rec_write(speech);
                    case 'speech'
                        obj.audio.rec_write(mic(:, 1));
                    otherwise   % 'composite' (default)
                        obj.audio.rec_write(comp);
                end
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
                    'Enable MWF', [0.15 0.45 0.70]);
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
            obj.set_button_state_(src, on, 'Enable MWF', [0.15 0.45 0.70]);
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
            want_on = logical(src.Value);
            if want_on && ~obj.recording
                path = obj.audio.rec_start();
                if isempty(path)
                    set(src, 'Value', 0);
                    warndlg('Could not open recording file.', 'Q-WiSE');
                    return;
                end
                obj.recording = true;
                src.String = 'Record: ● REC';
                src.BackgroundColor = [0.78 0.10 0.10];
                src.ForegroundColor = 'w';
            elseif ~want_on && obj.recording
                path = obj.audio.rec_stop();
                obj.recording = false;
                src.String = 'Record: OFF';
                src.BackgroundColor = [0.22 0.22 0.22];
                src.ForegroundColor = [0.60 0.60 0.60];
                if ~isempty(path)
                    fprintf('[Q-WiSE] WAV saved to %s\n', path);
                end
            end
        end

        function safe_release_(obj)
            try obj.audio.release(); catch, end
        end

        function on_delete_(obj)
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
function s = mic_label_(per_channel, m)
%MIC_LABEL_  Friendly ylabel for each mic row based on wiring mode.
    if per_channel
        switch m
            case 1, s = 'Mic 1  Noisy';   % speech + drone + env
            case 2, s = 'Mic 2  Drone';
            case 3, s = 'Mic 3  Env';
            otherwise, s = sprintf('Mic %d  Noisy·τ', m);
        end
    else
        s = sprintf('Mic %d', m);
    end
end
