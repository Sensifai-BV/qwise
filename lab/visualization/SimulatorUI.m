classdef SimulatorUI < handle
%SIMULATORUI  Q-WiSE acoustic simulator — file-based mix + ONNX denoise UI.
%
%   ui = SimulatorUI(cfg, geo, audio, enhancer)
%   ui.start()
%
%   Workflow:
%     1. Pick a clean-speech sample (dropdown) and listen to it.
%     2. Pick a scene preset (drone distance / position).
%     3. "Play Mixed" — mix speech + drone-fan + env noise through the
%        physical N-mic array and play it back with live per-mic
%        waveforms and a live spectrogram of the noisy mix.
%     4. "Clean (ONNX)" — feed the noisy N-mic array to qwise.onnx, play
%        the enhanced speech with a live waveform + spectrogram, and
%        auto-record mic01..micNN.wav + clean.wav to disk.
%     5. Latest-recording grid plays back each captured WAV.

    properties
        Cfg
        geo
        audio
        mixer
        enh                 % OnnxEnhancer
        fig
        H = struct()

        drone_gain
        env_gain
    end

    properties (Access = private)
        % --- cached audio ---
        speech_         = []     % loaded clean-speech sample [L x 1]
        speech_name_    = ''
        mic_mix_        = []     % noisy array [L x n_mics]
        mix_norm_       = 1      % shared display scale (preserves per-mic level)
        play_mix_       = []     % mono mix for playback [L x 1]
        clean_          = []     % ONNX output [L x 1]

        % --- playback / animation ---
        player_         = []
        anim_timer_     = []
        mode_           = ''     % '' | 'sample' | 'mixed' | 'clean'
        anim_sig_       = []     % mono signal currently animated (clean mode)

        % --- spectrogram ring buffers ---
        spec_ncols
        spec_mix_
        spec_clean_
        spec_col_mix_   = 1
        spec_col_clean_ = 1

        % --- right-pane layout (for rebuilding the per-mic rows) ---
        rx_ = 0
        rw_ = 0

        % --- latest-recording playback ---
        playback_panel_
        playback_player_     = []
        playback_active_btn_ = []
        playback_files_      = {}
        playback_dir_        = ''
    end

    methods
        function obj = SimulatorUI(cfg, geo, audio, enhancer)
            obj.Cfg   = cfg;
            obj.geo   = geo;
            obj.audio = audio;
            obj.enh   = enhancer;
            obj.mixer = SourceMixer(cfg, geo);

            obj.drone_gain = cfg.drone_gain_init;
            obj.env_gain   = cfg.env_gain_init;
            obj.spec_ncols = cfg.ui.spec_ncols;

            nb = cfg.frame_size/2 + 1;
            obj.spec_mix_   = zeros(nb, obj.spec_ncols);
            obj.spec_clean_ = zeros(nb, obj.spec_ncols);

            obj.build_figure_();
            obj.fig.DeleteFcn = @(~,~) obj.on_delete_();
        end

        function start(obj)
            fprintf('[Q-WiSE] Ready — pick a sample, then Play Mixed / Clean.\n');
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
                'Name', 'Q-WiSE Acoustic Simulator', ...
                'Color', [0.10 0.10 0.10], ...
                'Position', cfg.ui.fig_position, ...
                'MenuBar', 'none', 'ToolBar', 'figure', ...
                'NumberTitle', 'off');

            cpw = 0.165;
            obj.build_control_panel_(cpw);
            obj.build_axes_(cpw);
        end

        function build_control_panel_(obj, cpw)
            cfg  = obj.Cfg;
            ctrl = uipanel(obj.fig, ...
                'BackgroundColor', [0.14 0.14 0.14], ...
                'BorderType', 'none', ...
                'Units', 'normalized', 'Position', [0 0 cpw 1]);

            uicontrol(ctrl, 'Style', 'text', 'String', 'Q-WiSE', ...
                'FontSize', 13, 'FontWeight', 'bold', ...
                'BackgroundColor', [0.14 0.14 0.14], ...
                'ForegroundColor', [0.95 0.88 0.55], ...
                'Units', 'normalized', 'Position', [0.03 0.962 0.94 0.034]);

            % --- Speech sample selector + listen ---
            uicontrol(ctrl, 'Style', 'text', 'String', 'Speech sample', ...
                'FontSize', 9, 'FontWeight', 'bold', ...
                'BackgroundColor', [0.14 0.14 0.14], ...
                'ForegroundColor', [0.85 0.85 0.85], ...
                'HorizontalAlignment', 'left', ...
                'Units', 'normalized', 'Position', [0.05 0.930 0.90 0.022]);

            names = obj.audio.list_speech_samples();
            if isempty(names), names = {'(no samples found)'}; end
            obj.H.pop_speech = uicontrol(ctrl, 'Style', 'popupmenu', ...
                'String', names, 'Value', 1, ...
                'FontSize', 9, ...
                'BackgroundColor', [0.20 0.20 0.20], 'ForegroundColor', 'w', ...
                'Units', 'normalized', 'Position', [0.05 0.906 0.62 0.028], ...
                'Callback', @(~,~) obj.cb_speech_changed_());
            obj.H.btn_play_sample = uicontrol(ctrl, 'Style', 'togglebutton', ...
                'String', sprintf('%c Listen', 9658), ...
                'FontSize', 9, ...
                'BackgroundColor', [0.20 0.40 0.62], 'ForegroundColor', 'w', ...
                'Units', 'normalized', 'Position', [0.69 0.906 0.26 0.028], ...
                'Callback', @(s,~) obj.cb_play_sample_(s));

            obj.gui_sep_(ctrl, 0.898);

            % --- Scene presets (radio group) ---
            uicontrol(ctrl, 'Style', 'text', 'String', 'Scene preset', ...
                'FontSize', 9, 'FontWeight', 'bold', ...
                'BackgroundColor', [0.14 0.14 0.14], ...
                'ForegroundColor', [0.85 0.85 0.85], ...
                'HorizontalAlignment', 'left', ...
                'Units', 'normalized', 'Position', [0.05 0.872 0.90 0.022]);

            np = numel(cfg.presets);
            obj.H.bg_preset = uibuttongroup(ctrl, ...
                'Units', 'normalized', 'Position', [0.04 0.744 0.92 0.124], ...
                'BackgroundColor', [0.14 0.14 0.14], 'BorderType', 'none', ...
                'SelectionChangedFcn', @(~,e) obj.cb_preset_(e));
            rh = 1 / np;
            for k = 1:np
                rb = uicontrol(obj.H.bg_preset, 'Style', 'radiobutton', ...
                    'String', cfg.presets(k).name, ...
                    'FontSize', 8.5, ...
                    'BackgroundColor', [0.14 0.14 0.14], ...
                    'ForegroundColor', [0.80 0.86 0.80], ...
                    'Units', 'normalized', ...
                    'Position', [0.04 1 - k*rh 0.94 rh], ...
                    'UserData', k, ...
                    'Value', double(k == cfg.preset_default));
                if k == 1, obj.H.rb_first = rb; end
            end

            obj.gui_sep_(ctrl, 0.736);

            % --- Microphone array (count + spacing) ---
            uicontrol(ctrl, 'Style', 'text', 'String', 'Microphone array', ...
                'FontSize', 9, 'FontWeight', 'bold', ...
                'BackgroundColor', [0.14 0.14 0.14], ...
                'ForegroundColor', [0.85 0.85 0.85], ...
                'HorizontalAlignment', 'left', ...
                'Units', 'normalized', 'Position', [0.05 0.712 0.90 0.022]);

            mic_counts = 2:5;
            ni = find(mic_counts == cfg.n_mics, 1);  if isempty(ni), ni = 2; end
            obj.H.pop_nmics = uicontrol(ctrl, 'Style', 'popupmenu', ...
                'String', {'2 mics','3 mics','4 mics','5 mics'}, 'Value', ni, ...
                'FontSize', 8.5, ...
                'BackgroundColor', [0.20 0.20 0.20], 'ForegroundColor', 'w', ...
                'TooltipString', 'Number of microphones in the array', ...
                'Units', 'normalized', 'Position', [0.05 0.680 0.43 0.030], ...
                'Callback', @(s,~) obj.cb_nmics_(s));

            spacings_cm = [10 20 30];
            si = find(spacings_cm == round(cfg.mic_spacing*100), 1);
            if isempty(si), si = 1; end
            obj.H.pop_spacing = uicontrol(ctrl, 'Style', 'popupmenu', ...
                'String', {'10 cm','20 cm','30 cm'}, 'Value', si, ...
                'FontSize', 8.5, ...
                'BackgroundColor', [0.20 0.20 0.20], 'ForegroundColor', 'w', ...
                'TooltipString', 'Spacing between adjacent microphones', ...
                'Units', 'normalized', 'Position', [0.52 0.680 0.43 0.030], ...
                'Callback', @(s,~) obj.cb_spacing_(s));

            obj.gui_sep_(ctrl, 0.670);

            % --- Noise mix levels ---
            uicontrol(ctrl, 'Style', 'text', 'String', 'Noise mix', ...
                'FontSize', 9, 'FontWeight', 'bold', ...
                'BackgroundColor', [0.14 0.14 0.14], ...
                'ForegroundColor', [0.85 0.85 0.85], ...
                'HorizontalAlignment', 'left', ...
                'Units', 'normalized', 'Position', [0.05 0.646 0.90 0.022]);

            obj.H.lbl_dg = uicontrol(ctrl, 'Style', 'text', ...
                'String', sprintf('Drone fan  %.2f', obj.drone_gain), ...
                'FontSize', 8, 'BackgroundColor', [0.14 0.14 0.14], ...
                'ForegroundColor', [0.90 0.62 0.30], ...
                'HorizontalAlignment', 'left', ...
                'Units', 'normalized', 'Position', [0.05 0.624 0.90 0.020]);
            gmax = obj.gain_max_();
            obj.H.sld_drone = uicontrol(ctrl, 'Style', 'slider', ...
                'Min', 0, 'Max', gmax, 'Value', min(obj.drone_gain, gmax), ...
                'Units', 'normalized', 'Position', [0.05 0.604 0.90 0.020], ...
                'Callback', @(src,~) obj.cb_gain_(src, 'drone'));

            obj.H.lbl_eg = uicontrol(ctrl, 'Style', 'text', ...
                'String', sprintf('Environment  %.2f', obj.env_gain), ...
                'FontSize', 8, 'BackgroundColor', [0.14 0.14 0.14], ...
                'ForegroundColor', [0.72 0.45 0.90], ...
                'HorizontalAlignment', 'left', ...
                'Units', 'normalized', 'Position', [0.05 0.580 0.90 0.020]);
            obj.H.sld_env = uicontrol(ctrl, 'Style', 'slider', ...
                'Min', 0, 'Max', gmax, 'Value', min(obj.env_gain, gmax), ...
                'Units', 'normalized', 'Position', [0.05 0.560 0.90 0.020], ...
                'Callback', @(src,~) obj.cb_gain_(src, 'env'));

            obj.gui_sep_(ctrl, 0.550);

            % --- Transport: Play Mixed / Clean (ONNX) ---
            obj.H.btn_mixed = uicontrol(ctrl, 'Style', 'togglebutton', ...
                'String', sprintf('%c Play Mixed', 9658), ...
                'FontSize', 10, 'FontWeight', 'bold', ...
                'BackgroundColor', [0.16 0.52 0.66], 'ForegroundColor', 'w', ...
                'Units', 'normalized', 'Position', [0.05 0.500 0.90 0.044], ...
                'Callback', @(s,~) obj.cb_mixed_(s));

            obj.H.btn_clean = uicontrol(ctrl, 'Style', 'togglebutton', ...
                'String', sprintf('%c Clean (ONNX)', 9658), ...
                'FontSize', 10, 'FontWeight', 'bold', ...
                'BackgroundColor', [0.16 0.60 0.25], 'ForegroundColor', 'w', ...
                'Units', 'normalized', 'Position', [0.05 0.452 0.90 0.044], ...
                'Callback', @(s,~) obj.cb_clean_(s));

            obj.gui_sep_(ctrl, 0.444);

            % --- Status ---
            obj.H.lbl_status = uicontrol(ctrl, 'Style', 'text', ...
                'String', obj.status_text_('Idle'), ...
                'FontSize', 8, 'BackgroundColor', [0.14 0.14 0.14], ...
                'ForegroundColor', [0.55 0.78 0.55], ...
                'HorizontalAlignment', 'left', ...
                'Units', 'normalized', 'Position', [0.03 0.356 0.94 0.082]);

            obj.gui_sep_(ctrl, 0.348);

            % --- Latest recording playback ---
            uicontrol(ctrl, 'Style', 'text', 'String', 'Latest recording', ...
                'FontSize', 9, 'FontWeight', 'bold', ...
                'BackgroundColor', [0.14 0.14 0.14], ...
                'ForegroundColor', [0.85 0.85 0.85], ...
                'HorizontalAlignment', 'left', ...
                'Units', 'normalized', 'Position', [0.05 0.322 0.55 0.022]);
            obj.H.btn_refresh = uicontrol(ctrl, 'Style', 'pushbutton', ...
                'String', 'Refresh', 'FontSize', 8, ...
                'BackgroundColor', [0.30 0.30 0.30], 'ForegroundColor', 'w', ...
                'Units', 'normalized', 'Position', [0.62 0.322 0.33 0.024], ...
                'Callback', @(~,~) obj.refresh_playback_());
            obj.playback_panel_ = uipanel(ctrl, ...
                'BackgroundColor', [0.10 0.10 0.10], 'BorderType', 'line', ...
                'HighlightColor', [0.25 0.25 0.25], ...
                'Units', 'normalized', 'Position', [0.03 0.02 0.94 0.296]);
            obj.refresh_playback_();
        end

        function build_axes_(obj, cpw)
            cfg = obj.Cfg;
            rx  = cpw + 0.010;
            rw  = 1 - rx - 0.010;
            obj.rx_ = rx;  obj.rw_ = rw;     % remembered for per-mic rebuilds

            % -------- Row 1 : 3D scene + mixed spec + clean spec --------
            row1_y = 0.560;  row1_h = 0.410;
            scene_w = rw * 0.34;  spec_w = rw * 0.31;  gap = rw * 0.010;

            obj.H.ax_scene = axes(obj.fig, 'Units', 'normalized', ...
                'Position', [rx row1_y scene_w row1_h], ...
                'Color', [0.07 0.07 0.07], ...
                'XColor', [0.5 0.5 0.5], 'YColor', [0.5 0.5 0.5], ...
                'ZColor', [0.5 0.5 0.5], 'GridColor', [0.22 0.22 0.22]);

            specMix_x = rx + scene_w + gap;
            obj.H.ax_spec_mix = axes(obj.fig, 'Units', 'normalized', ...
                'Position', [specMix_x row1_y spec_w row1_h], ...
                'Color', [0.07 0.07 0.07], ...
                'XColor', [0.5 0.5 0.5], 'YColor', [0.5 0.5 0.5], ...
                'GridColor', [0.22 0.22 0.22]);

            specCln_x = specMix_x + spec_w + gap;
            specCln_w = rx + rw - specCln_x;
            obj.H.ax_spec_clean = axes(obj.fig, 'Units', 'normalized', ...
                'Position', [specCln_x row1_y specCln_w row1_h], ...
                'Color', [0.07 0.07 0.07], ...
                'XColor', [0.5 0.5 0.5], 'YColor', [0.5 0.5 0.5], ...
                'GridColor', [0.22 0.22 0.22]);

            N   = cfg.frame_size;
            nb  = N/2 + 1;
            frq = linspace(0, cfg.fs/2000, nb);

            obj.H.himg_mix = imagesc(obj.H.ax_spec_mix, 1:obj.spec_ncols, frq, ...
                                     obj.spec_mix_);
            colormap(obj.H.ax_spec_mix, 'hot');
            clim(obj.H.ax_spec_mix, [-80 -20]);
            axis(obj.H.ax_spec_mix, 'xy');
            ylabel(obj.H.ax_spec_mix, 'Freq (kHz)', 'Color', [0.5 0.5 0.5]);
            title(obj.H.ax_spec_mix, 'Spectrogram — Noisy Mix', ...
                  'Color', 'w', 'FontSize', 9, 'FontWeight', 'bold');

            obj.H.himg_clean = imagesc(obj.H.ax_spec_clean, 1:obj.spec_ncols, frq, ...
                                       obj.spec_clean_);
            colormap(obj.H.ax_spec_clean, 'hot');
            clim(obj.H.ax_spec_clean, [-80 -20]);
            axis(obj.H.ax_spec_clean, 'xy');
            ylabel(obj.H.ax_spec_clean, 'Freq (kHz)', 'Color', [0.5 0.5 0.5]);
            title(obj.H.ax_spec_clean, 'Spectrogram — ONNX Clean', ...
                  'Color', 'w', 'FontSize', 9, 'FontWeight', 'bold');

            % -------- Row 2 : per-mic noisy waveforms (rebuildable) -----
            obj.build_wave_rows_();

            % -------- Row 3 : clean output waveform --------
            row3_y = 0.060;  row3_h = 0.150;
            obj.H.ax_clean_wave = axes(obj.fig, 'Units', 'normalized', ...
                'Position', [rx row3_y rw row3_h], ...
                'Color', [0.06 0.06 0.06], ...
                'XColor', [0.45 0.45 0.45], 'YColor', [0.20 0.85 0.40], ...
                'GridColor', [0.20 0.20 0.20]);
            hold(obj.H.ax_clean_wave, 'on'); grid(obj.H.ax_clean_wave, 'on');
            ylim(obj.H.ax_clean_wave, [-1.05 1.05]); xlim(obj.H.ax_clean_wave, [0 1]);
            ylabel(obj.H.ax_clean_wave, 'Amplitude', ...
                'Color', [0.70 0.70 0.70], 'FontSize', 9);
            xlabel(obj.H.ax_clean_wave, 'time (s)', 'Color', [0.50 0.50 0.50]);
            title(obj.H.ax_clean_wave, ...
                'ONNX Enhanced Output  (gray = noisy mic 1, green = clean)', ...
                'Color', 'w', 'FontSize', 9, 'FontWeight', 'bold');
            % Gray noisy (behind) + green clean (front) + playhead. Filled
            % with the full clip when Clean runs; see draw_clean_overlay_.
            obj.H.hline_noisy = plot(obj.H.ax_clean_wave, NaN, NaN, ...
                'Color', [0.72 0.72 0.72], 'LineWidth', 0.5);
            obj.H.hline_clean = plot(obj.H.ax_clean_wave, NaN, NaN, ...
                'Color', [0.16 0.78 0.36], 'LineWidth', 0.7);
            obj.H.clean_playhead = plot(obj.H.ax_clean_wave, [0 0], [-1.05 1.05], ...
                '-', 'Color', [0.95 0.85 0.30], 'LineWidth', 1.0);
            legend(obj.H.ax_clean_wave, ...
                [obj.H.hline_noisy, obj.H.hline_clean], ...
                {'noisy (mic 1)', 'clean'}, ...
                'TextColor', [0.8 0.8 0.8], 'Color', [0.10 0.10 0.10], ...
                'EdgeColor', [0.30 0.30 0.30], 'FontSize', 7, ...
                'Location', 'northeast');

            draw_scene(obj.H.ax_scene, obj.geo, cfg);
        end

        function gui_sep_(~, parent, y)
            uicontrol(parent, 'Style', 'text', 'String', '', ...
                'BackgroundColor', [0.32 0.32 0.32], ...
                'Units', 'normalized', 'Position', [0.03 y 0.94 0.003]);
        end

        function build_wave_rows_(obj)
        %BUILD_WAVE_ROWS_  (Re)create the per-mic waveform axes for the
        %   current cfg.n_mics. Safe to call after the mic count changes —
        %   it deletes any existing rows first.
            cfg = obj.Cfg;
            N   = cfg.frame_size;

            if isfield(obj.H, 'wave_axes')
                for a = reshape(obj.H.wave_axes, 1, [])
                    if isgraphics(a), delete(a); end
                end
            end

            rx = obj.rx_;  rw = obj.rw_;
            row2_y_top = 0.535;  row2_y_bot = 0.250;
            wh = (row2_y_top - row2_y_bot) / cfg.n_mics;

            mc = {[0.00 0.78 0.32], [0.00 0.55 0.95], [1.00 0.52 0.00], ...
                  [0.95 0.35 0.75], [0.60 0.85 0.20]};
            obj.H.wave_axes = gobjects(cfg.n_mics, 1);
            obj.H.hlines    = gobjects(cfg.n_mics, 1);
            for m = 1:cfg.n_mics
                idx = mod(m-1, numel(mc)) + 1;
                yp  = row2_y_bot + (cfg.n_mics - m) * wh;
                ax  = axes(obj.fig, 'Units', 'normalized', ...
                    'Position', [rx yp rw wh - 0.010], ...
                    'Color', [0.06 0.06 0.06], ...
                    'XColor', [0.45 0.45 0.45], 'YColor', mc{idx}, ...
                    'GridColor', [0.20 0.20 0.20]);
                hold(ax, 'on'); grid(ax, 'on');
                ylim(ax, [-1.05 1.05]); xlim(ax, [1 N]);
                ylabel(ax, sprintf('Mic %d', m), 'Color', mc{idx}, 'FontSize', 9);
                obj.H.wave_axes(m) = ax;
                obj.H.hlines(m) = plot(ax, 1:N, zeros(N, 1), ...
                    'Color', mc{idx}, 'LineWidth', 0.75);
                if m == 1
                    title(ax, sprintf(['Physical %d-Mic Array  ' ...
                        '(speech + drone + env, per-mic TDOA + 1/r gain)'], ...
                        cfg.n_mics), ...
                        'Color', 'w', 'FontSize', 9, 'FontWeight', 'bold');
                end
            end
        end

        function s = status_text_(obj, headline)
            cfg = obj.Cfg;
            d_drone = norm(obj.geo.pos_drone - obj.geo.pos_human);
            spname  = obj.speech_name_;
            if isempty(spname), spname = '(none)'; end
            s = sprintf([ ...
                '%s\n' ...
                'sample : %s\n' ...
                'fs %d  mics %d  N %d\n' ...
                'drone %.2f m  human %.0f cm  env %.1f m'], ...
                headline, spname, cfg.fs, cfg.n_mics, cfg.frame_size, ...
                d_drone, cfg.human_height*100, cfg.env.distance_from_mouth);
        end

        function g = gain_max_(obj)
            if isfield(obj.Cfg, 'gain_max') && ~isempty(obj.Cfg.gain_max)
                g = obj.Cfg.gain_max;
            else
                g = 0.30;
            end
        end

        function set_status_(obj, msg, color)
            if isfield(obj.H, 'lbl_status') && isvalid(obj.H.lbl_status)
                obj.H.lbl_status.String          = msg;
                obj.H.lbl_status.ForegroundColor = color;
            end
        end
    end

    % ==================================================================
    %  MIXING + ONNX
    % ==================================================================
    methods (Access = private)
        function ok = ensure_speech_(obj)
        %ENSURE_SPEECH_  Load the currently-selected sample if needed.
            ok = false;
            names = obj.H.pop_speech.String;
            if ~iscell(names), names = {names}; end
            sel = names{obj.H.pop_speech.Value};
            if startsWith(sel, '(')
                obj.set_status_('No speech samples in samples folder.', ...
                                [0.95 0.55 0.20]);
                return;
            end
            if ~strcmp(sel, obj.speech_name_) || isempty(obj.speech_)
                try
                    obj.speech_      = obj.audio.load_speech_sample(sel);
                    obj.speech_name_ = sel;
                    obj.mic_mix_     = [];      % invalidate cached mix
                    obj.clean_       = [];
                catch ME
                    obj.set_status_(sprintf('Load failed: %s', ME.message), ...
                                    [0.95 0.45 0.20]);
                    return;
                end
            end
            ok = ~isempty(obj.speech_);
        end

        function ok = build_mix_(obj)
        %BUILD_MIX_  Mix the loaded speech with drone + env noise through
        %   the physical array. Caches mic_mix_ (=[L x N]) and play_mix_.
            ok = false;
            if ~obj.ensure_speech_(), return; end
            if ~isempty(obj.mic_mix_), ok = true; return; end   % cached

            L = numel(obj.speech_);
            obj.audio.reset_noise();
            speech = obj.Cfg.speech_gain * obj.speech_;
            drone  = obj.drone_gain * obj.audio.next_drone_chunk(L);
            envn   = obj.env_gain   * obj.audio.next_env_chunk(L);

            obj.mixer.reset();
            mic = obj.mixer.mix(speech, drone, envn);           % [L x N]
            obj.mic_mix_ = mic;

            % FIXED headroom — the SAME scale for every preset, so the
            % distance falloff is audibly/visibly louder (near) or quieter
            % (far). No per-preset auto-gain (that flattened the levels).
            master = 0.90;
            obj.mix_norm_ = 1 / master;      % display divides -> x master

            ref  = obj.mixer.ref_mic();
            obj.play_mix_ = max(min(mic(:, ref) * master, 1), -1);
            ok = true;
        end
    end

    % ==================================================================
    %  CALLBACKS — controls
    % ==================================================================
    methods (Access = private)
        function cb_speech_changed_(obj)
            obj.stop_playback_();
            obj.mic_mix_ = [];
            obj.clean_   = [];
            obj.ensure_speech_();
            obj.set_status_(obj.status_text_('Idle'), [0.55 0.78 0.55]);
        end

        function cb_play_sample_(obj, src)
            if logical(src.Value)
                if ~obj.ensure_speech_()
                    set(src, 'Value', 0); return;
                end
                obj.start_playback_(obj.speech_, 'sample', src);
                src.String = sprintf('%c Stop', 9632);
            else
                obj.stop_playback_();
            end
        end

        function cb_preset_(obj, e)
        %CB_PRESET_  Apply the selected scene preset.
            obj.stop_playback_();

            % Read the selected preset index robustly (UserData on the
            % chosen radio button), with a fallback to SelectedObject.
            k = [];
            try, k = e.NewValue.UserData; catch, end
            if isempty(k) || ~isnumeric(k)
                sel = obj.H.bg_preset.SelectedObject;
                if isempty(sel) || isempty(sel.UserData), return; end
                k = sel.UserData;
            end
            k = double(k);
            if k < 1 || k > numel(obj.Cfg.presets), return; end

            obj.apply_preset_(k);
        end

        function apply_preset_(obj, k)
        %APPLY_PRESET_  Push preset k into the config, rebuild geometry,
        %   the mixer, and the 3-D scene.
            p = obj.Cfg.presets(k);
            obj.Cfg.human_height            = p.human_h;
            obj.Cfg.mouth_height            = 0.88 * p.human_h;
            obj.Cfg.slant_dist              = p.slant_dist;
            obj.Cfg.drone.azimuth_deg       = p.drone_az;
            obj.Cfg.env.distance_from_mouth = p.env_dist;

            obj.geo = build_geometry(obj.Cfg);
            obj.mixer.update_geometry(obj.geo);
            obj.mic_mix_ = [];          % geometry changed -> remix
            obj.clean_   = [];

            if isfield(obj.H, 'ax_scene') && isvalid(obj.H.ax_scene)
                cla(obj.H.ax_scene, 'reset');
                set(obj.H.ax_scene, 'Color', [0.07 0.07 0.07], ...
                    'XColor', [0.5 0.5 0.5], 'YColor', [0.5 0.5 0.5], ...
                    'ZColor', [0.5 0.5 0.5], 'GridColor', [0.22 0.22 0.22]);
                draw_scene(obj.H.ax_scene, obj.geo, obj.Cfg);
                drawnow;
            end
            d_drone = norm(obj.geo.pos_drone - obj.geo.pos_human);
            fprintf('[Q-WiSE] Preset %d: %s — drone at [%.2f %.2f %.2f] m (%.2f m)\n', ...
                    k, p.name, obj.geo.pos_drone, d_drone);
            obj.set_status_(obj.status_text_(sprintf('Preset: %s', p.name)), ...
                            [0.60 0.85 0.95]);
        end

        function cb_nmics_(obj, src)
        %CB_NMICS_  Change the microphone count (2..5).
            counts = 2:5;
            obj.set_array_(counts(src.Value), obj.Cfg.mic_spacing);
        end

        function cb_spacing_(obj, src)
        %CB_SPACING_  Change the inter-mic spacing (10/20/30 cm).
            cm = [10 20 30];
            obj.set_array_(obj.Cfg.n_mics, cm(src.Value) / 100);
        end

        function set_array_(obj, n_mics, spacing)
        %SET_ARRAY_  Apply a new array geometry. Changing the mic count
        %   needs a fresh SourceMixer (it forbids live n_mics changes) and
        %   a rebuild of the per-mic waveform rows.
            obj.stop_playback_();
            n_changed = (n_mics ~= obj.Cfg.n_mics);
            obj.Cfg.n_mics      = n_mics;
            obj.Cfg.mic_spacing = spacing;

            obj.geo   = build_geometry(obj.Cfg);
            obj.mixer = SourceMixer(obj.Cfg, obj.geo);   % rebuilt (n_mics may differ)
            obj.mic_mix_ = [];  obj.clean_ = [];
            fprintf('[Q-WiSE] Array set: %d mics @ %.0f cm (mixer rebuilt)\n', ...
                    obj.Cfg.n_mics, obj.Cfg.mic_spacing*100);

            if n_changed
                obj.build_wave_rows_();
            end
            if isfield(obj.H, 'ax_scene') && isvalid(obj.H.ax_scene)
                cla(obj.H.ax_scene, 'reset');
                set(obj.H.ax_scene, 'Color', [0.07 0.07 0.07], ...
                    'XColor', [0.5 0.5 0.5], 'YColor', [0.5 0.5 0.5], ...
                    'ZColor', [0.5 0.5 0.5], 'GridColor', [0.22 0.22 0.22]);
                draw_scene(obj.H.ax_scene, obj.geo, obj.Cfg);
            end
            drawnow;
            obj.set_status_(obj.status_text_(sprintf('Array: %d mics @ %.0f cm', ...
                n_mics, spacing*100)), [0.60 0.85 0.95]);
        end

        function cb_gain_(obj, src, which)
            if strcmp(which, 'drone')
                obj.drone_gain = src.Value;
                obj.H.lbl_dg.String = sprintf('Drone fan  %.2f', src.Value);
            else
                obj.env_gain = src.Value;
                obj.H.lbl_eg.String = sprintf('Environment  %.2f', src.Value);
            end
            obj.mic_mix_ = [];          % remix on next play
            obj.clean_   = [];
        end

        function cb_mixed_(obj, src)
            if logical(src.Value)
                if ~obj.build_mix_()
                    set(src, 'Value', 0); return;
                end
                obj.spec_mix_(:)   = 0;  obj.spec_col_mix_ = 1;
                obj.start_playback_(obj.play_mix_, 'mixed', src);
                src.String = sprintf('%c Stop Mixed', 9632);
                obj.set_status_(obj.status_text_('Playing mixed'), [0.40 0.80 0.95]);
            else
                obj.stop_playback_();
            end
        end

        function cb_clean_(obj, src)
            if logical(src.Value)
                if ~obj.build_mix_()
                    set(src, 'Value', 0); return;
                end
                [ready, why] = obj.enh.check();
                if ~ready
                    set(src, 'Value', 0);
                    warndlg(['ONNX runtime not available:' newline newline why], ...
                            'Q-WiSE');
                    obj.set_status_('ONNX runtime not available.', [0.95 0.45 0.20]);
                    return;
                end
                folder = '';
                if isempty(obj.clean_)
                    % First run for this mix: enhance + auto-record.
                    obj.set_status_(obj.status_text_('Running ONNX...'), ...
                                    [0.95 0.85 0.40]);
                    drawnow;
                    try
                        obj.clean_ = obj.enh.enhance(obj.mic_mix_);
                    catch ME
                        set(src, 'Value', 0);
                        obj.set_status_(sprintf('ONNX failed: %s', ME.message), ...
                                        [0.95 0.40 0.20]);
                        return;
                    end
                    folder = obj.audio.save_recording(obj.mic_mix_, obj.clean_);
                    obj.refresh_playback_();
                end

                obj.draw_clean_overlay_();      % gray noisy + green clean

                mono = obj.clean_;
                pk = max(abs(mono));
                if pk > 1e-9, mono = mono / pk * 0.97; end

                obj.spec_clean_(:) = 0;  obj.spec_col_clean_ = 1;
                obj.start_playback_(mono, 'clean', src);
                src.String = sprintf('%c Stop Clean', 9632);
                if ~isempty(folder)
                    obj.set_status_(obj.status_text_('Playing clean (recorded)'), ...
                                    [0.40 0.90 0.45]);
                end
            else
                obj.stop_playback_();
            end
        end
    end

    % ==================================================================
    %  PLAYBACK + LIVE ANIMATION
    % ==================================================================
    methods (Access = private)
        function start_playback_(obj, mono, mode, btn)
        %START_PLAYBACK_  Play `mono` from the start and animate the view
        %   according to `mode` ('sample' | 'mixed' | 'clean').
            obj.stop_playback_();
            mono = max(min(mono(:), 1), -1);
            if isempty(mono), return; end

            obj.mode_     = mode;
            obj.anim_sig_ = mono;
            obj.H.active_btn = btn;

            p = audioplayer(mono, obj.Cfg.fs);
            p.StopFcn = @(~,~) obj.on_play_finished_();
            obj.player_ = p;
            play(p);

            if ~strcmp(mode, 'sample')
                period = max(0.04, obj.Cfg.frame_size / obj.Cfg.fs);
                obj.anim_timer_ = timer('ExecutionMode', 'fixedRate', ...
                    'Period', period, 'BusyMode', 'drop', ...
                    'TimerFcn', @(~,~) obj.anim_tick_());
                start(obj.anim_timer_);
            end
        end

        function anim_tick_(obj)
        %ANIM_TICK_  Update waveform(s) + spectrogram for the playing clip.
            if ~isvalid(obj.fig) || isempty(obj.player_) || ...
               ~isvalid(obj.player_) || ~isplaying(obj.player_)
                return;
            end
            cfg = obj.Cfg;
            N   = cfg.frame_size;
            nb  = N/2 + 1;
            cs  = double(obj.player_.CurrentSample);

            switch obj.mode_
                case 'mixed'
                    L = size(obj.mic_mix_, 1);
                    hi = min(max(cs, 1), L);
                    [lo, hi] = obj.win_bounds_(hi, N, L);
                    for m = 1:cfg.n_mics
                        win = obj.frame_window_(obj.mic_mix_(:, m), lo, hi, N);
                        set(obj.H.hlines(m), 'YData', win / obj.mix_norm_);
                    end
                    obj.push_spec_('mix', obj.play_mix_, lo, hi, N, nb);

                case 'clean'
                    L = numel(obj.anim_sig_);
                    hi = min(max(cs, 1), L);
                    [lo, hi] = obj.win_bounds_(hi, N, L);
                    % Move the playhead along the full-clip overlay and
                    % scroll the clean spectrogram live.
                    if isfield(obj.H, 'clean_playhead') && ...
                            isvalid(obj.H.clean_playhead)
                        tnow = hi / obj.Cfg.fs;
                        set(obj.H.clean_playhead, 'XData', [tnow tnow]);
                    end
                    obj.push_spec_('clean', obj.anim_sig_, lo, hi, N, nb);
            end
            drawnow limitrate;
        end

        function draw_clean_overlay_(obj)
        %DRAW_CLEAN_OVERLAY_  Plot the full clip: gray noisy mic 1 behind,
        %   green clean on top (the comparison view), and reset the
        %   playhead to the start.
            if isempty(obj.clean_) || isempty(obj.mic_mix_), return; end
            fs    = obj.Cfg.fs;
            noisy = obj.mic_mix_(:, 1);
            clean = obj.clean_(:);
            tn = (0:numel(noisy)-1) / fs;
            tc = (0:numel(clean)-1) / fs;
            set(obj.H.hline_noisy, 'XData', tn, 'YData', noisy);
            set(obj.H.hline_clean, 'XData', tc, 'YData', clean);
            dur = max(tn(end), tc(end));
            if dur > 0
                xlim(obj.H.ax_clean_wave, [0 dur]);
            end
            ymax = max([1.0, max(abs(noisy)), max(abs(clean))]) * 1.05;
            ylim(obj.H.ax_clean_wave, [-ymax ymax]);
            set(obj.H.clean_playhead, 'XData', [0 0], 'YData', [-ymax ymax]);
        end

        function [lo, hi] = win_bounds_(~, hi, N, L)
            hi = min(hi, L);
            lo = max(1, hi - N + 1);
        end

        function win = frame_window_(~, sig, lo, hi, N)
            seg = sig(lo:hi);
            win = zeros(N, 1);
            win(N - numel(seg) + 1:N) = seg;     % right-align newest samples
        end

        function push_spec_(obj, kind, sig, lo, hi, N, nb)
            seg = sig(lo:hi);
            buf = zeros(N, 1);
            buf(N - numel(seg) + 1:N) = seg;
            w  = hann(N, 'periodic');
            X  = abs(fft(buf .* w, N));
            col = 20*log10(X(1:nb) + 1e-12);
            if strcmp(kind, 'mix')
                obj.spec_mix_(:, obj.spec_col_mix_) = col;
                obj.spec_col_mix_ = mod(obj.spec_col_mix_, obj.spec_ncols) + 1;
                set(obj.H.himg_mix, 'CData', obj.spec_mix_);
            else
                obj.spec_clean_(:, obj.spec_col_clean_) = col;
                obj.spec_col_clean_ = mod(obj.spec_col_clean_, obj.spec_ncols) + 1;
                set(obj.H.himg_clean, 'CData', obj.spec_clean_);
            end
        end

        function on_play_finished_(obj)
        %ON_PLAY_FINISHED_  audioplayer StopFcn — reset the active button.
            obj.kill_anim_timer_();
            if isfield(obj.H, 'active_btn') && ~isempty(obj.H.active_btn) ...
                    && isvalid(obj.H.active_btn)
                obj.reset_transport_btn_(obj.H.active_btn);
            end
            obj.H.active_btn = [];
            obj.player_      = [];
            obj.mode_        = '';
        end

        function stop_playback_(obj)
        %STOP_PLAYBACK_  Halt the transport player + animation and reset.
            obj.kill_anim_timer_();
            if ~isempty(obj.player_) && isvalid(obj.player_)
                try
                    if isplaying(obj.player_), stop(obj.player_); end
                catch
                end
            end
            if isfield(obj.H, 'active_btn') && ~isempty(obj.H.active_btn) ...
                    && isvalid(obj.H.active_btn)
                obj.reset_transport_btn_(obj.H.active_btn);
            end
            obj.H.active_btn = [];
            obj.player_      = [];
            obj.mode_        = '';
        end

        function kill_anim_timer_(obj)
            if ~isempty(obj.anim_timer_) && isvalid(obj.anim_timer_)
                try stop(obj.anim_timer_);   catch, end
                try delete(obj.anim_timer_); catch, end
            end
            obj.anim_timer_ = [];
        end

        function reset_transport_btn_(~, btn)
            btn.Value = 0;
            s = btn.String;
            if contains(s, 'Mixed')
                btn.String = sprintf('%c Play Mixed', 9658);
            elseif contains(s, 'Clean')
                btn.String = sprintf('%c Clean (ONNX)', 9658);
            else
                btn.String = sprintf('%c Listen', 9658);
            end
        end
    end

    % ==================================================================
    %  LATEST-RECORDING PLAYBACK GRID
    % ==================================================================
    methods (Access = private)
        function refresh_playback_(obj)
            obj.stop_grid_playback_();
            if isempty(obj.playback_panel_) || ~isvalid(obj.playback_panel_)
                return;
            end
            delete(obj.playback_panel_.Children);
            obj.playback_files_ = {};
            obj.playback_dir_   = '';

            rec_dir = obj.Cfg.record.dir;
            if ~isfolder(rec_dir)
                obj.render_grid_empty_('No recordings yet.'); return;
            end
            d = dir(rec_dir);
            d = d([d.isdir] & ~startsWith({d.name}, '.'));
            if isempty(d)
                obj.render_grid_empty_('No recordings yet.'); return;
            end
            [~, ord] = sort([d.datenum], 'descend');
            latest = fullfile(rec_dir, d(ord(1)).name);

            wavs = dir(fullfile(latest, '*.wav'));
            if isempty(wavs)
                obj.render_grid_empty_('(empty recording)'); return;
            end
            [~, ow] = sort({wavs.name});
            wavs = wavs(ow);
            obj.playback_dir_   = latest;
            obj.playback_files_ = arrayfun( ...
                @(w) fullfile(w.folder, w.name), wavs, 'UniformOutput', false);
            obj.render_grid_({wavs.name});
        end

        function render_grid_(obj, names)
            [~, leaf] = fileparts(obj.playback_dir_);
            uicontrol(obj.playback_panel_, 'Style', 'text', 'String', leaf, ...
                'FontSize', 7, 'BackgroundColor', [0.10 0.10 0.10], ...
                'ForegroundColor', [0.55 0.55 0.55], ...
                'HorizontalAlignment', 'left', ...
                'Units', 'normalized', 'Position', [0.02 0.90 0.96 0.08]);

            n      = numel(names);
            n_cols = min(2, max(1, n));
            n_rows = max(1, ceil(n / n_cols));
            gx0 = 0.03; gx1 = 0.97; gy0 = 0.04; gy1 = 0.86;
            cw  = (gx1 - gx0) / n_cols;
            ch  = (gy1 - gy0) / n_rows;
            px  = cw * 0.06;  py = ch * 0.10;

            for k = 1:n
                col = mod(k - 1, n_cols);
                row = floor((k - 1) / n_cols);
                xp = gx0 + col * cw + px;
                yp = gy1 - (row + 1) * ch + py;
                short = obj.short_label_(names{k});
                uicontrol(obj.playback_panel_, 'Style', 'togglebutton', ...
                    'String', sprintf('%c %s', 9658, short), ...
                    'FontSize', 8.5, ...
                    'BackgroundColor', [0.18 0.40 0.22], 'ForegroundColor', 'w', ...
                    'Units', 'normalized', 'Position', [xp yp cw-2*px ch-2*py], ...
                    'TooltipString', names{k}, ...
                    'UserData', struct('index', k, 'short', short, ...
                                       'name', names{k}), ...
                    'Callback', @(s,~) obj.cb_grid_play_(s));
            end
        end

        function s = short_label_(~, name)
            [~, stem] = fileparts(name);
            m = regexp(stem, '^mic0*(\d+)$', 'tokens', 'once');
            if ~isempty(m), s = sprintf('Mic %s', m{1}); else, s = upper(stem); end
        end

        function render_grid_empty_(obj, msg)
            uicontrol(obj.playback_panel_, 'Style', 'text', 'String', msg, ...
                'FontSize', 8, 'BackgroundColor', [0.10 0.10 0.10], ...
                'ForegroundColor', [0.55 0.55 0.55], ...
                'Units', 'normalized', 'Position', [0.02 0.42 0.96 0.20], ...
                'HorizontalAlignment', 'center');
        end

        function cb_grid_play_(obj, src)
            ud = src.UserData;
            if logical(src.Value)
                obj.stop_grid_playback_();
                try
                    [y, fs] = audioread(obj.playback_files_{ud.index});
                    if size(y, 2) > 1, y = mean(y, 2); end
                    p = audioplayer(y, fs);
                    p.StopFcn = @(~,~) obj.cb_grid_stopped_(src);
                    play(p);
                    obj.playback_player_     = p;
                    obj.playback_active_btn_ = src;
                    src.String = sprintf('%c %s', 9632, ud.short);
                    src.BackgroundColor = [0.78 0.10 0.10];
                catch ME
                    set(src, 'Value', 0);
                    warndlg(sprintf('Could not play %s:\n%s', ud.name, ME.message), ...
                            'Q-WiSE');
                end
            else
                obj.stop_grid_playback_();
            end
        end

        function cb_grid_stopped_(obj, src)
            if isvalid(src)
                ud = src.UserData;
                src.Value = 0;
                src.String = sprintf('%c %s', 9658, ud.short);
                src.BackgroundColor = [0.18 0.40 0.22];
            end
            obj.playback_player_     = [];
            obj.playback_active_btn_ = [];
        end

        function stop_grid_playback_(obj)
            if ~isempty(obj.playback_player_)
                try
                    if isvalid(obj.playback_player_) && isplaying(obj.playback_player_)
                        stop(obj.playback_player_);
                    end
                catch
                end
            end
            if ~isempty(obj.playback_active_btn_) && isvalid(obj.playback_active_btn_)
                ud = obj.playback_active_btn_.UserData;
                obj.playback_active_btn_.Value = 0;
                obj.playback_active_btn_.String = sprintf('%c %s', 9658, ud.short);
                obj.playback_active_btn_.BackgroundColor = [0.18 0.40 0.22];
            end
            obj.playback_player_     = [];
            obj.playback_active_btn_ = [];
        end

        function on_delete_(obj)
            try obj.stop_playback_();      catch, end
            try obj.stop_grid_playback_(); catch, end
            try obj.audio.release();       catch, end
        end
    end
end
