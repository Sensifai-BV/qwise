function draw_scene(ax, geo, cfg)
%DRAW_SCENE  Render the static 3D acoustic scene into axes `ax`.
%   The view is centred on the human + drone so the speaker and the drone
%   stay prominent and clearly move when the scene preset changes. The
%   far-off environment-noise source does NOT drive the axis bounds; if it
%   falls outside the view it is shown as a labelled direction arrow.
    hold(ax, 'on'); grid(ax, 'on'); box(ax, 'on');

    margin = 0.6;

    % Bounds come from the human, drone, mic array and ground image only —
    % the env source (typically 8 m away) is handled separately so it can
    % never shrink the human/drone to a dot in the corner.
    key   = [geo.pos_human; geo.pos_drone; geo.pos_mics; geo.pos_img_src];
    focus = (geo.pos_human + geo.pos_drone) / 2;       % view centre

    rxy  = max(vecnorm(key(:,1:2) - focus(1:2), 2, 2));
    half = max(rxy + margin, 1.5);
    xr = focus(1) + [-half, half];
    yr = focus(2) + [-half, half];

    zr = [min(key(:,3)) - margin, max(key(:,3)) + margin];
    zr(1) = min(zr(1), -0.2);
    zr(2) = max(zr(2),  0.5);

    set(ax, 'DataAspectRatio', [1 1 1], 'PlotBoxAspectRatio', [1.1 1.1 0.9]);
    set(ax, 'XLim', xr, 'YLim', yr, 'ZLim', zr);

    % Asphalt ground spans the visible X/Y window.
    [gx, gy] = meshgrid(linspace(xr(1), xr(2), 10), ...
                        linspace(yr(1), yr(2), 10));
    surf(ax, gx, gy, zeros(size(gx)), ...
         'FaceColor', [0.27 0.27 0.27], 'FaceAlpha', 0.25, ...
         'EdgeColor', [0.42 0.42 0.42], 'EdgeAlpha', 0.12);

    % Human body / mouth
    xh = geo.pos_human(1);  yh = geo.pos_human(2);
    Hh = cfg.human_height;
    plot3(ax, [xh xh xh], [yh yh yh], [0 Hh*0.52 Hh*0.87], ...
          'Color', [0.30 0.45 0.90], 'LineWidth', 2.5);
    th = linspace(0, 2*pi, 36);  rh = 0.08;
    plot3(ax, xh + rh*cos(th), yh + zeros(1,36), Hh*0.935 + rh*sin(th), ...
          'Color', [0.30 0.45 0.90], 'LineWidth', 2.0);
    scatter3(ax, xh, yh, geo.pos_human(3), 65, [0.30 0.55 1.0], 'filled', '^');
    text(xh+0.09, yh, geo.pos_human(3)+0.09, ...
         sprintf('Mouth %.2fm', geo.pos_human(3)), ...
         'FontSize', 7, 'Color', [0.40 0.65 1.0], 'Parent', ax);
    text(xh-0.08, yh, Hh+0.14, sprintf('%.0fcm', Hh*100), ...
         'FontSize', 7, 'FontWeight', 'bold', ...
         'Color', [0.40 0.55 0.90], 'Parent', ax);

    % Image source (for ground reflection)
    scatter3(ax, geo.pos_img_src(1), geo.pos_img_src(2), geo.pos_img_src(3), ...
             40, [0.50 0.50 0.50], 'd', 'LineWidth', 1.2);
    plot3(ax, [xh xh], [yh yh], [geo.pos_human(3) geo.pos_img_src(3)], ...
          ':', 'Color', [0.52 0.52 0.52], 'LineWidth', 0.9);

    % Drone body + mic array
    draw_drone(ax, geo.pos_drone, geo.pos_mics, cfg);

    % Direct human -> drone path with distance label
    d = norm(geo.pos_drone - geo.pos_human);
    plot3(ax, [xh geo.pos_drone(1)], [yh geo.pos_drone(2)], ...
          [geo.pos_human(3) geo.pos_drone(3)], ...
          '--', 'Color', [0.30 0.55 1.0], 'LineWidth', 1.5);
    mid = (geo.pos_human + geo.pos_drone)/2;
    text(mid(1)+0.07, mid(2), mid(3)+0.10, sprintf('%.2f m', d), ...
         'FontSize', 8, 'FontWeight', 'bold', ...
         'Color', [0.55 0.75 1.0], 'Parent', ax);

    % Environment-noise source: plot in place if inside the view, else
    % show a direction arrow + distance at the view edge.
    env = geo.pos_env;
    in_view = env(1) >= xr(1) && env(1) <= xr(2) && ...
              env(2) >= yr(1) && env(2) <= yr(2);
    d_env = norm(env - geo.pos_human);
    if in_view
        scatter3(ax, env(1), env(2), env(3), 75, [0.65 0.10 0.85], 'p', 'filled');
        text(env(1)+0.09, env(2), env(3)+0.11, sprintf('Env noise %.1f m', d_env), ...
             'FontSize', 7, 'Color', [0.75 0.30 0.95], 'Parent', ax);
    else
        dir = env(1:2) - focus(1:2);
        dir = dir / (norm(dir) + 1e-9);
        tip = focus(1:2) + dir * half * 0.93;
        base = focus(1:2) + dir * half * 0.55;
        plot3(ax, [base(1) tip(1)], [base(2) tip(2)], [0.05 0.05], ...
              '-', 'Color', [0.65 0.10 0.85], 'LineWidth', 2.0);
        scatter3(ax, tip(1), tip(2), 0.05, 90, [0.75 0.30 0.95], 'p', 'filled');
        text(base(1), base(2), 0.18, sprintf('Env noise %.1f m \\rightarrow', d_env), ...
             'FontSize', 7, 'FontWeight', 'bold', ...
             'Color', [0.75 0.30 0.95], 'Parent', ax);
    end

    xlabel(ax, 'X (m)'); ylabel(ax, 'Y (m)'); zlabel(ax, 'Z (m)');
    title(ax, 'Acoustic Scene  (outdoor / asphalt)', ...
          'Color', 'w', 'FontSize', 9, 'FontWeight', 'bold');
    view(ax, 38, 24);
    camproj(ax, 'perspective');
end
