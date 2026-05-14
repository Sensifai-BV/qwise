function draw_scene(ax, geo, cfg)
%DRAW_SCENE  Render the static 3D acoustic scene into axes `ax`.
%   Axis limits are computed from the union of all visible objects
%   (human, drone, env, mics, image source) so the scene stays centred
%   no matter how far the env source or drone move. Equal X/Y span
%   keeps top-down distances readable.
    hold(ax, 'on'); grid(ax, 'on'); box(ax, 'on');

    % Collect every position that will be plotted so the bounds enclose
    % them all with a uniform margin.
    pts = [geo.pos_human;
           geo.pos_drone;
           geo.pos_env;
           geo.pos_mics;
           geo.pos_img_src];

    margin = 0.6;
    xr = [min(pts(:,1))-margin, max(pts(:,1))+margin];
    yr = [min(pts(:,2))-margin, max(pts(:,2))+margin];
    zr = [min(pts(:,3))-margin, max(pts(:,3))+margin];

    % Force x/y to share the same half-span so the scene doesn't stretch
    % asymmetrically when the env source sits far off-axis.
    cx = mean(xr);  cy = mean(yr);
    half_xy = max(diff(xr), diff(yr)) / 2;
    half_xy = max(half_xy, 1.5);                % minimum so a tight scene
                                                 % still feels readable.
    xr = [cx - half_xy, cx + half_xy];
    yr = [cy - half_xy, cy + half_xy];

    % Z gets a tighter range, but always show the ground plane.
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
    text(xr(1)+0.3, yr(1)+0.3, 0.02, sprintf('Asphalt (R=%.2f)', cfg.ground_R), ...
         'FontSize', 7, 'Color', [0.48 0.48 0.48], 'Parent', ax);

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
    text(geo.pos_img_src(1)+0.08, geo.pos_img_src(2), geo.pos_img_src(3)-0.13, ...
         sprintf('img z=%.2f', geo.pos_img_src(3)), ...
         'FontSize', 7, 'Color', [0.50 0.50 0.50], 'Parent', ax);

    % Drone body
    draw_drone(ax, geo.pos_drone, geo.pos_mics, cfg);

    % Env noise source
    scatter3(ax, geo.pos_env_noise(1), geo.pos_env_noise(2), geo.pos_env_noise(3), ...
             75, [0.65 0.10 0.85], 'p', 'filled');
    text(geo.pos_env_noise(1)+0.09, geo.pos_env_noise(2), geo.pos_env_noise(3)+0.11, ...
         'Env Noise', 'FontSize', 7, 'Color', [0.65 0.10 0.85], 'Parent', ax);

    % Direct path
    d = norm(geo.pos_drone - geo.pos_human);
    plot3(ax, [xh geo.pos_drone(1)], [yh geo.pos_drone(2)], ...
          [geo.pos_human(3) geo.pos_drone(3)], ...
          '--', 'Color', [0.30 0.55 1.0], 'LineWidth', 1.5);
    mid = (geo.pos_human + geo.pos_drone)/2;
    text(mid(1)+0.07, mid(2), mid(3)+0.08, sprintf('%.2fm', d), ...
         'FontSize', 7, 'Color', [0.40 0.65 1.0], 'Parent', ax);

    % Reflected path via centre mic
    img = geo.pos_img_src;
    mc2 = geo.pos_mics(ceil(size(geo.pos_mics,1)/2), :);
    tsp = -img(3)/(mc2(3) - img(3));
    sp  = img + tsp*(mc2 - img);
    plot3(ax, [xh sp(1)], [yh sp(2)], [geo.pos_human(3) sp(3)], ...
          '-', 'Color', [0.10 0.72 0.45], 'LineWidth', 1.2);
    plot3(ax, [sp(1) mc2(1)], [sp(2) mc2(2)], [sp(3) mc2(3)], ...
          '-', 'Color', [0.10 0.72 0.45], 'LineWidth', 1.2);
    scatter3(ax, sp(1), sp(2), sp(3), 35, [0.10 0.62 0.38], 's', 'LineWidth', 1.2);
    text(sp(1)+0.06, sp(2), sp(3)+0.07, sprintf('R=%.2f', cfg.ground_R), ...
         'FontSize', 7, 'Color', [0.10 0.72 0.45], 'Parent', ax);

    % Env → drone
    plot3(ax, [geo.pos_env_noise(1) geo.pos_drone(1)], ...
              [geo.pos_env_noise(2) geo.pos_drone(2)], ...
              [geo.pos_env_noise(3) geo.pos_drone(3)], ...
          ':', 'Color', [0.60 0.10 0.80], 'LineWidth', 0.9);

    xlabel(ax, 'X(m)'); ylabel(ax, 'Y(m)'); zlabel(ax, 'Z(m)');
    title(ax, 'Acoustic Scene  (outdoor / asphalt)', ...
          'Color', 'w', 'FontSize', 9, 'FontWeight', 'bold');
    view(ax, 38, 24);
    camproj(ax, 'perspective');
end
