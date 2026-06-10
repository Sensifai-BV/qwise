function draw_drone(ax, pd, pm, cfg)
%DRAW_DRONE  3D quad-rotor body + rotor discs + mic markers in the scene axes.

    arm_l  = 0.19;  bw = 0.09;  bh = 0.045;
    blift  = 0.075; rr = 0.075;

    mz  = pm(1, 3);
    bz0 = mz + blift;
    bz1 = bz0 + bh;
    rz  = bz1 + 0.008;
    dx  = pd(1);  dy = pd(2);

    V = [dx-bw dy-bw bz0; dx+bw dy-bw bz0; dx+bw dy+bw bz0; dx-bw dy+bw bz0;
         dx-bw dy-bw bz1; dx+bw dy-bw bz1; dx+bw dy+bw bz1; dx-bw dy+bw bz1];
    F = [1 2 3 4; 5 6 7 8; 1 2 6 5; 3 4 8 7; 2 3 7 6; 1 4 8 5];
    patch(ax, 'Vertices', V, 'Faces', F, ...
          'FaceColor', [0.13 0.13 0.13], 'FaceAlpha', 0.88, ...
          'EdgeColor', 'k', 'EdgeAlpha', 0.85, 'LineWidth', 0.8);

    rcx = dx + arm_l * [ 1 -1  0  0];
    rcy = dy + arm_l * [ 0  0  1 -1];
    th  = linspace(0, 2*pi, 32);
    for k = 1:4
        plot3(ax, [dx rcx(k)], [dy rcy(k)], [bz1 rz], ...
              '-', 'Color', [0.22 0.22 0.22], 'LineWidth', 2.3);
        rxc = rcx(k) + rr * cos(th);
        ryc = rcy(k) + rr * sin(th);
        patch(ax, rxc, ryc, rz * ones(1, 32), [0.82 0.14 0.14], ...
              'FaceAlpha', 0.52, 'EdgeColor', [0.55 0.08 0.08], 'LineWidth', 0.8);
        plot3(ax, [rcx(k)-rr rcx(k)+rr], [rcy(k) rcy(k)], [rz rz], ...
              'k-', 'LineWidth', 1.2);
        plot3(ax, [rcx(k) rcx(k)], [rcy(k)-rr rcy(k)+rr], [rz rz], ...
              'k-', 'LineWidth', 1.2);
    end

    mc = {[0.00 0.78 0.32], [0.00 0.55 0.95], [1.00 0.52 0.00], ...
          [0.95 0.35 0.75], [0.60 0.85 0.20]};
    for m = 1:size(pm, 1)
        idx = mod(m-1, numel(mc)) + 1;
        scatter3(ax, pm(m,1), pm(m,2), pm(m,3), 60, mc{idx}, 'filled', 'o');
        text(pm(m,1), pm(m,2), pm(m,3) + 0.055, ...
             sprintf('M%d', m), 'FontSize', 7, 'Color', mc{idx}, 'Parent', ax);
    end

    bpf = round(cfg.drone_rpm * cfg.drone_blades / 60);
    text(dx + 0.11, dy, rz + 0.10, ...
         sprintf('BPF %dHz  AGL %.2fm', bpf, pd(3)), ...
         'FontSize', 7, 'FontWeight', 'bold', ...
         'Color', [0.90 0.18 0.18], 'Parent', ax);
end
