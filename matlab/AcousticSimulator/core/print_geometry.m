function print_geometry(geo)
%PRINT_GEOMETRY  Pretty-print the scene geometry returned by build_geometry().
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
