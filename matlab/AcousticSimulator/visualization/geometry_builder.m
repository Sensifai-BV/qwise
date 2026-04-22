function geometry_builder(ax, geo, cfg)
%GEOMETRY_BUILDER  Thin alias: render the static 3D scene via draw_scene.
%
%   Kept as a separate entry point so external callers that already refer
%   to the old name continue to work.
    draw_scene(ax, geo, cfg);
end
