function tests = test_distance_to_gain()
%TEST_DISTANCE_TO_GAIN  Unit tests for the perChannel speech-level table.
    tests = functiontests(localfunctions);
end

function setupOnce(~) %#ok<*DEFNU>
    here = fileparts(fileparts(mfilename('fullpath')));
    addpath(fullfile(here, 'core'));
end

function test_band_below_one_metre(tc)
    verifyEqual(tc, distance_to_gain(0.10), 0.95, 'AbsTol', 1e-12);
    verifyEqual(tc, distance_to_gain(0.50), 0.95, 'AbsTol', 1e-12);
    verifyEqual(tc, distance_to_gain(1.00), 0.95, 'AbsTol', 1e-12);
end

function test_band_one_to_two(tc)
    verifyEqual(tc, distance_to_gain(1.001), 0.75, 'AbsTol', 1e-12);
    verifyEqual(tc, distance_to_gain(1.50),  0.75, 'AbsTol', 1e-12);
    verifyEqual(tc, distance_to_gain(2.00),  0.75, 'AbsTol', 1e-12);
end

function test_band_two_to_four(tc)
    verifyEqual(tc, distance_to_gain(2.50), 0.50, 'AbsTol', 1e-12);
    verifyEqual(tc, distance_to_gain(3.50), 0.50, 'AbsTol', 1e-12);
    verifyEqual(tc, distance_to_gain(4.00), 0.50, 'AbsTol', 1e-12);
end

function test_band_four_to_six(tc)
    verifyEqual(tc, distance_to_gain(4.50), 0.30, 'AbsTol', 1e-12);
    verifyEqual(tc, distance_to_gain(6.00), 0.30, 'AbsTol', 1e-12);
end

function test_band_six_to_eight(tc)
    verifyEqual(tc, distance_to_gain(7.00), 0.10, 'AbsTol', 1e-12);
    verifyEqual(tc, distance_to_gain(8.00), 0.10, 'AbsTol', 1e-12);
end

function test_band_eight_to_ten(tc)
    verifyEqual(tc, distance_to_gain(8.50),  0.00, 'AbsTol', 1e-12);
    verifyEqual(tc, distance_to_gain(10.00), 0.00, 'AbsTol', 1e-12);
end

function test_below_min_clamps_to_top_band(tc)
    verifyEqual(tc, distance_to_gain(0.05), 0.95, 'AbsTol', 1e-12);
    verifyEqual(tc, distance_to_gain(0.00), 0.95, 'AbsTol', 1e-12);
end

function test_beyond_ten_metres_is_zero(tc)
    verifyEqual(tc, distance_to_gain(10.5), 0.00, 'AbsTol', 1e-12);
    verifyEqual(tc, distance_to_gain(50.0), 0.00, 'AbsTol', 1e-12);
end

function test_vectorised_input(tc)
    d = [0.5, 1.5, 3.0, 5.0, 7.5, 9.0, 12.0];
    expected = [0.95, 0.75, 0.50, 0.30, 0.10, 0.00, 0.00];
    verifyEqual(tc, distance_to_gain(d), expected, 'AbsTol', 1e-12);
end

function test_shape_preserved(tc)
    d = reshape(linspace(0.10, 10.0, 12), [3 4]);
    g = distance_to_gain(d);
    verifyEqual(tc, size(g), size(d));
end

function test_band_edges_inclusive_on_upper(tc)
    % Each band is (lower, upper] except the first which is [0, 1.0].
    % Verify the upper edges land in the lower-gain band:
    verifyEqual(tc, distance_to_gain(1.0), 0.95);   % upper of band 1 (inclusive)
    verifyEqual(tc, distance_to_gain(2.0), 0.75);   % upper of band 2 (inclusive)
    verifyEqual(tc, distance_to_gain(4.0), 0.50);   % upper of band 3 (inclusive)
    verifyEqual(tc, distance_to_gain(6.0), 0.30);   % upper of band 4 (inclusive)
    verifyEqual(tc, distance_to_gain(8.0), 0.10);   % upper of band 5 (inclusive)
end
