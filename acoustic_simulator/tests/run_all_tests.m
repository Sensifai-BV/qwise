function results = run_all_tests()
%RUN_ALL_TESTS  Execute every test_*.m file in this folder.
%
%   results = run_all_tests();    returns a TestResult array.

    here = fileparts(mfilename('fullpath'));
    proj = fileparts(here);
    addpath(fullfile(proj, 'config'));
    addpath(fullfile(proj, 'core'));
    addpath(fullfile(proj, 'vad'));
    addpath(fullfile(proj, 'mwf'));

    import matlab.unittest.TestSuite
    suite = TestSuite.fromFolder(here);
    results = run(suite);
    disp(table(results));
end
