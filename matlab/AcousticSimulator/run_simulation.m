function run_simulation()
%RUN_SIMULATION  Q-WiSE Acoustic Simulator entry point.
%
%   Adds project folders to the MATLAB path, loads the default config,
%   wires up the core/VAD/MWF/visualization modules, and launches the
%   real-time UI.
%
%   Edit `config/default.m` to reshape any runtime parameter.

    setup_paths();

    cfg   = default();
    geo   = build_geometry(cfg);
    print_geometry(geo);

    audio   = AudioIO(cfg);
    vad_obj = vad(cfg);
    mwf_obj = mwf(cfg);

    ui = SimulatorUI(cfg, geo, audio, vad_obj, mwf_obj);
    ui.start();
end


function setup_paths()
    here = fileparts(mfilename('fullpath'));
    addpath(fullfile(here, 'config'));
    addpath(fullfile(here, 'core'));
    addpath(fullfile(here, 'vad'));
    addpath(fullfile(here, 'mwf'));
    addpath(fullfile(here, 'visualization'));
    addpath(fullfile(here, 'tests'));
end
