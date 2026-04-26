function setup_silero_python(varargin)
%SETUP_SILERO_PYTHON  Configure MATLAB's pyenv to use the Q-WiSE venv.
%
%   setup_silero_python()
%     Creates <project>/.pyenv if missing, installs numpy + onnxruntime,
%     and points MATLAB's pyenv at the venv in OutOfProcess mode so
%     VADSilero can load Silero-VAD through Python's onnxruntime.
%
%   setup_silero_python('Rebuild', true)
%     Delete and re-create the venv from scratch.
%
%   IMPORTANT: pyenv can only be changed when Python is NOT yet loaded in
%   this MATLAB session.  If Python is already loaded (e.g. because any
%   py.* call has run), MATLAB will refuse to switch.  In that case:
%       - restart MATLAB,
%       - call ``setup_silero_python`` as the FIRST thing you do,
%       - then run the simulator.
%
%   R2025b officially supports Python 3.9–3.12.  Homebrew's Python 3.14 is
%   NOT supported, so this helper prefers /usr/bin/python3 (macOS 3.9.6),
%   then /opt/homebrew/opt/python@3.12/bin/python3.12, etc.

    p = inputParser;
    addParameter(p, 'Rebuild', false, @(x) islogical(x) || isnumeric(x));
    parse(p, varargin{:});
    do_rebuild = logical(p.Results.Rebuild);

    here      = fileparts(fileparts(mfilename('fullpath')));  % project root
    venv_dir  = fullfile(here, '.pyenv');
    venv_py   = fullfile(venv_dir, 'bin', 'python');

    % ---- 1. Pick a MATLAB-supported system Python --------------------
    cands = { ...
        '/opt/homebrew/opt/python@3.12/bin/python3.12', ...
        '/opt/homebrew/opt/python@3.11/bin/python3.11', ...
        '/opt/homebrew/opt/python@3.10/bin/python3.10', ...
        '/opt/homebrew/bin/python3.12', ...
        '/opt/homebrew/bin/python3.11', ...
        '/opt/homebrew/bin/python3.10', ...
        '/usr/bin/python3' ...
    };
    sys_py = '';
    for k = 1:numel(cands)
        if isfile(cands{k})
            sys_py = cands{k};
            break;
        end
    end
    if isempty(sys_py)
        error('QWiSE:Silero:NoSystemPython', ...
            ['No supported system Python (3.9–3.12) found. Install ' ...
             'via Homebrew: brew install python@3.12']);
    end

    % ---- 2. Create (or rebuild) the venv -----------------------------
    if do_rebuild && isfolder(venv_dir)
        fprintf('[Q-WiSE] Removing existing venv at %s\n', venv_dir);
        rmdir(venv_dir, 's');
    end
    if ~isfolder(venv_dir)
        fprintf('[Q-WiSE] Creating venv %s (using %s)\n', venv_dir, sys_py);
        [st, out] = system(sprintf('"%s" -m venv "%s" 2>&1', sys_py, venv_dir));
        if st ~= 0
            error('QWiSE:Silero:VenvFailed', ...
                'venv creation failed: %s', out);
        end
    end
    if ~isfile(venv_py)
        error('QWiSE:Silero:NoVenvPython', ...
            'Expected python binary not found: %s', venv_py);
    end

    % ---- 3. Install / upgrade deps inside the venv -------------------
    fprintf('[Q-WiSE] Ensuring numpy + onnxruntime are installed in venv...\n');
    [st, out] = system(sprintf( ...
        '"%s" -m pip install --upgrade pip 2>&1', venv_py));
    if st ~= 0, warning('QWiSE:Silero:PipUpgrade', 'pip upgrade: %s', out); end
    [st, out] = system(sprintf( ...
        '"%s" -m pip install --upgrade numpy onnxruntime 2>&1', venv_py));
    if st ~= 0
        error('QWiSE:Silero:PipInstall', ...
            'pip install failed:\n%s', out);
    end

    % ---- 4. Point MATLAB's pyenv at the venv -------------------------
    e = pyenv();
    if strcmpi(char(e.Status), 'Loaded') && ...
       ~strcmp(e.Executable, venv_py)
        error('QWiSE:Silero:PyEnvLocked', ...
            ['Python is already loaded in this MATLAB session\n' ...
             '  (current: %s,  status: %s).\n' ...
             'Restart MATLAB and call ''setup_silero_python'' as the ' ...
             'FIRST command so the env can be switched.'], ...
            e.Executable, char(e.Status));
    end

    e = pyenv('Version', venv_py, 'ExecutionMode', 'OutOfProcess');
    fprintf('[Q-WiSE] pyenv  : %s\n', e.Executable);
    fprintf('[Q-WiSE] version: %s   mode: %s\n', ...
        char(e.Version), char(e.ExecutionMode));

    % ---- 5. Smoke test -----------------------------------------------
    py.importlib.import_module('numpy');
    py.importlib.import_module('onnxruntime');
    fprintf('[Q-WiSE] numpy + onnxruntime load OK — Silero backend ready.\n');
end
