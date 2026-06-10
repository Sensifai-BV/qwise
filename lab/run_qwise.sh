#!/usr/bin/env bash
#
# run_qwise.sh — set up the Python venv for the Q-WiSE acoustic simulator
# and launch it in MATLAB.
#
#   1. check python3
#   2. create the venv  (lab/.pyenv)
#   3. install deps     (onnxruntime, numpy, soundfile)
#   4. run run_simulation.m via MATLAB -batch
#
# Usage:
#   ./run_qwise.sh                 # full setup + launch run_simulation
#   ./run_qwise.sh some_script     # launch a different .m script instead
#   SKIP_SETUP=1 ./run_qwise.sh    # skip venv/deps, just launch MATLAB
#   MATLAB_BIN=/path/to/matlab ./run_qwise.sh
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # the lab/ directory
VENV="$HERE/.pyenv"
VPY="$VENV/bin/python3"
DEPS=(onnxruntime numpy soundfile)
SCRIPT="${1:-run_simulation}"
SCRIPT="${SCRIPT%.m}"                                   # strip .m if given
MATLAB_BIN="${MATLAB_BIN:-/Applications/MATLAB_R2025b.app/bin/matlab}"

if [ "${SKIP_SETUP:-0}" != "1" ]; then
    # 1. base python ---------------------------------------------------
    PY="$(command -v python3 || true)"
    if [ -z "$PY" ]; then
        echo "ERROR: python3 not found on PATH." >&2
        exit 1
    fi
    echo "Base python : $PY ($("$PY" --version 2>&1))"

    # 2. venv ----------------------------------------------------------
    if [ ! -x "$VPY" ]; then
        echo "Creating venv: $VENV"
        "$PY" -m venv "$VENV"
    else
        echo "Venv exists : $VENV"
    fi

    # 3. deps ----------------------------------------------------------
    echo "Installing deps: ${DEPS[*]}"
    "$VPY" -m pip install --upgrade pip >/dev/null
    "$VPY" -m pip install "${DEPS[@]}"
    "$VPY" - <<'PYCHK'
import onnxruntime, numpy
print(f"  ok: onnxruntime {onnxruntime.__version__}, numpy {numpy.__version__}")
PYCHK
fi

# 4. run MATLAB --------------------------------------------------------
if [ ! -x "$MATLAB_BIN" ]; then
    echo "ERROR: MATLAB not found at: $MATLAB_BIN" >&2
    echo "       Set it, e.g.:  MATLAB_BIN=/Applications/MATLAB_R2025a.app/bin/matlab ./run_qwise.sh" >&2
    exit 1
fi

clear
echo "Running MATLAB script: $SCRIPT ..."
cd "$HERE"
"$MATLAB_BIN" -batch "$SCRIPT"
