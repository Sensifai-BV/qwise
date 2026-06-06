"""Pre-deploy gate — invokes ``scripts/verify_hf_config.py`` so the
test suite refuses to land a config that would fail an HF Space build.

The script is self-contained (no pytest dependencies, no import side
effects on the backend), so we shell out via ``subprocess`` instead of
importing it. Exit code 0 ⇒ the README front-matter, Dockerfile and
bundled assets all look correct.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFY = REPO_ROOT / "scripts" / "verify_hf_config.py"


def test_verify_hf_config_passes() -> None:
    assert VERIFY.is_file(), f"missing {VERIFY}"
    r = subprocess.run(
        [sys.executable, str(VERIFY)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
    assert r.returncode == 0, (
        f"verify_hf_config.py failed:\n{r.stdout}\n{r.stderr}"
    )
    assert "PASS" in r.stdout
