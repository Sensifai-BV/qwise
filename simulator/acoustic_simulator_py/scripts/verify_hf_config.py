#!/usr/bin/env python3
"""Pre-push checks for the HuggingFace Docker Space build.

HF Spaces only inspects three things at upload time:

  1. The YAML front-matter at the top of README.md (sdk, app_port, …).
  2. The Dockerfile at the repo root.
  3. Whatever the Dockerfile COPYs in — typically asset folders.

This script verifies each of those without needing Docker installed.
Run it before ``git push`` to catch the easy misconfigurations (missing
``app_port``, dropped ONNX model, broken ``USER 1000`` switch) that
otherwise only surface after the Space's build finishes.

Exit code 0 on success, 1 on any failure.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------- #
class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []

    def ok(self, msg: str) -> None:
        print(f"  ✓ {msg}")

    def fail(self, msg: str) -> None:
        print(f"  ✗ {msg}")
        self.failures.append(msg)

    def section(self, title: str) -> None:
        print(f"\n[{title}]")


# --------------------------------------------------------------------- #
# README front-matter
# --------------------------------------------------------------------- #
HF_REQUIRED_FIELDS = {
    "title":            str,
    "sdk":              str,
    "app_port":         int,
}


def _extract_front_matter(readme_text: str) -> dict[str, object]:
    m = re.match(
        r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n", readme_text, flags=re.DOTALL
    )
    if not m:
        return {}
    out: dict[str, object] = {}
    for raw in m.group(1).splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        v = v.strip().strip("\"'")
        if v.isdigit():
            out[k.strip()] = int(v)
        elif v.lower() in {"true", "false"}:
            out[k.strip()] = v.lower() == "true"
        else:
            out[k.strip()] = v
    return out


def check_readme(rep: Report) -> None:
    rep.section("README.md / HuggingFace Space front-matter")
    p = REPO_ROOT / "README.md"
    if not p.is_file():
        rep.fail("README.md not found at repo root")
        return
    fm = _extract_front_matter(p.read_text(encoding="utf-8"))
    if not fm:
        rep.fail("README.md is missing the YAML --- ... --- block")
        return

    for field, kind in HF_REQUIRED_FIELDS.items():
        if field not in fm:
            rep.fail(f"front-matter missing: {field}")
            continue
        if not isinstance(fm[field], kind):
            rep.fail(
                f"front-matter {field} should be {kind.__name__}, "
                f"got {type(fm[field]).__name__}"
            )
        else:
            rep.ok(f"{field}: {fm[field]!r}")

    if fm.get("sdk") != "docker":
        rep.fail(f'sdk must be "docker" (got {fm.get("sdk")!r})')
    if fm.get("app_port") != 7860:
        rep.fail(f"app_port must be 7860 (got {fm.get('app_port')!r})")


# --------------------------------------------------------------------- #
# Dockerfile
# --------------------------------------------------------------------- #
DOCKERFILE_REQUIRED = (
    ("FROM ",         "Dockerfile must declare a base image"),
    ("EXPOSE 7860",   "Dockerfile must EXPOSE 7860"),
    ("USER ",         "Dockerfile should switch to a non-root USER"),
    ("CMD",           "Dockerfile must define a CMD"),
)


def check_dockerfile(rep: Report) -> None:
    rep.section("Dockerfile")
    p = REPO_ROOT / "Dockerfile"
    if not p.is_file():
        rep.fail("Dockerfile not found at repo root")
        return
    text = p.read_text(encoding="utf-8")
    for needle, msg in DOCKERFILE_REQUIRED:
        if needle in text:
            rep.ok(f"contains '{needle}'")
        else:
            rep.fail(msg)

    # Soft checks — warn-only, not fatal.
    if "libsndfile1" not in text:
        rep.fail("libsndfile1 not installed — required by soundfile")
    if "uvicorn" not in text:
        rep.fail("Dockerfile CMD doesn't seem to call uvicorn")
    if "QWISE_DATA_DIR" not in text:
        rep.fail("QWISE_DATA_DIR not set — recordings will land in cwd")
    if "HEALTHCHECK" in text:
        rep.ok("HEALTHCHECK present")


# --------------------------------------------------------------------- #
# Bundled assets
# --------------------------------------------------------------------- #
REQUIRED_ASSETS = [
    "models/qwise_vad.onnx",
    "wavs/drone_fan.wav",
    "wavs/env_ambient.wav",
    "frontend/index.html",
    "frontend/static/styles.css",
    "frontend/static/app.js",
    "frontend/static/audio.js",
    "frontend/static/plots.js",
    "frontend/static/sidebar.js",
    "frontend/static/upload.js",
    "frontend/static/recordings.js",
    "frontend/static/session.js",
    "frontend/static/fft.js",
    "backend/api/app.py",
    "backend/config/default.py",
    "requirements.txt",
]


def check_assets(rep: Report) -> None:
    rep.section("Bundled assets")
    for relpath in REQUIRED_ASSETS:
        p = REPO_ROOT / relpath
        if p.is_file():
            kb = p.stat().st_size / 1024
            rep.ok(f"{relpath:<35} {kb:7.1f} KB")
        else:
            rep.fail(f"missing: {relpath}")


# --------------------------------------------------------------------- #
# pyproject.toml + requirements.txt sanity
# --------------------------------------------------------------------- #
def check_python_metadata(rep: Report) -> None:
    rep.section("Python metadata")
    py = REPO_ROOT / "pyproject.toml"
    if py.is_file():
        text = py.read_text(encoding="utf-8")
        if 'requires-python = ">=3.10' in text:
            rep.ok("requires-python = >=3.10")
        else:
            rep.fail("pyproject.toml: requires-python should be >=3.10")
    else:
        rep.fail("pyproject.toml not found")

    req = REPO_ROOT / "requirements.txt"
    if not req.is_file():
        rep.fail("requirements.txt not found")
        return
    text = req.read_text(encoding="utf-8")
    for pkg in ("fastapi", "uvicorn", "numpy", "scipy",
                "soundfile", "onnxruntime", "pydantic"):
        if pkg in text:
            rep.ok(f"requirements lists {pkg}")
        else:
            rep.fail(f"requirements.txt missing {pkg}")


# --------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------- #
def main() -> int:
    print(f"Q-WiSE HF Space build-config verifier")
    print(f"repo root: {REPO_ROOT}")
    rep = Report()
    check_readme(rep)
    check_dockerfile(rep)
    check_assets(rep)
    check_python_metadata(rep)

    print()
    if rep.failures:
        print(f"FAILED with {len(rep.failures)} issue(s):")
        for f in rep.failures:
            print(f"  - {f}")
        return 1
    print("PASS — image should build cleanly on HuggingFace.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
