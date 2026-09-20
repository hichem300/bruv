"""Build standalone one-file artifacts for the release matrix.

Usage:
    python packaging/build_standalone.py

Builds a PyInstaller one-file bundle for the current platform. The release
workflow calls this once per OS/architecture runner.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "packaging" / "bruv.spec"
DIST = ROOT / "dist"


def target_name() -> str:
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        arch = "x86_64"
    elif machine in ("arm64", "aarch64"):
        arch = "aarch64"
    else:
        arch = machine
    system = platform.system().lower()
    ext = ".exe" if system == "windows" else ""
    return f"bruv-{system}-{arch}{ext}"


def build() -> Path:
    DIST.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            str(SPEC),
            "--onefile",
            "--noconfirm",
            "--distpath",
            str(DIST),
        ],
        cwd=ROOT,
        check=True,
    )
    produced = DIST / ("bruv.exe" if platform.system().lower() == "windows" else "bruv")
    if not produced.is_file():
        raise FileNotFoundError(f"build did not produce {produced}")
    target = DIST / target_name()
    shutil.move(str(produced), str(target))
    return target


if __name__ == "__main__":
    artifact = build()
    print(f"built: {artifact}")
