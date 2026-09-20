"""Clean wheel install smoke test.

Builds the wheel, installs it in a fresh virtual environment, and runs the
offline commands the release must support. No network or provider calls.
"""

from __future__ import annotations

import subprocess
import sys
import venv
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _run(cmd: list[str], cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, check=False)


def _has_build() -> bool:
    try:
        import build  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


@pytest.mark.skipif(not _has_build(), reason="build not installed")
def test_wheel_clean_install_offline(tmp_path: Path) -> None:
    env_dir = tmp_path / "venv"
    venv.create(env_dir, with_pip=True, clear=True)
    bin_subdir = (
        Path("Scripts") / "python.exe" if sys.platform == "win32" else Path("bin") / "python"
    )
    python = env_dir / bin_subdir

    # Build wheel into a clean dist using the dev interpreter (has build)
    dist = tmp_path / "dist"
    dist.mkdir()
    build_env = {**__import__("os").environ}
    build_proc = _run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(dist)],
        cwd=ROOT,
        env=build_env,
    )
    assert build_proc.returncode == 0, build_proc.stderr
    wheels = list(dist.glob("*.whl"))
    assert wheels, "no wheel produced"

    # Install the wheel into the fresh venv
    install_proc = _run(
        [str(python), "-m", "pip", "install", str(wheels[0])], cwd=ROOT, env=build_env
    )
    assert install_proc.returncode == 0, install_proc.stderr

    clean_env = {**__import__("os").environ, "VIRTUAL_ENV": str(env_dir)}
    clean_env["PATH"] = (
        str(env_dir / ("Scripts" if sys.platform == "win32" else "bin")) + ":" + clean_env["PATH"]
    )

    def bruv(args: list[str]) -> subprocess.CompletedProcess[str]:
        return _run([str(python), "-m", "bruv", *args], cwd=ROOT, env=clean_env)

    assert bruv(["--help"]).returncode == 0
    assert bruv(["spec", "--output", "json"]).returncode == 0
    assert bruv(["schema", "output"]).returncode == 0

    demo = ROOT / "examples" / "funnel-audit-agent" / "audit.yaml"
    assert bruv(["validate", "-f", str(demo)]).returncode == 0

    install_skill = bruv(
        [
            "skill",
            "install",
            "--target",
            "pi",
            "--scope",
            "project",
            "--project-root",
            str(tmp_path),
        ]
    )
    assert install_skill.returncode == 0
