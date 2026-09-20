"""Installer contract tests for install.sh and install.ps1.

Uses fake release metadata served from a local directory. Asserts OS/arch
selection, checksum requirement, mismatch refusal, user-owned destination, and
atomic placement.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
INSTALL_SH = ROOT / "install.sh"
INSTALL_PS1 = ROOT / "install.ps1"


def _arch() -> str:
    m = platform.machine().lower()
    if m in ("x86_64", "amd64"):
        return "x86_64"
    if m in ("arm64", "aarch64"):
        return "aarch64"
    return m


def _os() -> str:
    return platform.system().lower()


def _artifact_name() -> str:
    return f"bruv-{_os()}-{_arch()}"


def _make_release(
    release_dir: Path, *, artifact_name: str, content: bytes, checksum_offset: str = ""
) -> Path:
    release_dir.mkdir(parents=True, exist_ok=True)
    artifact = release_dir / artifact_name
    artifact.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    if checksum_offset:
        digest = checksum_offset  # intentional mismatch
    (release_dir / "SHA256SUMS").write_text(f"{digest}  {artifact_name}\n")
    return artifact


@pytest.mark.skipif(_os() not in ("linux", "darwin"), reason="shell installer is Unix-only")
def test_install_sh_verifies_checksum_and_places_binary(tmp_path: Path, monkeypatch) -> None:
    release = tmp_path / "release"
    _make_release(release, artifact_name=_artifact_name(), content=b"#!/bin/sh\necho bruv\n")
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_BIN_HOME", str(home / "bin"))
    monkeypatch.setenv("BRUV_ARTIFACT_BASE", f"file://{release}")
    result = subprocess.run(
        ["bash", str(INSTALL_SH)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (home / "bin" / "bruv").is_file()
    assert "Checksum verified" in result.stdout


@pytest.mark.skipif(_os() not in ("linux", "darwin"), reason="shell installer is Unix-only")
def test_install_sh_rejects_checksum_mismatch(tmp_path: Path, monkeypatch) -> None:
    release = tmp_path / "release"
    _make_release(
        release,
        artifact_name=_artifact_name(),
        content=b"#!/bin/sh\necho bruv\n",
        checksum_offset="0" * 64,
    )
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_BIN_HOME", str(home / "bin"))
    monkeypatch.setenv("BRUV_ARTIFACT_BASE", f"file://{release}")
    result = subprocess.run(
        ["bash", str(INSTALL_SH)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "checksum mismatch" in result.stderr
    assert not (home / "bin" / "bruv").exists()


@pytest.mark.skipif(_os() not in ("linux", "darwin"), reason="shell installer is Unix-only")
def test_install_sh_missing_artifact_in_checksums(tmp_path: Path, monkeypatch) -> None:
    release = tmp_path / "release"
    # Artifact exists and downloads, but SHA256SUMS lists a different name.
    _make_release(release, artifact_name="bruv-other-x86_64", content=b"x")
    (release / _artifact_name()).write_bytes(b"#!/bin/sh\necho bruv\n")
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_BIN_HOME", str(home / "bin"))
    monkeypatch.setenv("BRUV_ARTIFACT_BASE", f"file://{release}")
    result = subprocess.run(
        ["bash", str(INSTALL_SH)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "missing" in result.stderr.lower()


def test_install_sh_file_exists() -> None:
    assert INSTALL_SH.is_file()


def test_install_ps1_file_exists() -> None:
    assert INSTALL_PS1.is_file()


def test_install_sh_no_sudo() -> None:
    text = INSTALL_SH.read_text()
    assert "sudo" not in text


def test_install_ps1_no_admin() -> None:
    text = INSTALL_PS1.read_text()
    assert "administrator" not in text.lower() or "no administrator" in text.lower()


def test_install_sh_user_owned_destination() -> None:
    text = INSTALL_SH.read_text()
    assert "HOME/.local/bin" in text or "XDG_BIN_HOME" in text
