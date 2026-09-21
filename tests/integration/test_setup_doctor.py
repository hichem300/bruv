"""Integration tests for `bruv setup` and `bruv doctor`."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from bruv.cli import app
from bruv.config import AppConfig
from bruv.onboarding.credentials import Credentials
from bruv.onboarding.setup import run_setup

runner = CliRunner()


def test_setup_non_tty_exits_two() -> None:
    result = runner.invoke(app, ["setup"])
    assert result.exit_code == 2
    assert "non-interactive" in result.output.lower()


def test_setup_typesafe_persists(monkeypatch, tmp_path: Path) -> None:
    saved: dict[str, Path] = {}
    monkeypatch.setattr(
        "bruv.onboarding.setup.save_credentials",
        lambda key, path=None: saved.setdefault("path", tmp_path / "creds"),
    )
    prompts = iter(["typesafe", "secret-key"])

    def prompt(msg: str) -> str:
        return next(prompts)

    result = run_setup(
        prompt=prompt,
        confirm=lambda msg: True,
        print_line=lambda msg: None,
        env={},
        credential_path=tmp_path / "creds",
    )
    assert result.backend == "typesafe"
    assert result.persisted is True
    assert saved["path"] == tmp_path / "creds"


def test_setup_simple_jev() -> None:
    prompts = iter(["simple-jev", "http://127.0.0.1:8000"])

    def prompt(msg: str) -> str:
        return next(prompts)

    result = run_setup(
        prompt=prompt,
        confirm=lambda msg: True,
        print_line=lambda msg: None,
        env={},
    )
    assert result.backend == "simple-jev"
    assert result.persisted is False


def test_setup_invalid_backend_raises() -> None:
    import pytest

    with pytest.raises(Exception):  # noqa: B017
        run_setup(
            prompt=lambda msg: "bogus",
            confirm=lambda msg: True,
            print_line=lambda msg: None,
            env={},
        )


def test_setup_needle() -> None:
    lines: list[str] = []
    result = run_setup(
        prompt=lambda msg: "needle",
        confirm=lambda msg: True,
        print_line=lines.append,
        env={},
    )
    assert result.backend == "needle"
    assert result.persisted is False
    assert result.next_command == "bruv doctor"
    joined = "\n".join(lines)
    assert "pip install 'bruv[needle]'" in joined
    assert "35 MB" in joined
    # Needle must never ask for a credential.
    assert "API key (input hidden)" not in joined


def test_setup_choices_come_from_registry() -> None:
    lines: list[str] = []
    try:
        run_setup(
            prompt=lambda msg: "needle",
            confirm=lambda msg: True,
            print_line=lines.append,
            env={},
        )
    except Exception:  # noqa: BLE001, S110 - menu output is the target
        pass
    from bruv.backends.registry import backend_names

    menu = "\n".join(lines)
    for name in backend_names:
        assert name in menu


def test_doctor_runs_and_reports(monkeypatch) -> None:
    monkeypatch.setattr(
        "bruv.onboarding.doctor.load_config", lambda env={}: AppConfig(backend="simple-jev")
    )
    monkeypatch.setattr(
        "bruv.onboarding.doctor.load_credentials",
        lambda env={}: Credentials(),
    )
    monkeypatch.setattr(
        "bruv.onboarding.doctor.credentials_path", lambda: Path("/var/empty/bruv-creds")
    )
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code in (0, 1)
    assert "version" in result.output
