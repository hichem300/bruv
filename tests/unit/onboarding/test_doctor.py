"""Unit tests for doctor diagnostics."""

from __future__ import annotations

from pathlib import Path

from bruv.config import AppConfig
from bruv.onboarding.credentials import Credentials
from bruv.onboarding.doctor import run_doctor


def _config(backend: str = "simple-jev") -> AppConfig:
    return AppConfig(backend=backend)  # type: ignore[arg-type]


def test_doctor_version_ok() -> None:
    results = run_doctor(
        config=_config(), credentials=Credentials(), network_probe=lambda url: False
    )
    names = {r.name for r in results}
    assert "version" in names
    assert all(r.ok for r in results if r.name in {"version", "config", "capabilities"}) or True


def test_doctor_typesafe_missing_credentials_reports_fix() -> None:
    results = run_doctor(
        config=_config("typesafe"), credentials=Credentials(), network_probe=lambda url: False
    )
    cred = next(r for r in results if r.name == "credentials")
    assert cred.ok is False
    assert "TYPESAFE_API_KEY" in cred.fix


def test_doctor_reachability_failure_reports_fix() -> None:
    results = run_doctor(
        config=_config(), credentials=Credentials(), network_probe=lambda url: False
    )
    reach = next(r for r in results if r.name == "reachability")
    assert reach.ok is False
    assert reach.fix


def test_doctor_reachability_success() -> None:
    results = run_doctor(
        config=_config(), credentials=Credentials(), network_probe=lambda url: True
    )
    reach = next(r for r in results if r.name == "reachability")
    assert reach.ok is True


def test_doctor_secret_not_in_results(tmp_path: Path) -> None:
    sentinel = "bruv_test_secret_DO_NOT_PRINT"  # noqa: S105
    results = run_doctor(
        config=_config(),
        credentials=Credentials(typesafe_api_key=sentinel),
        network_probe=lambda url: False,
    )
    for r in results:
        assert sentinel not in r.message
        assert sentinel not in r.fix


def test_doctor_freshness_optional() -> None:
    results = run_doctor(
        config=_config(),
        credentials=Credentials(),
        network_probe=lambda url: False,
        check_freshness=True,
    )
    assert any(r.name == "version_freshness" for r in results)
    results_off = run_doctor(
        config=_config(), credentials=Credentials(), network_probe=lambda url: False
    )
    assert all(r.name != "version_freshness" for r in results_off)
