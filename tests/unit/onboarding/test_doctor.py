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
    version = next(r for r in results if r.name == "version")
    assert version.ok is True
    assert "bruv " in version.message
    config_result = next(r for r in results if r.name == "config")
    assert config_result.ok is True
    assert "backend=simple-jev" in config_result.message


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


# --- Needle backend diagnostics ---


def _needle_config() -> AppConfig:
    return AppConfig(backend="needle")  # type: ignore[arg-type]


def test_doctor_needle_credentials_not_required() -> None:
    results = run_doctor(
        config=_needle_config(), credentials=Credentials(), network_probe=lambda url: False
    )
    cred = next(r for r in results if r.name == "credentials")
    assert cred.ok is True
    assert "not required" in cred.message


def test_doctor_needle_endpoint_and_reachability_skipped() -> None:
    results = run_doctor(
        config=_needle_config(), credentials=Credentials(), network_probe=lambda url: False
    )
    for name in ("endpoint", "reachability"):
        item = next(r for r in results if r.name == name)
        assert item.ok is True
        assert "local runtime" in item.message


def test_doctor_needle_missing_dependency_reports_install_hint(monkeypatch) -> None:
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    results = run_doctor(
        config=_needle_config(), credentials=Credentials(), network_probe=lambda url: False
    )
    dep = next(r for r in results if r.name == "dependency")
    assert dep.ok is False
    assert "pip install 'bruv[needle]'" in dep.fix


def test_doctor_needle_dependency_installed(monkeypatch) -> None:
    import importlib.util
    import types

    fake = types.ModuleType("needle")
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name: fake if name == "needle" else None
    )
    results = run_doctor(
        config=_needle_config(), credentials=Credentials(), network_probe=lambda url: False
    )
    dep = next(r for r in results if r.name == "dependency")
    assert dep.ok is True


def test_doctor_needle_unsupported_platform(monkeypatch) -> None:
    import platform

    monkeypatch.setattr(platform, "system", lambda: "FreeBSD")
    monkeypatch.setattr(platform, "machine", lambda: "riscv64")
    results = run_doctor(
        config=_needle_config(), credentials=Credentials(), network_probe=lambda url: False
    )
    plat = next(r for r in results if r.name == "platform")
    assert plat.ok is False
    assert "FreeBSD" in plat.message


def test_doctor_needle_supported_platforms(monkeypatch) -> None:
    import platform

    for system, machine in (
        ("Linux", "x86_64"),
        ("Linux", "aarch64"),
        ("Darwin", "arm64"),
        ("Darwin", "x86_64"),
        ("Windows", "AMD64"),
        ("Windows", "ARM64"),
    ):
        monkeypatch.setattr(platform, "system", lambda s=system: s)
        monkeypatch.setattr(platform, "machine", lambda m=machine: m)
        results = run_doctor(
            config=_needle_config(), credentials=Credentials(), network_probe=lambda url: False
        )
        plat = next(r for r in results if r.name == "platform")
        assert plat.ok is True, (system, machine, plat.message)


def test_doctor_needle_cache_writable(monkeypatch, tmp_path) -> None:
    import platformdirs

    monkeypatch.setattr(platformdirs, "user_cache_dir", lambda *a, **kw: str(tmp_path / "cache"))
    results = run_doctor(
        config=_needle_config(), credentials=Credentials(), network_probe=lambda url: False
    )
    cache = next(r for r in results if r.name == "cache")
    assert cache.ok is True


def test_doctor_needle_cache_parent_missing(monkeypatch, tmp_path) -> None:
    import platformdirs

    monkeypatch.setattr(
        platformdirs, "user_cache_dir", lambda *a, **kw: str(tmp_path / "nope" / "cache")
    )
    results = run_doctor(
        config=_needle_config(), credentials=Credentials(), network_probe=lambda url: False
    )
    cache = next(r for r in results if r.name == "cache")
    assert cache.ok is False


def test_doctor_needle_cache_parent_unwritable(monkeypatch, tmp_path) -> None:
    import platformdirs

    monkeypatch.setattr(platformdirs, "user_cache_dir", lambda *a, **kw: str(tmp_path / "cache"))
    monkeypatch.setattr("bruv.onboarding.doctor.os.access", lambda *_a, **_kw: False)
    results = run_doctor(
        config=_needle_config(), credentials=Credentials(), network_probe=lambda url: False
    )
    cache = next(r for r in results if r.name == "cache")
    assert cache.ok is False


def test_doctor_needle_no_runtime_construction(monkeypatch) -> None:
    """Doctor must not construct the Needle runtime or import the package."""
    from bruv.backends import needle as needle_module

    def boom(*_a: object, **_kw: object) -> None:
        raise AssertionError("CactusNeedleRuntime constructed during doctor")

    monkeypatch.setattr(needle_module.CactusNeedleRuntime, "__init__", boom)
    results = run_doctor(
        config=_needle_config(), credentials=Credentials(), network_probe=lambda url: False
    )
    assert results
    assert all(isinstance(r.ok, bool) for r in results)
