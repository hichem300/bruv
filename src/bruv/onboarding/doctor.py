"""Doctor diagnostics.

Doctor never performs paid inference. Every failed item includes one
copy-pasteable fix. Secret values never enter ``DiagnosticResult``.
"""

from __future__ import annotations

import importlib.util
import os
import platform
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import platformdirs

from bruv import __version__
from bruv.backends.registry import backend_capabilities, get_backend_definition
from bruv.config import AppConfig, load_config
from bruv.domain.validation import BackendCapabilities
from bruv.onboarding.credentials import Credentials, credentials_path, load_credentials

NetworkProbe = Callable[[str], bool]


@dataclass(frozen=True, slots=True)
class DiagnosticResult:
    name: str
    ok: bool
    message: str
    fix: str = ""


def _check_version() -> DiagnosticResult:
    return DiagnosticResult(name="version", ok=True, message=f"bruv {__version__}")


def _check_config(config: AppConfig) -> DiagnosticResult:
    return DiagnosticResult(name="config", ok=True, message=f"backend={config.backend}")


def _check_credentials(creds: Credentials, backend: str) -> DiagnosticResult:
    from bruv.backends.registry import get_backend_definition

    if not get_backend_definition(backend).needs_credentials:
        return DiagnosticResult(name="credentials", ok=True, message=f"not required for {backend}")
    if creds.has_typesafe:
        return DiagnosticResult(name="credentials", ok=True, message="present")
    return DiagnosticResult(
        name="credentials",
        ok=False,
        message="no TypeSafe API key found",
        fix="Set TYPESAFE_API_KEY or run `bruv setup`.",
    )


def _check_credential_permissions(path: Path) -> DiagnosticResult:
    if not path.is_file():
        return DiagnosticResult(
            name="credential_permissions", ok=True, message="no credential file"
        )
    import stat

    mode = path.stat().st_mode
    if mode & (stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH):
        return DiagnosticResult(
            name="credential_permissions",
            ok=False,
            message="credential file is group/world readable",
            fix=f"Run `chmod 600 {path}`.",
        )
    return DiagnosticResult(name="credential_permissions", ok=True, message="restricted")


def _check_endpoint(config: AppConfig) -> DiagnosticResult:
    if get_backend_definition(config.backend).runs_local:
        return DiagnosticResult(name="endpoint", ok=True, message="skipped (local runtime)")
    if config.backend == "simple-jev":
        url = str(config.simple_jev_base_url)
    elif config.typesafe_endpoint is not None:
        url = str(config.typesafe_endpoint)
    else:
        return DiagnosticResult(
            name="endpoint",
            ok=True,
            message="using default TypeSafe endpoint",
        )
    if url.startswith("https://") or url.startswith("http://"):
        return DiagnosticResult(name="endpoint", ok=True, message=url)
    return DiagnosticResult(
        name="endpoint",
        ok=False,
        message=f"endpoint URL is not http(s): {url}",
        fix="Set a valid http(s):// endpoint in config.",
    )


def _check_reachability(config: AppConfig, probe: NetworkProbe) -> DiagnosticResult:
    if get_backend_definition(config.backend).runs_local:
        return DiagnosticResult(name="reachability", ok=True, message="skipped (local runtime)")
    if config.backend == "simple-jev":
        url = str(config.simple_jev_base_url) + "/health"
    elif config.typesafe_endpoint is not None:
        url = str(config.typesafe_endpoint)
    else:
        return DiagnosticResult(name="reachability", ok=True, message="skipped (hosted default)")
    try:
        ok = probe(url)
    except Exception:  # noqa: BLE001
        ok = False
    if ok:
        return DiagnosticResult(name="reachability", ok=True, message="reachable")
    return DiagnosticResult(
        name="reachability",
        ok=False,
        message=f"could not reach {url}",
        fix="Check that the backend is running and the URL is correct.",
    )


def _check_capabilities(capabilities: BackendCapabilities) -> DiagnosticResult:
    return DiagnosticResult(
        name="capabilities",
        ok=True,
        message=f"types={sorted(capabilities.question_types)} calibrated={capabilities.calibrated}",
    )


def _check_dependency(backend: str) -> DiagnosticResult:
    """Check optional packages without importing them or building a runtime."""
    from bruv.backends.registry import get_backend_definition

    definition = get_backend_definition(backend)
    if definition.runtime_packages:
        missing = [
            package
            for package in definition.runtime_packages
            if importlib.util.find_spec(package) is None
        ]
        if not missing:
            return DiagnosticResult(name="dependency", ok=True, message="installed")
        hint = definition.install_hint or f"Install the {backend} dependency."
        return DiagnosticResult(
            name="dependency",
            ok=False,
            message=f"missing runtime packages: {', '.join(missing)}",
            fix=hint,
        )
    module = definition.optional_module
    if module is None:
        return DiagnosticResult(name="dependency", ok=True, message="no optional dependency")
    if importlib.util.find_spec(module) is not None:
        return DiagnosticResult(name="dependency", ok=True, message="installed")
    optional_hint = get_backend_definition(backend).install_hint
    return DiagnosticResult(
        name="dependency",
        ok=False,
        message=f"optional dependency {module!r} is not installed",
        fix=optional_hint if optional_hint else f"Install the {backend} dependency.",
    )


def _check_platform(backend: str) -> DiagnosticResult:
    if not get_backend_definition(backend).runs_local:
        return DiagnosticResult(name="platform", ok=True, message="skipped")
    system = platform.system()
    machine = platform.machine().lower()
    if machine in ("amd64", "x86_64"):
        arch = "x86_64"
    elif machine in ("arm64", "aarch64"):
        arch = "arm64"
    else:
        arch = ""
    if system in ("Linux", "Darwin", "Windows") and arch:
        return DiagnosticResult(
            name="platform", ok=True, message=f"supported platform ({system} {arch})"
        )
    return DiagnosticResult(
        name="platform",
        ok=False,
        message=f"unsupported platform ({system} {machine or 'unknown'})",
        fix=(
            f"{backend} supports Linux, macOS, and Windows on x86_64 and arm64. "
            "Use the typesafe or simple-jev backend on this platform."
        ),
    )


def _check_cache_writable(backend: str) -> DiagnosticResult:
    """Check the cache parent directory without triggering any download."""
    if not get_backend_definition(backend).runs_local:
        return DiagnosticResult(name="cache", ok=True, message="skipped")
    cache_dir = Path(platformdirs.user_cache_dir("bruv", appauthor=False))
    parent = cache_dir.parent
    if not parent.is_dir():
        return DiagnosticResult(
            name="cache",
            ok=False,
            message=f"cache parent directory does not exist: {parent}",
            fix=f"Create {parent} or fix your platform cache directory.",
        )
    if not os.access(parent, os.W_OK):
        return DiagnosticResult(
            name="cache",
            ok=False,
            message=f"cache parent directory is not writable: {parent}",
            fix=f"Grant write access to {parent} or fix your platform cache directory.",
        )
    return DiagnosticResult(name="cache", ok=True, message=f"writable ({parent})")


def _check_rlcd_cache(backend: str) -> DiagnosticResult:
    """Check pinned RLCD cache status without network, download, or model load."""
    if backend != "rlcd-modernbert":
        return DiagnosticResult(name="rlcd_cache", ok=True, message="skipped")
    from bruv.backends.rlcd_artifacts import (
        STATUS_DEPENDENCY_MISSING,
        STATUS_HASH_MISMATCH,
        STATUS_MISSING,
        STATUS_VERIFIED,
        cached_artifact_status,
    )

    statuses = cached_artifact_status()
    detail = "; ".join(f"{name}: {status}" for name, status in statuses.items())
    if any(status == STATUS_DEPENDENCY_MISSING for status in statuses.values()):
        return DiagnosticResult(
            name="rlcd_cache",
            ok=True,
            message="skipped (optional rlcd-modernbert dependency not installed; "
            "see the dependency check)",
        )
    if any(status == STATUS_HASH_MISMATCH for status in statuses.values()):
        return DiagnosticResult(
            name="rlcd_cache",
            ok=False,
            message=f"cached RLCD artifact failed verification ({detail})",
            fix=(
                "Delete the corrupt cached artifact, then run one evaluation again so "
                "it is re-downloaded and verified."
            ),
        )
    if any(status == STATUS_MISSING for status in statuses.values()):
        return DiagnosticResult(
            name="rlcd_cache",
            ok=False,
            message=f"cached RLCD artifact missing ({detail})",
            fix=(
                "Run one evaluation with the rlcd-modernbert backend while online so the "
                "pinned artifacts (about 606 MB) download to the local cache and verify."
            ),
        )
    if all(status == STATUS_VERIFIED for status in statuses.values()):
        return DiagnosticResult(name="rlcd_cache", ok=True, message=f"verified ({detail})")
    return DiagnosticResult(
        name="rlcd_cache",
        ok=False,
        message=f"unexpected RLCD cache status ({detail})",
        fix=(
            "Reinstall or update the pinned rlcd-modernbert backend with "
            "pip install 'bruv[rlcd-modernbert]', then run one evaluation again."
        ),
    )


def _check_version_freshness(*, enabled: bool) -> DiagnosticResult:
    if not enabled:
        return DiagnosticResult(
            name="version_freshness", ok=True, message="skipped (not requested)"
        )
    return DiagnosticResult(name="version_freshness", ok=True, message=f"bruv {__version__}")


def run_doctor(
    *,
    config: AppConfig | None = None,
    credentials: Credentials | None = None,
    credential_file: Path | None = None,
    network_probe: NetworkProbe | None = None,
    check_freshness: bool = False,
) -> list[DiagnosticResult]:
    """Run all diagnostics without any paid inference."""
    cfg = config if config is not None else load_config(env={})
    creds = credentials if credentials is not None else load_credentials(env={})
    cred_path = credential_file if credential_file is not None else credentials_path()
    probe = network_probe if network_probe is not None else (lambda url: False)

    results: list[DiagnosticResult] = [
        _check_version(),
        _check_config(cfg),
        _check_credentials(creds, cfg.backend),
        _check_credential_permissions(cred_path),
        _check_endpoint(cfg),
        _check_reachability(cfg, probe),
        _check_dependency(cfg.backend),
        _check_platform(cfg.backend),
        _check_cache_writable(cfg.backend),
        _check_rlcd_cache(cfg.backend),
        _check_capabilities(backend_capabilities(cfg.backend)),
    ]
    if check_freshness:
        results.append(_check_version_freshness(enabled=True))
    return results


__all__ = ["DiagnosticResult", "run_doctor"]
