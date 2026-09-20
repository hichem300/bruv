"""Doctor diagnostics.

Doctor never performs paid inference. Every failed item includes one
copy-pasteable fix. Secret values never enter ``DiagnosticResult``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from bruv import __version__
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
    if backend != "typesafe":
        return DiagnosticResult(name="credentials", ok=True, message="not required for simple-jev")
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
        fix="Set a valid http(s:// endpoint in config.",
    )


def _check_reachability(config: AppConfig, probe: NetworkProbe) -> DiagnosticResult:
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
    ]
    if check_freshness:
        results.append(_check_version_freshness(enabled=True))
    return results


__all__ = ["DiagnosticResult", "run_doctor"]
