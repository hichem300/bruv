"""Guided setup service.

Prompting, filesystem, environment, and network probing are injected so the
service is testable without a TTY or live endpoints. Secrets never enter
diagnostic output.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from bruv.application import ConfigurationError
from bruv.onboarding.credentials import CREDENTIAL_ENV, save_credentials

PromptFn = Callable[[str], str]
ConfirmFn = Callable[[str], bool]
PrintFn = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class SetupResult:
    backend: str
    persisted: bool
    next_command: str


def run_setup(
    *,
    prompt: PromptFn,
    confirm: ConfirmFn,
    print_line: PrintFn,
    env: Mapping[str, str],
    config_path: Path | None = None,
    credential_path: Path | None = None,
) -> SetupResult:
    """Interactive setup. Non-interactive callers should refuse before this."""
    print_line("bruv setup: choose a backend.")
    print_line("  typesafe: hosted, calibrated. Needs TYPESAFE_API_KEY.")
    print_line("  simple-jev: local/self-hosted, uncalibrated. No key needed.")
    backend = prompt("Backend [typesafe|simple-jev]").strip().lower()
    if backend not in ("typesafe", "simple-jev"):
        raise ConfigurationError(
            message="backend must be typesafe or simple-jev",
            paid_request=False,
            action="Re-run setup and choose a supported backend.",
        )

    if backend == "typesafe":
        api_key = prompt("TypeSafe API key (input hidden)")
        if not api_key.strip():
            raise ConfigurationError(
                message="an API key is required for the typesafe backend",
                paid_request=False,
                action=f"Set {CREDENTIAL_ENV} or re-run setup with a key.",
            )
        if not confirm("Persist the key to a protected credential file?"):
            print_line(f"Skipping persistence. Set {CREDENTIAL_ENV} in your shell.")
            return SetupResult(backend=backend, persisted=False, next_command="bruv doctor")
        save_credentials(api_key, path=credential_path)
        return SetupResult(
            backend=backend, persisted=True, next_command="bruv validate -f request.yaml"
        )

    base_url = prompt("Simple Jev base URL [http://127.0.0.1:8000]").strip()
    if not base_url:
        base_url = "http://127.0.0.1:8000"
    print_line(f"Simple Jev endpoint set to {base_url}.")
    print_line("No credential needed for Simple Jev.")
    return SetupResult(backend=backend, persisted=False, next_command="bruv doctor")


__all__ = ["SetupResult", "run_setup"]
