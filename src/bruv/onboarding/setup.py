"""Guided setup service.

Prompting, filesystem, environment, and network probing are injected so the
service is testable without a TTY or live endpoints. Secrets never enter
diagnostic output.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

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


class _BackendSetupHandler(Protocol):
    def __call__(
        self,
        *,
        prompt: PromptFn,
        confirm: ConfirmFn,
        print_line: PrintFn,
        credential_path: Path | None,
    ) -> SetupResult: ...


def _typesafe_handler(
    *,
    prompt: PromptFn,
    confirm: ConfirmFn,
    print_line: PrintFn,
    credential_path: Path | None,
) -> SetupResult:
    api_key = prompt("TypeSafe API key (input hidden)")
    if not api_key.strip():
        raise ConfigurationError(
            message="an API key is required for the typesafe backend",
            paid_request=False,
            action=f"Set {CREDENTIAL_ENV} or re-run setup with a key.",
        )
    if not confirm("Persist the key to a protected credential file?"):
        print_line(f"Skipping persistence. Set {CREDENTIAL_ENV} in your shell.")
        return SetupResult(backend="typesafe", persisted=False, next_command="bruv doctor")
    save_credentials(api_key, path=credential_path)
    return SetupResult(
        backend="typesafe", persisted=True, next_command="bruv validate -f request.yaml"
    )


def _simple_jev_handler(
    *,
    prompt: PromptFn,
    confirm: ConfirmFn,
    print_line: PrintFn,
    credential_path: Path | None,
) -> SetupResult:
    base_url = prompt("Simple Jev base URL [http://127.0.0.1:8000]").strip()
    if not base_url:
        base_url = "http://127.0.0.1:8000"
    print_line(f"Simple Jev endpoint set to {base_url}.")
    print_line("No credential needed for Simple Jev.")
    return SetupResult(backend="simple-jev", persisted=False, next_command="bruv doctor")


def _needle_handler(
    *,
    prompt: PromptFn,
    confirm: ConfirmFn,
    print_line: PrintFn,
    credential_path: Path | None,
) -> SetupResult:
    from bruv.backends.registry import get_backend_definition

    install_hint = get_backend_definition("needle").install_hint
    if install_hint:
        print_line(f"{install_hint} (optional; only needed if the package is missing).")
    print_line("Needle runs fully local. First use downloads about 35 MB of model weights.")
    print_line("No credential needed for Needle.")
    return SetupResult(backend="needle", persisted=False, next_command="bruv doctor")


_HANDLERS: dict[str, _BackendSetupHandler] = {
    "typesafe": _typesafe_handler,
    "simple-jev": _simple_jev_handler,
    "needle": _needle_handler,
}


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
    from bruv.backends.registry import backend_names, get_backend_definition

    print_line("bruv setup: choose a backend.")
    for name in backend_names:
        definition = get_backend_definition(name)
        suffix = " (set it during setup)" if definition.needs_credentials else ""
        print_line(f"  {name}: {definition.setup_description}{suffix}")
    backend = prompt(f"Backend [{'|'.join(backend_names)}]").strip().lower()
    if backend not in backend_names:
        supported = "|".join(backend_names)
        raise ConfigurationError(
            message=f"backend must be one of: {supported}",
            paid_request=False,
            action="Re-run setup and choose a supported backend.",
        )

    return _HANDLERS[backend](
        prompt=prompt,
        confirm=confirm,
        print_line=print_line,
        credential_path=credential_path,
    )


__all__ = ["SetupResult", "run_setup"]
