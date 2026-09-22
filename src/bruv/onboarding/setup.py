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


def _rlcd_handler(
    *,
    prompt: PromptFn,
    confirm: ConfirmFn,
    print_line: PrintFn,
    credential_path: Path | None,
) -> SetupResult:
    from bruv.backends.registry import get_backend_definition

    install_hint = get_backend_definition("rlcd-modernbert").install_hint
    if install_hint:
        print_line(f"{install_hint} (optional; only needed if the packages are missing).")
    print_line("RLCD runs fully local with free inference.")
    print_line("First use downloads about 606 MB from Hugging Face.")
    print_line("No credential needed for RLCD.")
    return SetupResult(backend="rlcd-modernbert", persisted=False, next_command="bruv doctor")


_HANDLERS: dict[str, _BackendSetupHandler] = {
    "typesafe": _typesafe_handler,
    "simple-jev": _simple_jev_handler,
    "needle": _needle_handler,
    "rlcd-modernbert": _rlcd_handler,
}


def _managed_simple_jev_setup(
    *,
    confirm: ConfirmFn,
    print_line: PrintFn,
    config_path: Path | None,
    repair: bool,
) -> SetupResult:
    """Install, persist, launch, and smoke-test the managed Simple Jev runtime."""
    import httpx

    from bruv.backends.simple_jev import SimpleJevAdapter
    from bruv.config import update_config
    from bruv.domain.questions import NoulQuestion
    from bruv.domain.requests import DecisionRequest
    from bruv.onboarding.simple_jev_runtime import (
        ManagedSimpleJevRuntime,
        ManagedSimpleJevSettings,
        SimpleJevPaths,
    )

    settings = ManagedSimpleJevSettings()
    paths = SimpleJevPaths.default()

    print_line(f"Model: {settings.model}")
    print_line(
        "Device: auto (auto-detects CUDA through the managed torch and "
        "falls back to CPU when unavailable)"
    )
    print_line(f"Endpoint: {settings.base_url}")
    print_line(
        "Simple Jev will be installed isolated under "
        f"{paths.root}; nothing is written outside that managed root."
    )
    print_line(
        "First launch downloads the model from Hugging Face into the "
        "managed cache and can take several minutes."
    )
    if not confirm("Install and start the managed Simple Jev runtime now?"):
        print_line("Cancelled. Nothing was installed or persisted.")
        return SetupResult(
            backend="simple-jev", persisted=False, next_command="bruv setup simple-jev"
        )

    runtime = ManagedSimpleJevRuntime(settings)
    report = runtime.install(repair=repair)

    update_config(
        {
            "backend": "simple-jev",
            "simple_jev_managed": True,
            "simple_jev_model": report.model,
            "simple_jev_base_url": report.base_url,
            "simple_jev_device": report.device,
            "simple_jev_dtype": report.dtype,
        },
        path=config_path,
    )

    outcome = runtime.ensure_running()

    try:
        with httpx.Client() as client:
            adapter = SimpleJevAdapter(
                base_url=outcome.base_url,
                model=report.model,
                client=client,
            )
            request = DecisionRequest(
                state="smoke-check",
                questions={
                    "smoke": NoulQuestion(
                        instructions="Answer anything; this is a smoke check."
                    )
                },
            )
            result = adapter.evaluate(request)
    except Exception as exc:  # noqa: BLE001 - converted to actionable unpaid error
        raise ConfigurationError(
            message=(
                "Managed Simple Jev smoke classification failed "
                f"({exc.__class__.__name__}); the runtime is installed and the "
                "config is persisted, but the endpoint did not answer "
                "correctly."
            ),
            paid_request=False,
            action=(
                "Run `bruv serve simple-jev status` and inspect the managed "
                f"log at {paths.log_file}. Fix or restart the server, then "
                "retry `bruv setup simple-jev`."
            ),
            details={"log_file": str(paths.log_file)},
        ) from exc
    if not result.answers:
        raise ConfigurationError(
            message=(
                "Managed Simple Jev smoke classification returned no answers; "
                "the runtime is installed and the config is persisted, but the "
                "endpoint response was empty."
            ),
            paid_request=False,
            action=(
                "Run `bruv serve simple-jev status` and inspect the managed "
                f"log at {paths.log_file}, then retry "
                "`bruv setup simple-jev`."
            ),
            details={"log_file": str(paths.log_file)},
        )

    print_line(
        f"Simple Jev ready at {outcome.base_url} "
        f"(model {report.model}, device {report.device}, dtype {report.dtype})."
    )
    print_line("Ready: bruv doctor")
    return SetupResult(backend="simple-jev", persisted=True, next_command="bruv doctor")


def run_setup(
    *,
    prompt: PromptFn,
    confirm: ConfirmFn,
    print_line: PrintFn,
    env: Mapping[str, str],
    config_path: Path | None = None,
    credential_path: Path | None = None,
    preselected_backend: str | None = None,
    repair: bool = False,
) -> SetupResult:
    """Interactive setup. Non-interactive callers should refuse before this.

    ``preselected_backend`` skips the menu when it names a registered backend;
    an unknown value falls back to the interactive menu unchanged.
    """
    from bruv.backends.registry import backend_names, get_backend_definition

    if preselected_backend is not None:
        if preselected_backend not in backend_names:
            print_line(
                f"Unknown preselected backend {preselected_backend!r}; "
                "showing the backend menu."
            )
        elif preselected_backend == "simple-jev":
            return _managed_simple_jev_setup(
                confirm=confirm,
                print_line=print_line,
                config_path=config_path,
                repair=repair,
            )
        else:
            return _HANDLERS[preselected_backend](
                prompt=prompt,
                confirm=confirm,
                print_line=print_line,
                credential_path=credential_path,
            )

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
