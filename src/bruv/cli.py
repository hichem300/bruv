"""bruv CLI: Typer composition and process boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, cast

import typer

from bruv.application import DecisionFacade
from bruv.backends.factory import backend_capabilities
from bruv.command_builders import load_eval_request, load_state_file
from bruv.config import BackendName, load_config
from bruv.domain.questions import JsonValue
from bruv.domain.requests import DecisionRequest
from bruv.domain.results import DecisionResult
from bruv.domain.validation import BackendCapabilities

app = typer.Typer(help="no yap, only fax")


@app.callback()
def _main_callback() -> None:
    """bruv: typed decisions from TypeSafe Jev or Simple Jev."""


class _NeverCallBackend:
    """Stand-in backend for offline validation; evaluate must never run."""

    def __init__(self, capabilities: BackendCapabilities) -> None:
        self.capabilities = capabilities

    def evaluate(self, request: DecisionRequest) -> DecisionResult:
        raise AssertionError("bruv validate must not call a backend")


def _emit(message: str, *, quiet: bool, output: str) -> None:
    if quiet:
        return
    if output == "json":
        typer.echo(message)
    else:
        typer.echo(message)


@app.command()
def validate(
    state: Annotated[str | None, typer.Option("--state", help="Inline state value.")] = None,
    state_file: Annotated[
        Path | None, typer.Option("--state-file", help="Path to a JSON/YAML state file.")
    ] = None,
    eval_file: Annotated[
        Path | None, typer.Option("-f", "--eval-file", help="Path to a JSON/YAML eval file.")
    ] = None,
    backend: Annotated[str | None, typer.Option("--backend", help="typesafe|simple-jev")] = None,
    output: Annotated[str, typer.Option("--output", help="human|json")] = "human",
    quiet: Annotated[bool, typer.Option("--quiet", help="Suppress diagnostics.")] = False,
) -> None:
    """Validate a canonical request without performing any paid call."""
    if backend is not None and backend not in ("typesafe", "simple-jev"):
        typer.echo(
            f"error: --backend must be 'typesafe' or 'simple-jev', got {backend!r}", err=True
        )
        raise typer.Exit(code=2)
    config = load_config(backend_override=cast(BackendName | None, backend))
    capabilities = backend_capabilities(config)
    facade = DecisionFacade(_NeverCallBackend(capabilities))

    if eval_file is None:
        typer.echo("error: --eval-file is required", err=True)
        raise typer.Exit(code=2)

    state_value: JsonValue | None = None
    if state is not None:
        state_value = state
    elif state_file is not None:
        state_value = load_state_file(state_file)

    try:
        request = load_eval_request(eval_file, state_value)
    except (ValueError, FileNotFoundError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    valid, issues = facade.validate(request)
    if valid:
        _emit("valid", quiet=quiet, output=output)
        raise typer.Exit(code=0)

    _emit(f"invalid: {len(issues)} issue(s)", quiet=quiet, output=output)
    for issue in issues:
        typer.echo(f"  {issue.code}: {issue.message}", err=True)
    raise typer.Exit(code=2)


def main() -> None:
    app()
