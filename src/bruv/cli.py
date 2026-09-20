"""bruv CLI: Typer composition and process boundary."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, cast

import typer
from pydantic import ValidationError

from bruv.application import ApplicationError, DecisionFacade
from bruv.backends.factory import backend_capabilities, create_backend
from bruv.command_builders import (
    build_single_question_request,
    load_eval_request,
    load_state_file,
    parse_levels,
    parse_options,
)
from bruv.config import BackendName, load_config
from bruv.contracts import SCHEMAS
from bruv.contracts import spec as build_spec
from bruv.domain.questions import ChoiceQuestion, JsonValue, NoulQuestion, ScoreQuestion
from bruv.domain.requests import DecisionRequest
from bruv.domain.results import DecisionResult
from bruv.domain.validation import BackendCapabilities
from bruv.exit_codes import CommandOutcome, exit_code_for
from bruv.gates import GateOptions, apply_gate
from bruv.onboarding.credentials import load_credentials
from bruv.onboarding.doctor import run_doctor
from bruv.onboarding.setup import run_setup
from bruv.output.fields import FieldError, get_field
from bruv.output.json_output import render_error, render_success
from bruv.output.terminal import render_field, render_human_error, render_human_result
from bruv.redaction import redacted_dry_run
from bruv.skills.installer import apply_skill_install, plan_skill_install

app = typer.Typer(help="no yap, only fax")


@app.callback()
def _main_callback() -> None:
    """bruv: typed decisions from TypeSafe Jev or Simple Jev."""


@dataclass(frozen=True, slots=True)
class EvaluationOptions:
    backend: str | None
    model: str | None
    output: str
    quiet: bool
    no_color: bool
    dry_run: bool
    field: str | None
    fail_under: float | None
    abstain_band: tuple[float, float] | None


def _resolve_state(
    state: str | None,
    state_file: Path | None,
) -> JsonValue:
    if state is not None:
        return state
    if state_file is not None:
        return load_state_file(state_file)
    typer.echo("error: --state or --state-file is required", err=True)
    raise typer.Exit(code=2)


def _gate_options(options: EvaluationOptions) -> GateOptions:
    return GateOptions(
        field=options.field or "answers.*.score",
        fail_under=options.fail_under,
        abstain_band=options.abstain_band,
    )


def _emit_result(result: DecisionResult, options: EvaluationOptions) -> None:
    if options.field is not None:
        value = get_field(result, options.field)
        typer.echo(render_field(value))
        return
    if options.output == "json":
        typer.echo(render_success(result))
    else:
        typer.echo(render_human_result(result), nl=False)


def _emit_error(error: ApplicationError, options: EvaluationOptions) -> None:
    if options.output == "json":
        typer.echo(render_error(error))
    else:
        typer.echo(render_human_error(error), err=True)


def _emit_dry_run(
    request: DecisionRequest, capabilities: BackendCapabilities, options: EvaluationOptions
) -> None:
    import json

    preview = redacted_dry_run(request, capabilities)
    typer.echo(json.dumps(preview, sort_keys=True, separators=(",", ":")))


def _build_request(builder: Callable[[], DecisionRequest]) -> DecisionRequest:
    """Build a canonical request, mapping validation errors to exit 2."""
    try:
        return builder()
    except ValidationError as exc:
        typer.echo(f"error: invalid request: {exc}", err=True)
        raise typer.Exit(code=2) from exc


def run_evaluation(options: EvaluationOptions, request: DecisionRequest) -> None:
    """Shared command runner: configure, validate, gate, and exit."""
    config = load_config(backend_override=cast(BackendName | None, options.backend))
    capabilities = backend_capabilities(config)

    if options.dry_run:
        _emit_dry_run(request, capabilities, options)
        raise typer.Exit(code=0)

    try:
        backend = create_backend(config, load_credentials())
    except ApplicationError as error:
        _emit_error(error, options)
        raise typer.Exit(code=exit_code_for(CommandOutcome(error=error))) from error

    facade = DecisionFacade(backend)
    try:
        result = facade.evaluate(request)
    except ApplicationError as error:
        _emit_error(error, options)
        raise typer.Exit(code=exit_code_for(CommandOutcome(error=error))) from error

    try:
        _emit_result(result, options)
    except FieldError as error:
        _emit_error(
            ApplicationError(code="usage_error", message=str(error), paid_request=False), options
        )
        raise typer.Exit(code=2) from error

    gate = apply_gate(result, _gate_options(options))
    raise typer.Exit(
        code=exit_code_for(CommandOutcome(abstained=gate.abstained, gate_passed=gate.gate_passed))
    )


@app.command()
def noul(
    instructions: Annotated[str, typer.Argument(help="Question instructions.")],
    question_id: Annotated[str, typer.Option("--id", help="Question identifier.")] = "q1",
    state: Annotated[str | None, typer.Option("--state")] = None,
    state_file: Annotated[Path | None, typer.Option("--state-file")] = None,
    model: Annotated[str | None, typer.Option("--model")] = None,
    backend: Annotated[str | None, typer.Option("--backend")] = None,
    output: Annotated[str, typer.Option("--output")] = "human",
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
    no_color: Annotated[bool, typer.Option("--no-color")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    field: Annotated[str | None, typer.Option("--field")] = None,
    fail_under: Annotated[float | None, typer.Option("--fail-under")] = None,
    abstain_band: Annotated[str | None, typer.Option("--abstain-band")] = None,
) -> None:
    """Ask a noul (yes/no confidence) question."""
    state_value = _resolve_state(state, state_file)
    request = _build_request(
        lambda: build_single_question_request(
            question_id=question_id,
            question=NoulQuestion(instructions=instructions),
            state=state_value,
            model=model,
        )
    )
    run_evaluation(
        _options(backend, model, output, quiet, no_color, dry_run, field, fail_under, abstain_band),
        request,
    )


@app.command()
def choice(
    instructions: Annotated[str, typer.Argument(help="Question instructions.")],
    question_id: Annotated[str, typer.Option("--id", help="Question identifier.")] = "q1",
    option: Annotated[list[str] | None, typer.Option("--option", help="key[=description]")] = None,
    state: Annotated[str | None, typer.Option("--state")] = None,
    state_file: Annotated[Path | None, typer.Option("--state-file")] = None,
    model: Annotated[str | None, typer.Option("--model")] = None,
    backend: Annotated[str | None, typer.Option("--backend")] = None,
    output: Annotated[str, typer.Option("--output")] = "human",
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
    no_color: Annotated[bool, typer.Option("--no-color")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    field: Annotated[str | None, typer.Option("--field")] = None,
    fail_under: Annotated[float | None, typer.Option("--fail-under")] = None,
    abstain_band: Annotated[str | None, typer.Option("--abstain-band")] = None,
) -> None:
    """Ask a choice question over two or more options."""
    criteria = parse_options(list(option) if option else [])
    state_value = _resolve_state(state, state_file)
    request = _build_request(
        lambda: build_single_question_request(
            question_id=question_id,
            question=ChoiceQuestion(instructions=instructions, criteria=criteria),
            state=state_value,
            model=model,
        )
    )
    run_evaluation(
        _options(backend, model, output, quiet, no_color, dry_run, field, fail_under, abstain_band),
        request,
    )


@app.command()
def score(
    instructions: Annotated[str, typer.Argument(help="Question instructions.")],
    question_id: Annotated[str, typer.Option("--id", help="Question identifier.")] = "q1",
    level: Annotated[list[str] | None, typer.Option("--level", help="Ordered score level.")] = None,
    state: Annotated[str | None, typer.Option("--state")] = None,
    state_file: Annotated[Path | None, typer.Option("--state-file")] = None,
    model: Annotated[str | None, typer.Option("--model")] = None,
    backend: Annotated[str | None, typer.Option("--backend")] = None,
    output: Annotated[str, typer.Option("--output")] = "human",
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
    no_color: Annotated[bool, typer.Option("--no-color")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    field: Annotated[str | None, typer.Option("--field")] = None,
    fail_under: Annotated[float | None, typer.Option("--fail-under")] = None,
    abstain_band: Annotated[str | None, typer.Option("--abstain-band")] = None,
) -> None:
    """Ask a score question over two or more ordered levels."""
    levels = parse_levels(list(level) if level else [])
    state_value = _resolve_state(state, state_file)
    request = _build_request(
        lambda: build_single_question_request(
            question_id=question_id,
            question=ScoreQuestion(instructions=instructions, criteria=levels),
            state=state_value,
            model=model,
        )
    )
    run_evaluation(
        _options(backend, model, output, quiet, no_color, dry_run, field, fail_under, abstain_band),
        request,
    )


@app.command("eval")
def eval_(
    eval_file: Annotated[Path, typer.Argument(help="Path to a JSON/YAML eval file.")],
    state: Annotated[str | None, typer.Option("--state")] = None,
    state_file: Annotated[Path | None, typer.Option("--state-file")] = None,
    model: Annotated[str | None, typer.Option("--model")] = None,
    backend: Annotated[str | None, typer.Option("--backend")] = None,
    output: Annotated[str, typer.Option("--output")] = "human",
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
    no_color: Annotated[bool, typer.Option("--no-color")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    field: Annotated[str | None, typer.Option("--field")] = None,
    fail_under: Annotated[float | None, typer.Option("--fail-under")] = None,
    abstain_band: Annotated[str | None, typer.Option("--abstain-band")] = None,
) -> None:
    """Evaluate a multi-question request from a file."""
    state_override: JsonValue | None = None
    if state is not None:
        state_override = state
    elif state_file is not None:
        state_override = load_state_file(state_file)
    try:
        request = load_eval_request(eval_file, state_override)
    except (ValueError, FileNotFoundError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    run_evaluation(
        _options(backend, model, output, quiet, no_color, dry_run, field, fail_under, abstain_band),
        request,
    )


@app.command()
def validate(
    state: Annotated[str | None, typer.Option("--state")] = None,
    state_file: Annotated[Path | None, typer.Option("--state-file")] = None,
    eval_file: Annotated[Path | None, typer.Option("-f", "--eval-file")] = None,
    backend: Annotated[str | None, typer.Option("--backend")] = None,
    output: Annotated[str, typer.Option("--output")] = "human",
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
) -> None:
    """Validate a canonical request without performing any paid call."""
    if backend is not None and backend not in ("typesafe", "simple-jev"):
        typer.echo(
            f"error: --backend must be 'typesafe' or 'simple-jev', got {backend!r}", err=True
        )
        raise typer.Exit(code=2)
    config = load_config(backend_override=cast(BackendName | None, backend))
    capabilities = backend_capabilities(config)

    class _NeverCallBackend:
        def __init__(self, caps: BackendCapabilities) -> None:
            self.capabilities = caps

        def evaluate(self, req: DecisionRequest) -> DecisionResult:
            raise AssertionError("bruv validate must not call a backend")

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
        typer.echo("valid")
        raise typer.Exit(code=0)
    typer.echo(f"invalid: {len(issues)} issue(s)")
    for issue in issues:
        typer.echo(f"  {issue.code}: {issue.message}", err=True)
    raise typer.Exit(code=2)


def _options(
    backend: str | None,
    model: str | None,
    output: str,
    quiet: bool,
    no_color: bool,
    dry_run: bool,
    field: str | None,
    fail_under: float | None,
    abstain_band: str | None,
) -> EvaluationOptions:
    band: tuple[float, float] | None = None
    if abstain_band is not None:
        low_str, high_str = abstain_band.split(":", 1)
        band = (float(low_str), float(high_str))
    return EvaluationOptions(
        backend=backend,
        model=model,
        output=output,
        quiet=quiet,
        no_color=no_color,
        dry_run=dry_run,
        field=field,
        fail_under=fail_under,
        abstain_band=band,
    )


@app.command()
def spec(
    output: Annotated[str, typer.Option("--output", help="human|json")] = "human",
) -> None:
    """Print the machine-readable CLI command spec."""
    data = build_spec()
    if output == "json":
        typer.echo(render_success_json(data))
    else:
        for command in data["commands"]:
            typer.echo(f"{command['name']}: {command['description']}")


@app.command()
def schema(
    name: Annotated[str, typer.Argument(help="request|output|error")],
) -> None:
    """Print a JSON Schema for the named contract."""
    if name not in SCHEMAS:
        typer.echo(f"error: unknown schema {name!r}; choose request, output, or error", err=True)
        raise typer.Exit(code=2)
    typer.echo(render_success_json(SCHEMAS[name]()))


@app.command("skill")
def skill_command(
    action: Annotated[str, typer.Argument(help="install")],
    target: Annotated[str, typer.Option("--target", help="codex|claude|pi")] = "pi",
    scope: Annotated[str, typer.Option("--scope", help="user|project")] = "user",
    project_root: Annotated[Path | None, typer.Option("--project-root")] = None,
    force: Annotated[bool, typer.Option("--force", help="Replace existing skill.")] = False,
) -> None:
    """Install the bruv skill into an agent's skill directory."""
    if action != "install":
        typer.echo(f"error: unsupported skill action {action!r}; use 'install'", err=True)
        raise typer.Exit(code=2)
    if target not in ("codex", "claude", "pi"):
        typer.echo(f"error: --target must be codex, claude, or pi, got {target!r}", err=True)
        raise typer.Exit(code=2)
    if scope not in ("user", "project"):
        typer.echo(f"error: --scope must be user or project, got {scope!r}", err=True)
        raise typer.Exit(code=2)
    try:
        plan = plan_skill_install(target, scope, project_root)  # type: ignore[arg-type]
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(f"plan: {plan.source} -> {plan.destination} (replacing={plan.replacing})")
    try:
        apply_skill_install(plan, force=force)
    except FileExistsError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(f"installed: {plan.destination}")


@app.command()
def setup() -> None:
    """Guided credential and backend setup (interactive only)."""
    import sys

    if not sys.stdin.isatty():
        typer.echo(
            "error: `bruv setup` is interactive. Set TYPESAFE_API_KEY or edit the "
            "config file in a non-interactive environment.",
            err=True,
        )
        raise typer.Exit(code=2)
    if not typer.confirm("Run interactive setup?", default=True):
        raise typer.Exit(code=0)
    try:
        result = run_setup(
            prompt=typer.prompt,
            confirm=typer.confirm,
            print_line=lambda msg: typer.echo(msg),
            env={},
        )
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(f"backend: {result.backend}")
    typer.echo(f"next: {result.next_command}")


@app.command()
def doctor(
    check_freshness: Annotated[bool, typer.Option("--check-freshness")] = False,
) -> None:
    """Run non-paid diagnostics and print fixes for failed items."""
    results = run_doctor(check_freshness=check_freshness)
    any_failed = False
    for item in results:
        status = "ok" if item.ok else "FAIL"
        typer.echo(f"{status} {item.name}: {item.message}")
        if not item.ok:
            any_failed = True
            if item.fix:
                typer.echo(f"  fix: {item.fix}")
    raise typer.Exit(code=1 if any_failed else 0)


def render_success_json(value: object) -> str:
    """Render an arbitrary JSON object deterministically."""
    import json

    return json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"


def main() -> None:
    app()
