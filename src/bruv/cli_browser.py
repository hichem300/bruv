"""Browser CLI: Typer group over the local loopback daemon.

Security model:

- Raw goal text and reply text are passed to the daemon in request bodies
  only. They are never printed, logged, or persisted; human output shows
  status, session, pending request metadata, artifact paths, and failure
  categories exclusively.
- ``bruv browser run`` auto-starts one detached local daemon when no valid
  connection exists. State/reply/stop/shutdown never auto-start one.
- The daemon start lock is an exclusive small file with restrictive
  permissions; it carries a pid only, never secrets. A live daemon
  connection file is never deleted here.
- Playwright is never imported at module import time; the production factory
  imports browser modules lazily.
"""

from __future__ import annotations

import hashlib
import os
import subprocess  # noqa: S404
import sys
import time
from collections.abc import Callable
from http.client import HTTPConnection
from pathlib import Path
from typing import Annotated, Any, Final, NoReturn

import typer
from pydantic import ValidationError

from bruv.application import ConfigurationError
from bruv.browser.artifacts import ArtifactOptions, privacy_warning
from bruv.browser.daemon import (
    BrowserDaemon,
    DaemonClient,
    DaemonControlledBrowser,
    DaemonError,
    default_connection_path,
    read_connection_file,
)
from bruv.browser.models import SessionState
from bruv.config import AppConfig, load_config
from bruv.onboarding.credentials import load_credentials
from bruv.output.terminal import render_human_error

browser_app = typer.Typer(help="Drive a review-gated browser session over a bruv backend.")

# Detached auto-start is bounded; the daemon child binds and publishes its
# connection file before serving.
_DAEMON_START_WAIT_SECONDS: Final[float] = 5.0
_PROBE_TIMEOUT_SECONDS: Final[float] = 0.5
_START_LOCK_STALE_SECONDS: Final[float] = 30.0

_BROWSER_OPTION_KEYS: Final[frozenset[str]] = frozenset(
    {"allow_origins", "headed", "trace", "video"}
)

_OUTPUTS: Final[tuple[str, ...]] = ("human", "json")


# -- validation helpers -----------------------------------------------------------


def _validate_output(output: str) -> str:
    if output not in _OUTPUTS:
        typer.echo(f"error: --output must be human or json, got {output!r}", err=True)
        raise typer.Exit(code=2)
    return output


def _validated_cli_origins(raw: list[str] | None) -> tuple[str, ...]:
    """Normalize and validate repeated --allow-origin values via AppConfig."""
    if not raw:
        return ()
    try:
        probe = AppConfig.model_validate({"browser_allowed_origins": tuple(raw)})
    except ValidationError:
        typer.echo(
            "error: --allow-origin must be a bare http(s) origin, "
            "e.g. https://example.com (repeatable, no duplicates)",
            err=True,
        )
        raise typer.Exit(code=2) from None
    return probe.browser_allowed_origins


def _validated_browser_options(options: Any) -> dict[str, Any]:
    """Fail closed on unknown or mistyped daemon create options."""
    if options is None:
        return {}
    if not isinstance(options, dict) or set(options) - _BROWSER_OPTION_KEYS:
        raise ValueError("unknown browser session options")
    for key in ("headed", "trace", "video"):
        if key in options and not isinstance(options[key], bool):
            raise ValueError(f"browser option {key!r} must be boolean")
    origins = options.get("allow_origins", ())
    if not isinstance(origins, (list, tuple)) or not all(
        isinstance(origin, str) for origin in origins
    ):
        raise ValueError("browser option 'allow_origins' must be a list of strings")
    return dict(options)


def _effective_allow_origins(config: AppConfig, options: dict[str, Any]) -> tuple[str, ...]:
    """Configured canonical origins plus validated CLI origins, deduplicated."""
    raw_cli = tuple(options.get("allow_origins") or ())
    try:
        validated = AppConfig.model_validate({"browser_allowed_origins": raw_cli})
    except ValidationError as exc:
        raise ValueError("invalid --allow-origin value") from exc
    merged: list[str] = []
    for origin in (*config.browser_allowed_origins, *validated.browser_allowed_origins):
        if origin not in merged:
            merged.append(origin)
    return tuple(merged)


# -- production controller factory -------------------------------------------------


def _production_controller_factory(
    *,
    session_id: str,
    url: str,
    goal_text: str,
    on_state: Callable[[SessionState], None],
    backend: str | None,
    options: dict[str, Any] | None,
) -> DaemonControlledBrowser:
    """Build one BrowserController from AppConfig and registry-derived compatibility.

    Runs inside the daemon worker process. The raw goal text stays in memory;
    the controller persists only the SHA-256 ``goal_ref``.
    """
    from bruv.browser.controller import BrowserController
    from bruv.browser.driver import BrowserBackendDriver
    from bruv.browser.executor import BrowserExecutor
    from bruv.browser.observation import ObservationBuilder

    validated = _validated_browser_options(options)
    config = load_config(backend_override=backend)
    credentials = load_credentials()
    # Registry-derived browser compatibility: BrowserBackendDriver rejects
    # backends that cannot answer canonical noul and choice questions.
    driver = BrowserBackendDriver.create(config, credentials)

    allow_origins = _effective_allow_origins(config, validated)
    builder = ObservationBuilder(
        max_elements=config.browser_observation_max_elements,
        max_page_text_chars=config.browser_page_text_max_chars,
        allowed_origins=allow_origins,
    )
    artifact_options = ArtifactOptions(
        trace=bool(validated.get("trace", False)) or config.browser_capture_trace,
        video=bool(validated.get("video", False)) or config.browser_capture_video,
    )
    executor = BrowserExecutor(
        session_id=session_id,
        initial_url=url,
        builder=builder,
        allowed_origins=allow_origins,
        artifact_dir=config.browser_artifact_dir,
        artifact_options=artifact_options,
        headless=config.browser_headless and not bool(validated.get("headed", False)),
    )
    goal_ref = f"sha256:{hashlib.sha256(goal_text.encode('utf-8')).hexdigest()}"
    return BrowserController(
        session_id=session_id,
        initial_url=url,
        goal_text=goal_text,
        goal_ref=goal_ref,
        driver=driver,
        executor=executor,
        max_steps=config.browser_max_steps,
        time_limit_seconds=config.browser_time_limit_seconds,
        origin_allowlist=allow_origins,
        on_state=on_state,
    )


# -- detached daemon auto-start -----------------------------------------------------


def _start_lock_path() -> Path:
    return default_connection_path().parent / "daemon-start.lock"


def _process_alive(pid: int) -> bool:
    """POSIX-only liveness probe; callers on other platforms must not use it."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True  # conservative: present but not ours to inspect
    return True


def _lock_file_is_stale(path: Path) -> bool:
    try:
        return time.time() - path.stat().st_mtime > _START_LOCK_STALE_SECONDS
    except OSError:
        return True


def _acquire_start_lock() -> int | None:
    """Exclusive O_EXCL lock file (0600, pid content only). None means held."""
    path = _start_lock_path()
    for _ in range(2):
        try:
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            if not _lock_file_is_stale(path):
                return None
            try:
                os.unlink(path)
            except OSError:
                return None
            continue
        except OSError:
            return None
        try:
            os.write(fd, str(os.getpid()).encode("ascii"))
        except OSError:
            pass
        return fd
    return None


def _release_start_lock(fd: int) -> None:
    try:
        os.close(fd)
    except OSError:
        pass
    try:
        os.unlink(_start_lock_path())
    except OSError:
        pass


def _probe_connection(info: Any, timeout: float) -> bool:
    """True only for an authenticated GET /sessions with status 200 or 404.

    A 401 means the token is stale (connection file recycled or overwritten);
    returning False lets auto-start replace the stale connection.
    """
    connection = HTTPConnection("127.0.0.1", info.port, timeout=timeout)
    try:
        connection.request("GET", "/sessions", headers={"Authorization": f"Bearer {info.token}"})
        response = connection.getresponse()
        response.read(4096)
        return response.status in (200, 404)
    except OSError:
        return False
    finally:
        connection.close()


def _try_client(probe_timeout: float = _PROBE_TIMEOUT_SECONDS) -> DaemonClient | None:
    try:
        info = read_connection_file()
    except DaemonError:
        return None
    if os.name == "posix" and not _process_alive(info.pid):
        return None
    if not _probe_connection(info, timeout=probe_timeout):
        return None
    try:
        return DaemonClient()
    except DaemonError:
        return None


def _spawn_detached_daemon() -> None:
    command = [sys.executable, "-m", "bruv.cli_browser", "_daemon"]
    kwargs: dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if os.name == "nt":
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        flags |= getattr(subprocess, "DETACHED_PROCESS", 0)
        kwargs["creationflags"] = flags
    else:
        kwargs["start_new_session"] = True
    try:
        subprocess.Popen(command, **kwargs)  # noqa: S603
    except OSError:
        raise DaemonError("browser daemon could not be started") from None


def _client_with_autostart() -> DaemonClient:
    """Return a client, auto-starting one detached daemon when none is valid."""
    client = _try_client()
    if client is not None:
        return client
    deadline = time.monotonic() + _DAEMON_START_WAIT_SECONDS
    lock_fd = _acquire_start_lock()
    if lock_fd is None:
        # Another process holds the start lock; wait for its connection.
        while time.monotonic() < deadline:
            client = _try_client()
            if client is not None:
                return client
            time.sleep(0.2)
        raise DaemonError("browser daemon is starting but did not become reachable")
    try:
        _spawn_detached_daemon()
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            client = _try_client(probe_timeout=min(remaining, _PROBE_TIMEOUT_SECONDS))
            if client is not None:
                return client
            time.sleep(0.2)
        raise DaemonError("browser daemon did not become reachable in time")
    finally:
        _release_start_lock(lock_fd)


# -- output rendering ---------------------------------------------------------------


def _render_state_human(state: SessionState) -> str:
    """Concise safe metadata only: status/session/pending/artifacts/failure."""
    lines = [
        f"status: {state.status}",
        f"session: {state.session_id}",
        f"backend: {state.backend}",
    ]
    pending = state.pending_request
    if pending is not None:
        lines.append(f"pending: {pending.kind} request={pending.request_id}")
        if pending.review_kind is not None:
            lines.append(f"review: {pending.review_kind}")
        if pending.candidate_id:
            lines.append(f"candidate: {pending.candidate_id}")
    artifacts = state.artifacts
    if artifacts is not None:
        for label, path in (
            ("screenshot", artifacts.screenshot_path),
            ("transcript", artifacts.transcript_path),
            ("trace", artifacts.trace_path),
            ("video", artifacts.video_path),
        ):
            if path:
                lines.append(f"artifact-{label}: {path}")
    if state.failure_reason:
        lines.append(f"failure: {state.failure_reason}")
    return "\n".join(lines)


def _state_json(state: SessionState) -> str:
    import json

    return json.dumps(
        state.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _emit_state(state: SessionState, output: str) -> None:
    if output == "json":
        typer.echo(_state_json(state))
    else:
        typer.echo(_render_state_human(state))


def _exit_code_for_state(state: SessionState) -> int:
    return 1 if state.status == "failed" else 0


def _config_error(error: ConfigurationError) -> typer.Exit:
    typer.echo(render_human_error(error), err=True)
    return typer.Exit(code=2)


def _daemon_error(error: DaemonError) -> typer.Exit:
    typer.echo(f"error: {error}", err=True)
    return typer.Exit(code=2)


def _client_no_autostart() -> DaemonClient:
    try:
        return DaemonClient()
    except DaemonError as error:
        raise _daemon_error(error) from error


# -- commands ------------------------------------------------------------------------


@browser_app.command()
def run(
    url: Annotated[str, typer.Argument(help="Initial absolute http(s) URL.")],
    goal: Annotated[
        str, typer.Option("--goal", help="Goal text. Never printed, logged, or persisted.")
    ],
    backend: Annotated[str | None, typer.Option("--backend", help="Backend override.")] = None,
    allow_origin: Annotated[
        list[str] | None,
        typer.Option("--allow-origin", help="Extra allowed origin. Repeatable."),
    ] = None,
    headed: Annotated[bool, typer.Option("--headed", help="Run with a visible window.")] = False,
    trace: Annotated[bool, typer.Option("--trace", help="Capture a Playwright trace.")] = False,
    video: Annotated[bool, typer.Option("--video", help="Capture a session video.")] = False,
) -> None:
    """Start a review-gated browser session. Auto-starts the local daemon."""
    cli_origins = _validated_cli_origins(list(allow_origin) if allow_origin else [])
    try:
        load_config(backend_override=backend)
    except ConfigurationError as error:
        raise _config_error(error) from error

    typer.echo(privacy_warning())
    options: dict[str, Any] = {
        "allow_origins": list(cli_origins),
        "headed": headed,
        "trace": trace,
        "video": video,
    }
    try:
        client = _client_with_autostart()
        state = client.create_session(url.strip(), goal, backend=backend, options=options)
    except DaemonError as error:
        raise _daemon_error(error) from error
    typer.echo(_render_state_human(state))
    raise typer.Exit(code=_exit_code_for_state(state))


@browser_app.command()
def state(
    session_id: Annotated[str | None, typer.Argument(help="Session ID; default latest.")] = None,
    output: Annotated[str, typer.Option("--output", help="human|json")] = "human",
) -> None:
    """Show session state. Does not start the daemon."""
    _validate_output(output)
    try:
        state_model = _client_no_autostart().state(session_id)
    except DaemonError as error:
        raise _daemon_error(error) from error
    _emit_state(state_model, output)
    raise typer.Exit(code=_exit_code_for_state(state_model))


@browser_app.command()
def reply(
    text: Annotated[str, typer.Option("--text", help="Reply text. Never printed or persisted.")],
    session_id: Annotated[str | None, typer.Argument(help="Session ID; default latest.")] = None,
    output: Annotated[str, typer.Option("--output", help="human|json")] = "human",
) -> None:
    """Reply to a pending text request. Does not start the daemon."""
    _validate_output(output)
    try:
        state_model = _client_no_autostart().reply(text, session_id)
    except DaemonError as error:
        raise _daemon_error(error) from error
    _emit_state(state_model, output)
    raise typer.Exit(code=_exit_code_for_state(state_model))


@browser_app.command()
def stop(
    session_id: Annotated[str | None, typer.Argument(help="Session ID; default latest.")] = None,
    output: Annotated[str, typer.Option("--output", help="human|json")] = "human",
) -> None:
    """Stop a live session. Does not start the daemon."""
    _validate_output(output)
    try:
        state_model = _client_no_autostart().stop(session_id)
    except DaemonError as error:
        raise _daemon_error(error) from error
    _emit_state(state_model, output)
    raise typer.Exit(code=_exit_code_for_state(state_model))


@browser_app.command()
def shutdown() -> None:
    """Shut down the local daemon. Does not auto-start it."""
    try:
        _client_no_autostart().shutdown()
    except DaemonError as error:
        raise _daemon_error(error) from error
    typer.echo("daemon stopped")
    raise typer.Exit(code=0)


# -- setup and internal daemon --------------------------------------------------------


def run_browser_setup() -> NoReturn:
    """Non-interactive Playwright Chromium install. Exits 0/1."""
    try:
        completed = subprocess.run(  # noqa: S603
            [sys.executable, "-m", "playwright", "install", "chromium"],
            check=False,
        )
    except OSError:
        typer.echo("error: browser setup failed", err=True)
        raise typer.Exit(code=1) from None
    if completed.returncode == 0:
        typer.echo("browser setup complete: chromium installed")
        raise typer.Exit(code=0)
    typer.echo("error: chromium installation failed", err=True)
    raise typer.Exit(code=1)


def _daemon_main(argv: list[str]) -> int:
    """Internal daemon entrypoint: serves BrowserDaemon with the production factory."""
    if argv:
        return 2
    daemon: BrowserDaemon | None = None
    try:
        daemon = BrowserDaemon(_production_controller_factory)
        daemon.start()
    except KeyboardInterrupt:
        pass
    except Exception:
        return 1
    finally:
        if daemon is not None:
            daemon.shutdown()
    return 0


def main() -> None:
    """Module entry: `_daemon` runs the daemon; anything else delegates to the CLI."""
    if len(sys.argv) > 1 and sys.argv[1] == "_daemon":
        raise SystemExit(_daemon_main(sys.argv[2:]))
    from bruv.cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()


__all__ = ["browser_app", "main", "run_browser_setup"]
