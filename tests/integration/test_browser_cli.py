"""Focused integration tests for the browser CLI surface.

All daemon interaction is faked; no sockets, browser, or paid calls. Goal and
reply text fixtures are safe placeholder strings that must never appear in CLI
output.
"""

from __future__ import annotations

import json
import sys
from http.client import HTTPConnection
from typing import Any

from typer.testing import CliRunner

from bruv import cli_browser
from bruv.browser.daemon import DaemonError
from bruv.browser.models import BrowserGoal, SessionState
from bruv.cli import app

runner = CliRunner()

GOAL_TEXT = "find the pricing page and report the plan names"
REPLY_TEXT = "use the saved work email"


def _state(status: str = "running") -> SessionState:
    return SessionState(
        session_id="sess-1",
        status=status,  # type: ignore[arg-type]
        goal=BrowserGoal(
            goal_id="goal-1",
            goal_ref="sha256:deadbeef",
            initial_url="https://example.com",
        ),
        backend="typesafe",
        origin_allowlist=("https://example.com",),
    )


class FakeClient:
    """Records calls, returns canned SessionState. Never touches the network."""

    def __init__(self, state: SessionState | None = None, error: DaemonError | None = None):
        self._canned = state if state is not None else _state()
        self.error = error
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def create_session(self, url: str, goal: str, backend=None, options=None) -> SessionState:
        self.calls.append(("create_session", (url, goal), {"backend": backend, "options": options}))
        if self.error:
            raise self.error
        return self._canned

    def state(self, session_id=None) -> SessionState:
        self.calls.append(("state", (session_id,), {}))
        if self.error:
            raise self.error
        return self._canned

    def reply(self, text: str, session_id=None) -> SessionState:
        self.calls.append(("reply", (text, session_id), {}))
        if self.error:
            raise self.error
        return self._canned

    def stop(self, session_id=None) -> SessionState:
        self.calls.append(("stop", (session_id,), {}))
        if self.error:
            raise self.error
        return self._canned

    def shutdown(self) -> None:
        self.calls.append(("shutdown", (), {}))
        if self.error:
            raise self.error


def _install_client(monkeypatch, client: FakeClient) -> None:
    monkeypatch.setattr(cli_browser, "DaemonClient", lambda: client)
    monkeypatch.setattr(cli_browser, "_client_with_autostart", lambda: client)
    from bruv.config import AppConfig

    monkeypatch.setattr(cli_browser, "load_config", lambda backend_override=None: AppConfig())


# -- help / command surface ----------------------------------------------------------


def test_browser_help_exposes_commands_and_hides_daemon() -> None:
    result = runner.invoke(app, ["browser", "--help"])
    assert result.exit_code == 0
    out = result.stdout
    for name in ("run", "state", "reply", "stop", "shutdown"):
        assert name in out
    assert "_daemon" not in out
    command_names = {cmd.name for cmd in cli_browser.browser_app.registered_commands}
    assert "_daemon" not in command_names


# -- run -----------------------------------------------------------------------------


def test_run_default_backend_passes_goal_and_prints_safe_state(monkeypatch) -> None:
    client = FakeClient()
    _install_client(monkeypatch, client)
    result = runner.invoke(
        app,
        ["browser", "run", "https://example.com", "--goal", GOAL_TEXT],
    )
    assert result.exit_code == 0
    assert "private page content" in result.stdout.lower()
    assert GOAL_TEXT not in result.stdout
    assert "status: running" in result.stdout
    assert "session: sess-1" in result.stdout
    ((name, args, kwargs),) = client.calls
    assert name == "create_session"
    assert args == ("https://example.com", GOAL_TEXT)
    assert kwargs["backend"] is None
    assert kwargs["options"]["allow_origins"] == []


def test_run_backend_override_and_repeated_allow_origins(monkeypatch) -> None:
    client = FakeClient()
    _install_client(monkeypatch, client)
    result = runner.invoke(
        app,
        [
            "browser",
            "run",
            "https://example.com",
            "--goal",
            GOAL_TEXT,
            "--backend",
            "typesafe",
            "--allow-origin",
            "https://a.example",
            "--allow-origin",
            "https://b.example",
        ],
    )
    assert result.exit_code == 0
    ((name, args, kwargs),) = client.calls
    assert kwargs["backend"] == "typesafe"
    assert kwargs["options"]["allow_origins"] == ["https://a.example", "https://b.example"]


def test_run_daemon_error_safe_exit(monkeypatch) -> None:
    client = FakeClient(error=DaemonError("browser daemon is not reachable"))
    _install_client(monkeypatch, client)
    result = runner.invoke(
        app,
        ["browser", "run", "https://example.com", "--goal", GOAL_TEXT],
    )
    assert result.exit_code == 2
    output = result.stdout + (result.stderr or "")
    assert "error:" in output
    assert GOAL_TEXT not in output


# -- state ---------------------------------------------------------------------------


def test_state_human(monkeypatch) -> None:
    client = FakeClient()
    _install_client(monkeypatch, client)
    result = runner.invoke(app, ["browser", "state"])
    assert result.exit_code == 0
    assert "status: running" in result.stdout
    assert "session: sess-1" in result.stdout


def test_state_json(monkeypatch) -> None:
    client = FakeClient()
    _install_client(monkeypatch, client)
    result = runner.invoke(app, ["browser", "state", "--output", "json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "running"
    assert payload["session_id"] == "sess-1"


def test_state_failed_status_exits_one(monkeypatch) -> None:
    client = FakeClient(state=_state("failed"))
    _install_client(monkeypatch, client)
    result = runner.invoke(app, ["browser", "state"])
    assert result.exit_code == 1


# -- reply ---------------------------------------------------------------------------


def test_reply_requires_text_option(monkeypatch) -> None:
    client = FakeClient()
    _install_client(monkeypatch, client)
    result = runner.invoke(app, ["browser", "reply"])
    assert result.exit_code == 2
    assert client.calls == []


def test_reply_passes_text_in_memory_never_prints(monkeypatch) -> None:
    client = FakeClient()
    _install_client(monkeypatch, client)
    result = runner.invoke(app, ["browser", "reply", "--text", REPLY_TEXT])
    assert result.exit_code == 0
    ((name, args, _kwargs),) = client.calls
    assert name == "reply"
    assert args == (REPLY_TEXT, None)
    assert REPLY_TEXT not in result.stdout


def test_reply_explicit_session(monkeypatch) -> None:
    client = FakeClient()
    _install_client(monkeypatch, client)
    result = runner.invoke(app, ["browser", "reply", "--text", REPLY_TEXT, "sess-9"])
    assert result.exit_code == 0
    ((name, args, _kwargs),) = client.calls
    assert args == (REPLY_TEXT, "sess-9")


# -- stop / shutdown -----------------------------------------------------------------


def test_stop_uses_fake_daemon_and_safe_output(monkeypatch) -> None:
    client = FakeClient(state=_state("stopped"))
    _install_client(monkeypatch, client)
    result = runner.invoke(app, ["browser", "stop", "sess-1"])
    assert result.exit_code == 0
    ((name, args, _kwargs),) = client.calls
    assert name == "stop"
    assert args == ("sess-1",)
    assert "status: stopped" in result.stdout


def test_shutdown_safe_output(monkeypatch) -> None:
    client = FakeClient()
    _install_client(monkeypatch, client)
    result = runner.invoke(app, ["browser", "shutdown"])
    assert result.exit_code == 0
    assert client.calls and client.calls[0][0] == "shutdown"
    assert "daemon stopped" in result.stdout


def test_stop_daemon_error_safe_exit(monkeypatch) -> None:
    client = FakeClient(error=DaemonError("browser daemon is not reachable"))
    _install_client(monkeypatch, client)
    result = runner.invoke(app, ["browser", "stop"])
    assert result.exit_code == 2
    assert "error:" in result.stdout + (result.stderr or "")


# -- setup browser -------------------------------------------------------------------


def test_setup_browser_bypasses_tty_runs_exact_command(monkeypatch) -> None:
    recorded: list[dict[str, Any]] = []

    def fake_run(command, **kwargs):
        recorded.append({"command": command, **kwargs})
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(cli_browser.subprocess, "run", fake_run)
    result = runner.invoke(app, ["setup", "browser"])
    assert result.exit_code == 0
    assert recorded == [
        {"command": [sys.executable, "-m", "playwright", "install", "chromium"], "check": False}
    ]
    assert "browser setup complete" in result.stdout


def test_setup_browser_nonzero_returns_one_never_success_text(monkeypatch) -> None:
    monkeypatch.setattr(
        cli_browser.subprocess,
        "run",
        lambda command, **kwargs: type("Completed", (), {"returncode": 7})(),
    )
    result = runner.invoke(app, ["setup", "browser"])
    assert result.exit_code == 1
    output = result.stdout + (result.stderr or "")
    assert "browser setup complete" not in output


def test_setup_browser_oserror_returns_one(monkeypatch) -> None:
    def boom(command, **kwargs):
        raise OSError("no playwright")

    monkeypatch.setattr(cli_browser.subprocess, "run", boom)
    result = runner.invoke(app, ["setup", "browser"])
    assert result.exit_code == 1


# -- _probe_connection ---------------------------------------------------------------


class _FakeResponse:
    def __init__(self, status: int) -> None:
        self.status = status

    def read(self, amount: int) -> bytes:
        return b""

    def close(self) -> None:
        pass


class _FakeHTTPConnection:
    """Tiny fake HTTPConnection; never opens a socket."""

    last: dict[str, Any] | None = None

    def __init__(self, host: str, port: int, timeout: float) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.closed = False
        _FakeHTTPConnection.last = {
            "host": host,
            "port": port,
            "timeout": timeout,
        }

    def request(self, method: str, path: str, headers: dict[str, str] | None = None) -> None:
        _FakeHTTPConnection.last["method"] = method  # type: ignore[index]
        _FakeHTTPConnection.last["path"] = path  # type: ignore[index]
        _FakeHTTPConnection.last["headers"] = headers  # type: ignore[index]
        self._pending_status = 200

    def getresponse(self) -> _FakeResponse:
        return _FakeResponse(self._pending_status)

    def close(self) -> None:
        self.closed = True


def _probe_with_status(monkeypatch, status: int) -> bool:
    class _Conn(_FakeHTTPConnection):
        def getresponse(self) -> _FakeResponse:
            return _FakeResponse(status)

    monkeypatch.setattr(cli_browser, "HTTPConnection", _Conn)
    info = type("Info", (), {"port": 4567, "token": "token-value"})()
    return cli_browser._probe_connection(info, timeout=0.5)


def test_probe_connection_live_on_200_and_404(monkeypatch) -> None:
    assert _probe_with_status(monkeypatch, 200) is True
    assert _probe_with_status(monkeypatch, 404) is True


def test_probe_connection_stale_on_401(monkeypatch) -> None:
    assert _probe_with_status(monkeypatch, 401) is False


def test_probe_connection_oserror_is_dead(monkeypatch) -> None:
    class _BrokenConn(_FakeHTTPConnection):
        def request(self, method: str, path: str, headers=None) -> None:
            raise OSError("connection refused")

    monkeypatch.setattr(cli_browser, "HTTPConnection", _BrokenConn)
    info = type("Info", (), {"port": 4567, "token": "token-value"})()
    assert cli_browser._probe_connection(info, timeout=0.5) is False


def test_probe_connection_sends_authenticated_sessions_request(monkeypatch) -> None:
    monkeypatch.setattr(cli_browser, "HTTPConnection", _FakeHTTPConnection)
    info = type("Info", (), {"port": 4567, "token": "token-value"})()
    cli_browser._probe_connection(info, timeout=0.5)
    sent = _FakeHTTPConnection.last
    assert sent is not None
    assert sent["port"] == 4567
    assert sent["method"] == "GET"
    assert sent["path"] == "/sessions"
    assert sent["headers"]["Authorization"] == "Bearer token-value"
    # No real sockets: the fake never subclasses/uses the real HTTPConnection.
    assert cli_browser.HTTPConnection is not HTTPConnection
