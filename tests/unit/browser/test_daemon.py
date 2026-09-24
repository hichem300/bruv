"""Unit tests for the loopback browser daemon over real 127.0.0.1 HTTP."""

from __future__ import annotations

import json
import os
import stat
import threading
from http.client import HTTPConnection
from types import SimpleNamespace

import pytest

from bruv.browser.controller import BrowserControllerError
from bruv.browser.daemon import (
    BrowserDaemon,
    ConnectionInfo,
    DaemonClient,
    DaemonError,
    read_connection_file,
    write_connection_file,
)
from bruv.browser.models import BrowserGoal, PendingRequest, SessionState
from bruv.browser.sessions import SessionStore


def make_state(session_id: str, status: str = "needs_text") -> SessionState:
    return SessionState(
        session_id=session_id,
        status=status,
        goal=BrowserGoal(goal_id="g-1", goal_ref="goals/g-1", initial_url="https://example.com"),
        backend="default",
    )


class FakeController:
    """Minimal controller-compatible double. No browsers, no providers."""

    def __init__(self, session_id, url, goal_text, on_state, backend=None, options=None):
        self.session_id = session_id
        self.url = url
        self.goal_text = goal_text
        self.on_state = on_state
        self.backend = backend
        self.options = options
        self.run_thread_name = None
        self.reply_log = []
        self._stop = threading.Event()
        self._state = make_state(
            session_id
        ).model_copy(  # PendingRequest is immutable; pending text request
            update={
                "pending_request": PendingRequest(
                    request_id="r-1", kind="text", prompt_ref="prompts/r-1"
                )
            }
        )

    @property
    def state(self):
        return self._state

    def run(self):
        self.run_thread_name = threading.current_thread().name
        self._stop.wait(timeout=10)

    def stop(self):
        self._stop.set()
        self._state = self._state.model_copy(update={"status": "stopped"})

    def reply(self, request_id, text):
        if request_id != "r-1" or not text.strip():
            raise BrowserControllerError("no matching pending text request")
        self.reply_log.append((request_id, text))
        self._state = self._state.model_copy(update={"status": "running", "pending_request": None})
        self.on_state(self._state)


@pytest.fixture()
def env(tmp_path):
    store = SessionStore(tmp_path / "store")
    connection_path = tmp_path / "daemon.json"
    daemon = BrowserDaemon(
        lambda **kwargs: FakeController(**kwargs),
        store=store,
        connection_path=connection_path,
        max_body_bytes=256,
    )
    server_thread = daemon.start_background()
    info = read_connection_file(connection_path)
    yield SimpleNamespace(
        daemon=daemon,
        store=store,
        connection_path=connection_path,
        server_thread=server_thread,
        info=info,
    )
    daemon.shutdown()
    server_thread.join(timeout=5)


def _client(env):
    return DaemonClient(env.connection_path, timeout=5.0)


def _request(env, method, path, *, headers=None, body=None, skip_length=False):
    connection = HTTPConnection("127.0.0.1", env.info.port, timeout=5)
    try:
        connection.putrequest(method, path)
        connection.putheader("Authorization", f"Bearer {env.info.token}")
        for key, value in (headers or {}).items():
            connection.putheader(key, value)
        if not skip_length and body is not None:
            connection.putheader("Content-Length", str(len(body)))
        connection.endheaders()
        if body is not None:
            connection.send(body)
        response = connection.getresponse()
        raw = response.read()
        return response.status, json.loads(raw.decode("utf-8"))
    finally:
        connection.close()


# -- connection file --------------------------------------------------------------


def test_connection_file_round_trip_permissions_and_no_temp(tmp_path):
    path = tmp_path / "sub" / "daemon.json"
    info = ConnectionInfo(port=45678, pid=123, token="A" * 32)
    write_connection_file(info, path)
    write_connection_file(info, path)
    assert read_connection_file(path) == info
    assert not list(path.parent.glob("*.tmp"))
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_connection_file_rejects_bad_files(tmp_path):
    path = tmp_path / "daemon.json"
    good = (json.dumps(ConnectionInfo(port=1, pid=1, token="B" * 16).to_payload()) + "\n").encode()
    cases = {
        "malformed": b"{ broken\n",
        "noncanonical": b'{"token":"' + b"B" * 16 + b'","pid":1,"port":1}\n',
        "oversize": b"x" * 8192,
    }
    for raw in cases.values():
        path.write_bytes(raw)
        with pytest.raises(DaemonError):
            read_connection_file(path)
    path.write_bytes(good)
    link = tmp_path / "link.json"
    link.symlink_to(path)
    with pytest.raises(DaemonError):
        read_connection_file(link)


def test_connection_file_error_never_contains_token(tmp_path):
    path = tmp_path / "daemon.json"
    token = "S" * 32
    info = ConnectionInfo(port=1, pid=1, token=token)
    write_connection_file(info, path)
    path.write_bytes(b"{ broken\n")
    try:
        read_connection_file(path)
    except DaemonError as exc:
        assert token not in str(exc)
    else:
        raise AssertionError("expected failure")
    link = tmp_path / "link.json"
    link.symlink_to(path)
    try:
        read_connection_file(link)
    except DaemonError as exc:
        assert token not in str(exc)
    else:
        raise AssertionError("expected failure")


# -- daemon HTTP --------------------------------------------------------------------


def test_bearer_required(env):
    connection = HTTPConnection("127.0.0.1", env.info.port, timeout=5)
    try:
        connection.request("GET", "/sessions")
        response = connection.getresponse()
        raw = response.read()
    finally:
        connection.close()
    assert response.status == 401
    assert env.info.token not in raw.decode("utf-8")


def test_origin_rejected(env):
    status, payload = _request(env, "GET", "/sessions", headers={"Origin": "https://evil.example"})
    assert status == 403
    assert payload["ok"] is False


def test_oversized_content_length_rejected(env):
    status, _ = _request(
        env,
        "POST",
        "/sessions",
        headers={"Content-Length": "257"},
        body=b"{",
    )
    assert status == 413


def test_missing_content_length_rejected(env):
    connection = HTTPConnection("127.0.0.1", env.info.port, timeout=5)
    try:
        connection.putrequest("POST", "/sessions")
        connection.putheader("Authorization", f"Bearer {env.info.token}")
        connection.putheader("Content-Type", "application/json")
        connection.endheaders()
        response = connection.getresponse()
        response.read()
    finally:
        connection.close()
    assert response.status == 400


def test_unsupported_method_requires_auth_then_405(env):
    connection = HTTPConnection("127.0.0.1", env.info.port, timeout=5)
    try:
        connection.request("DELETE", "/sessions")
        response = connection.getresponse()
        response.read()
    finally:
        connection.close()
    assert response.status == 401
    status, _ = _request(env, "DELETE", "/sessions")
    assert status == 405


def test_create_persists_text_free_state_with_dedicated_worker(env):
    client = _client(env)
    state = client.create_session("https://example.com", "SECRET-GOAL-TEXT")
    assert state.session_id.startswith("s-")
    stored = env.store.load(state.session_id)
    assert stored == state
    assert "SECRET-GOAL-TEXT" not in (env.store.root / f"{state.session_id}.json").read_text()
    live = env.daemon._live[state.session_id]
    live._stop.set()
    live._stop.wait(timeout=5)
    assert live.run_thread_name == f"bruv-browser-{state.session_id}"


def test_get_latest_and_by_id(env):
    client = _client(env)
    created = client.create_session("https://example.com", "goal")
    assert client.state().session_id == created.session_id
    assert client.state(created.session_id) == created


def test_reply_resolves_pending_in_memory_and_not_persisted(env):
    client = _client(env)
    created = client.create_session("https://example.com", "SECRET-GOAL")
    state = client.reply("SECRET-REPLY-TEXT", created.session_id)
    assert state.status == "running"
    assert state.pending_request is None
    controller = env.daemon._live[created.session_id]
    assert controller.reply_log == [("r-1", "SECRET-REPLY-TEXT")]
    raw_file = (env.store.root / f"{created.session_id}.json").read_text()
    assert "SECRET-REPLY-TEXT" not in raw_file
    assert "SECRET-GOAL" not in raw_file


def test_stop_transitions_state(env):
    client = _client(env)
    created = client.create_session("https://example.com", "goal")
    state = client.stop(created.session_id)
    assert state.status == "stopped"


def test_shutdown_stops_serving_and_removes_connection(env):
    client = _client(env)
    client.create_session("https://example.com", "goal")
    client.shutdown()
    env.server_thread.join(timeout=5)
    assert not env.server_thread.is_alive()
    assert not env.connection_path.exists()
