"""Authenticated loopback daemon and client for browser sessions.

Security model:

- The daemon binds strictly to ``127.0.0.1`` on an ephemeral port and requires
  a per-run bearer token on every request, compared in constant time. Any
  request carrying an ``Origin`` header is rejected outright.
- The connection file (port/pid/token) is written atomically with restrictive
  permissions under the bruv user-data directory. Tokens are never logged and
  never appear in errors.
- POST bodies must declare a bounded ``Content-Length``; chunked or
  missing-length bodies are rejected. Only JSON objects are accepted.
- Raw goal text and reply text live only in request/closure memory. Persisted
  state is exclusively the text-free ``SessionState`` model; nothing is logged.
- Controller construction is fully injected. This module never constructs
  browsers, drivers, or provider backends; the injected factory receives the
  session ID, URL, goal text, optional backend/options, and an ``on_state``
  callback, and returns a controller-compatible object. ``run()`` is called
  on exactly one dedicated worker thread per session for the controller's
  whole life.
"""

from __future__ import annotations

import hmac
import json
import math
import os
import re
import secrets
import stat
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from http import HTTPStatus
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Final, Protocol
from urllib.parse import urlsplit

import platformdirs

from bruv.browser.artifacts import validate_session_id
from bruv.browser.controller import BrowserControllerError
from bruv.browser.models import SessionState
from bruv.browser.sessions import (
    SessionStore,
    SessionStoreError,
    SessionStoreNotFoundError,
)

DEFAULT_MAX_BODY_BYTES: Final[int] = 64 * 1024
MAX_CONNECTION_FILE_BYTES: Final[int] = 4096
MAX_CLIENT_RESPONSE_BYTES: Final[int] = 8 * 1024 * 1024

MAX_URL_CHARS: Final[int] = 2048
MAX_GOAL_CHARS: Final[int] = 20_000
MAX_REPLY_CHARS: Final[int] = 20_000
MAX_BACKEND_CHARS: Final[int] = 64

_WORKER_JOIN_TIMEOUT_SECONDS: Final[float] = 10.0

_TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9_-]{16,256}$")

_CANONICAL_JSON_KWARGS: Final[dict[str, Any]] = {
    "sort_keys": True,
    "separators": (",", ":"),
    "ensure_ascii": False,
}


class DaemonError(Exception):
    """Fail-closed daemon/client error. Messages are generic and text-safe."""


class _HTTPError(DaemonError):
    """Internal error carrying a bounded HTTP status and generic message."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def _generic_error(status: int) -> _HTTPError:
    messages: Final[dict[int, str]] = {
        int(HTTPStatus.BAD_REQUEST): "invalid request",
        int(HTTPStatus.UNAUTHORIZED): "unauthorized",
        int(HTTPStatus.FORBIDDEN): "forbidden",
        int(HTTPStatus.NOT_FOUND): "not found",
        int(HTTPStatus.METHOD_NOT_ALLOWED): "method not allowed",
        int(HTTPStatus.CONFLICT): "request conflicts with current session state",
        int(HTTPStatus.REQUEST_ENTITY_TOO_LARGE): "request body too large",
        int(HTTPStatus.INTERNAL_SERVER_ERROR): "internal error",
    }
    return _HTTPError(status, messages.get(status, "request failed"))


# -- connection file -----------------------------------------------------------


@dataclass(frozen=True)
class ConnectionInfo:
    """Immutable loopback daemon coordinates. Strictly validated."""

    port: int
    pid: int
    token: str

    def __post_init__(self) -> None:
        if isinstance(self.port, bool) or not isinstance(self.port, int):
            raise ValueError("port must be an integer")
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        if isinstance(self.pid, bool) or not isinstance(self.pid, int) or self.pid < 1:
            raise ValueError("pid must be a positive integer")
        if not isinstance(self.token, str) or not _TOKEN_PATTERN.match(self.token):
            raise ValueError("token has an invalid format")

    def to_payload(self) -> dict[str, Any]:
        return {"pid": self.pid, "port": self.port, "token": self.token}

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> ConnectionInfo:
        if set(data) != {"port", "pid", "token"}:
            raise ValueError("connection data has unexpected fields")
        return cls(port=data["port"], pid=data["pid"], token=data["token"])


def default_connection_path() -> Path:
    """Default connection file under the bruv user-data browser directory."""
    return Path(platformdirs.user_data_dir("bruv", appauthor=False)) / "browser" / "daemon.json"


def _apply_private_mode(path: Path, *, is_dir: bool) -> None:
    """Best-effort restrictive permissions; cross-platform safe."""
    try:
        mode = stat.S_IRWXU if is_dir else stat.S_IRUSR | stat.S_IWUSR
        os.chmod(path, mode)
    except OSError:
        pass


def _fsync_directory(path: Path) -> None:
    """Best-effort directory fsync (POSIX only; no-op elsewhere)."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _reject_symlink(path: Path, label: str) -> None:
    try:
        if path.is_symlink():
            raise DaemonError(f"{label} is not a regular file")
    except OSError:
        raise DaemonError(f"{label} could not be inspected") from None


def write_connection_file(info: ConnectionInfo, path: str | os.PathLike[str] | None = None) -> Path:
    """Atomically write canonical JSON connection data with restrictive modes.

    The parent directory gets ``0700`` and the file ``0600`` where supported.
    The token is never logged.
    """
    target = Path(path) if path is not None else default_connection_path()
    parent = target.parent
    try:
        parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    except OSError:
        raise DaemonError("connection file directory could not be prepared") from None
    _apply_private_mode(parent, is_dir=True)
    _reject_symlink(target, "connection file")
    if target.exists() and not target.is_file():
        raise DaemonError("connection path is not a regular file")

    encoded = (json.dumps(info.to_payload(), **_CANONICAL_JSON_KWARGS) + "\n").encode("utf-8")
    if len(encoded) > MAX_CONNECTION_FILE_BYTES:
        raise DaemonError("connection data is too large")

    # Unique O_EXCL temp file in the same parent: no fixed-name temp symlink
    # or clobber window. The descriptor is created 0600 by mkstemp.
    try:
        fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=parent)
    except OSError:
        raise DaemonError("connection file could not be written") from None
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(encoded.decode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, target)
    except OSError:
        with suppress(OSError):
            temp_path.unlink()
        raise DaemonError("connection file could not be written") from None
    except BaseException:
        with suppress(OSError):
            temp_path.unlink()
        raise
    _apply_private_mode(target, is_dir=False)
    _fsync_directory(parent)
    return target


def read_connection_file(path: str | os.PathLike[str] | None = None) -> ConnectionInfo:
    """Read and strictly validate the connection file. Fail closed on anything odd."""
    target = Path(path) if path is not None else default_connection_path()
    _reject_symlink(target, "connection file")
    try:
        raw = target.read_bytes()
    except FileNotFoundError:
        raise DaemonError("daemon connection file not found") from None
    except OSError:
        raise DaemonError("connection file could not be read") from None
    if len(raw) > MAX_CONNECTION_FILE_BYTES:
        raise DaemonError("connection file is too large")
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise DaemonError("connection file is not valid") from None
    canonical = (json.dumps(data, **_CANONICAL_JSON_KWARGS) + "\n").encode("utf-8")
    if canonical != raw:
        raise DaemonError("connection file is not canonical")
    if not isinstance(data, dict):
        raise DaemonError("connection file data is invalid")
    try:
        return ConnectionInfo.from_mapping(data)
    except (TypeError, ValueError):
        raise DaemonError("connection file data is invalid") from None


# -- controller protocol --------------------------------------------------------


class DaemonControlledBrowser(Protocol):
    """Controller-compatible surface the daemon drives from its worker thread."""

    @property
    def state(self) -> SessionState: ...

    def run(self) -> SessionState: ...

    def stop(self) -> None: ...

    def reply(self, request_id: str, text: str) -> None: ...


class BrowserControllerFactory(Protocol):
    """Injected factory: builds one controller per session. No browsers here."""

    def __call__(
        self,
        *,
        session_id: str,
        url: str,
        goal_text: str,
        on_state: Callable[[SessionState], None],
        backend: str | None,
        options: Mapping[str, Any] | None,
    ) -> DaemonControlledBrowser: ...


# -- daemon ----------------------------------------------------------------------


class BrowserDaemon:
    """Lean authenticated loopback daemon over an injected controller factory.

    ``start()`` binds to ``127.0.0.1:0``, writes the protected connection file,
    and serves until ``shutdown()`` (invoke it from another thread, or use
    ``POST /shutdown`` / ``DaemonClient.shutdown()``). One dedicated worker
    thread runs each controller's ``run()`` for the session's whole life; the
    controller is removed from the live map only after ``run()`` returns.
    """

    def __init__(
        self,
        controller_factory: BrowserControllerFactory,
        *,
        store: SessionStore | None = None,
        connection_path: str | os.PathLike[str] | None = None,
        max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
    ) -> None:
        self._factory = controller_factory
        self._store = store if store is not None else SessionStore()
        self._connection_path = (
            Path(connection_path) if connection_path is not None else default_connection_path()
        )
        if isinstance(max_body_bytes, bool) or not isinstance(max_body_bytes, int):
            raise DaemonError("max_body_bytes must be an integer")
        if max_body_bytes < 1:
            raise DaemonError("max_body_bytes must be positive")
        self._max_body_bytes = max_body_bytes

        self._token = ""
        self._server: ThreadingHTTPServer | None = None
        self._serving = False
        self._closing = False
        self._lifecycle_lock = threading.Lock()
        self._live_lock = threading.Lock()
        self._live: dict[str, DaemonControlledBrowser] = {}
        self._workers: list[threading.Thread] = []

    # -- lifecycle ---------------------------------------------------------------

    def _bind(self) -> None:
        """Bind, publish the connection file, and mark serving. Synchronous."""
        with self._lifecycle_lock:
            if self._serving or self._closing:
                raise DaemonError("daemon already started or shutting down")
        server = ThreadingHTTPServer(("127.0.0.1", 0), self._make_handler())
        server.daemon_threads = True
        token = secrets.token_urlsafe(32)
        port = int(server.server_address[1])
        try:
            write_connection_file(
                ConnectionInfo(port=port, pid=os.getpid(), token=token),
                self._connection_path,
            )
        except BaseException:
            server.server_close()
            raise
        with self._lifecycle_lock:
            if self._serving or self._closing:
                server.server_close()
                raise DaemonError("daemon already started or shutting down")
            self._server = server
            self._token = token
            self._serving = True

    def start(self) -> None:
        """Bind, publish the connection file, then serve. Blocks until shutdown."""
        self._bind()
        self._serve()

    def _serve(self) -> None:
        with self._lifecycle_lock:
            server = self._server
        if server is None:
            raise DaemonError("daemon is not bound")
        try:
            server.serve_forever(poll_interval=0.2)
        finally:
            with self._lifecycle_lock:
                self._serving = False
            if not self._closing:
                self._remove_connection_file()
                try:
                    server.server_close()
                except OSError:
                    pass

    def start_background(self) -> threading.Thread:
        """Bind synchronously, then serve on a daemon thread.

        Binding before returning guarantees the connection file exists (and
        bind errors surface to the caller) as soon as this method returns.
        """
        self._bind()
        thread = threading.Thread(target=self._serve, name="bruv-browser-daemon", daemon=True)
        thread.start()
        return thread

    def shutdown(self) -> None:
        """Idempotently stop controllers, the server, and workers. Never hangs.

        Must be called from a thread other than the one running ``start()``;
        the ``/shutdown`` endpoint arranges that automatically.
        """
        with self._lifecycle_lock:
            if self._closing:
                return
            self._closing = True
            server = self._server
            serving = self._serving
        with self._live_lock:
            controllers = list(self._live.values())
            workers = list(self._workers)
        for controller in controllers:
            with suppress(Exception):
                controller.stop()
        if server is not None:
            if serving:
                server.shutdown()
            try:
                server.server_close()
            except OSError:
                pass
        deadline = time.monotonic() + _WORKER_JOIN_TIMEOUT_SECONDS
        for worker in workers:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            worker.join(timeout=remaining)
        self._remove_connection_file()

    def _remove_connection_file(self) -> None:
        try:
            self._connection_path.unlink()
        except OSError:
            pass

    # -- session management ---------------------------------------------------------

    @staticmethod
    def _validate_url(url: object) -> None:
        """Absolute http/https URL with a hostname, else generic 400."""
        if not isinstance(url, str) or not url.strip() or len(url) > MAX_URL_CHARS:
            raise _generic_error(400)
        try:
            split = urlsplit(url.strip())
        except ValueError:
            raise _generic_error(400) from None
        if split.scheme not in ("http", "https") or not split.netloc:
            raise _generic_error(400)
        try:
            hostname = split.hostname
        except ValueError:
            raise _generic_error(400) from None
        if not hostname:
            raise _generic_error(400)

    def _create_session(self, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        url = body.get("url")
        goal = body.get("goal")
        backend = body.get("backend")
        options = body.get("options")
        self._validate_url(url)
        if not isinstance(goal, str) or not goal.strip() or len(goal) > MAX_GOAL_CHARS:
            raise _generic_error(400)
        if backend is not None and (
            not isinstance(backend, str) or not backend.strip() or len(backend) > MAX_BACKEND_CHARS
        ):
            raise _generic_error(400)
        if options is not None and not isinstance(options, dict):
            raise _generic_error(400)

        session_id = f"s-{uuid.uuid4().hex}"

        def on_state(state: SessionState) -> None:
            self._store.save(state, supplied_texts=(goal,))

        try:
            controller = self._factory(
                session_id=session_id,
                url=url,
                goal_text=goal,
                on_state=on_state,
                backend=backend,
                options=options,
            )
            self._store.save(controller.state, supplied_texts=(goal,))
        except Exception:
            raise _generic_error(500) from None

        with self._live_lock:
            self._live[session_id] = controller
            worker = threading.Thread(
                target=self._run_worker,
                args=(controller, session_id),
                name=f"bruv-browser-{session_id}",
                daemon=True,
            )
            self._workers.append(worker)
        worker.start()
        return int(HTTPStatus.OK), controller.state.model_dump(mode="json")

    def _run_worker(self, controller: DaemonControlledBrowser, session_id: str) -> None:
        """Own ``controller.run()`` on this one thread for the session's life."""
        try:
            controller.run()
        except Exception:  # noqa: S110 - terminal state already persisted via on_state
            pass
        finally:
            with self._live_lock:
                self._live.pop(session_id, None)
                # Prune this completed worker under the same lock so shutdown
                # snapshots only live threads.
                with suppress(ValueError):
                    self._workers.remove(threading.current_thread())

    def _get_state_response(self, session_id: str | None) -> tuple[int, dict[str, Any]]:
        try:
            state = self._store.load(session_id) if session_id else self._store.latest()
        except SessionStoreNotFoundError:
            raise _generic_error(404) from None
        except SessionStoreError:
            raise _generic_error(500) from None
        if state is None:
            raise _generic_error(404)
        return int(HTTPStatus.OK), state.model_dump(mode="json")

    def _reply(self, session_id: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        text = body.get("text")
        if not isinstance(text, str) or not text.strip() or len(text) > MAX_REPLY_CHARS:
            raise _generic_error(400)
        controller = self._live_controller(session_id)
        pending = controller.state.pending_request
        if pending is None or pending.kind != "text":
            raise _HTTPError(int(HTTPStatus.CONFLICT), "no pending text request")
        try:
            controller.reply(pending.request_id, text)
        except BrowserControllerError:
            raise _HTTPError(int(HTTPStatus.CONFLICT), "no matching pending text request") from None
        return int(HTTPStatus.OK), controller.state.model_dump(mode="json")

    def _stop(self, session_id: str) -> tuple[int, dict[str, Any]]:
        controller = self._live_controller(session_id)
        controller.stop()
        return int(HTTPStatus.OK), controller.state.model_dump(mode="json")

    def _live_controller(self, session_id: str) -> DaemonControlledBrowser:
        with self._live_lock:
            controller = self._live.get(session_id)
        if controller is None:
            raise _HTTPError(int(HTTPStatus.CONFLICT), "session is not active")
        return controller

    def _shutdown_endpoint(self) -> tuple[int, dict[str, Any]]:
        threading.Thread(target=self.shutdown, name="bruv-daemon-shutdown", daemon=True).start()
        return int(HTTPStatus.OK), {"shutting_down": True}

    # -- HTTP plumbing ----------------------------------------------------------------

    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        daemon = self

        class _Handler(BaseHTTPRequestHandler):
            server_version = "bruv-browser-daemon"
            sys_version = ""
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                """Emit nothing. Requests, bodies, and tokens are never logged."""

            def _send_json(self, status: int, payload: dict[str, Any]) -> None:
                encoded = json.dumps(payload, **_CANONICAL_JSON_KWARGS).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(encoded)

            def _reject(self, status: int, message: str) -> None:
                self._send_json(status, {"ok": False, "error": {"message": message}})

            def _authorized(self) -> bool:
                expected = f"Bearer {daemon._token}".encode()
                provided = self.headers.get("Authorization", "").encode("utf-8")
                return hmac.compare_digest(provided, expected)

            def _read_body_object(self) -> dict[str, Any]:
                if self.headers.get("Transfer-Encoding") is not None:
                    raise _generic_error(400)
                raw_length = self.headers.get("Content-Length")
                if raw_length is None:
                    raise _generic_error(400)
                try:
                    length = int(raw_length)
                except ValueError:
                    raise _generic_error(400) from None
                if length < 0:
                    raise _generic_error(400)
                if length > daemon._max_body_bytes:
                    raise _generic_error(413)
                body = self.rfile.read(length)
                if len(body) != length:
                    raise _generic_error(400)
                try:
                    data = json.loads(body.decode("utf-8"))
                except (UnicodeDecodeError, ValueError):
                    raise _generic_error(400) from None
                if not isinstance(data, dict):
                    raise _generic_error(400)
                return data

            def _route(self, method: str) -> tuple[str, str | None]:
                parts = [part for part in urlsplit(self.path).path.split("/") if part]
                if parts == ["shutdown"] and method == "POST":
                    return "POST /shutdown", None
                if not parts or parts[0] != "sessions":
                    raise _generic_error(404)
                if len(parts) == 1:
                    return f"{method} /sessions", None
                session_id = parts[1]
                try:
                    validate_session_id(session_id)
                except ValueError:
                    raise _generic_error(404) from None
                if len(parts) == 2:
                    return f"{method} /sessions/<id>", session_id
                if len(parts) == 3 and method == "POST" and parts[2] in ("reply", "stop"):
                    return f"POST /sessions/<id>/{parts[2]}", session_id
                raise _generic_error(404)

            def _dispatch(self, method: str) -> None:
                try:
                    if self.headers.get("Origin") is not None:
                        raise _generic_error(403)
                    if not self._authorized():
                        raise _generic_error(401)
                    route, session_id = self._route(method)
                    body = self._read_body_object() if method == "POST" else {}
                    status, data = self._handle(method, route, session_id, body)
                    self._send_json(status, {"ok": True, "data": data})
                except _HTTPError as exc:
                    self._reject(exc.status, exc.message)
                except DaemonError:
                    self._reject(int(HTTPStatus.INTERNAL_SERVER_ERROR), "internal error")
                except Exception:
                    self._reject(int(HTTPStatus.INTERNAL_SERVER_ERROR), "internal error")

            def _handle(
                self, method: str, route: str, session_id: str | None, body: dict[str, Any]
            ) -> tuple[int, dict[str, Any]]:
                if route == "POST /shutdown":
                    return daemon._shutdown_endpoint()
                if route == "POST /sessions":
                    return daemon._create_session(body)
                if route == "GET /sessions":
                    return daemon._get_state_response(None)
                if route == "GET /sessions/<id>":
                    return daemon._get_state_response(session_id)
                if route == "POST /sessions/<id>/reply":
                    return daemon._reply(session_id or "", body)
                if route == "POST /sessions/<id>/stop":
                    return daemon._stop(session_id or "")
                raise _generic_error(405)

            def do_GET(self) -> None:
                self._dispatch("GET")

            def do_POST(self) -> None:
                self._dispatch("POST")

            def do_HEAD(self) -> None:
                self._dispatch_unsupported()

            def do_PUT(self) -> None:
                self._dispatch_unsupported()

            def do_PATCH(self) -> None:
                self._dispatch_unsupported()

            def do_DELETE(self) -> None:
                self._dispatch_unsupported()

            def do_OPTIONS(self) -> None:
                self._dispatch_unsupported()

            def _dispatch_unsupported(self) -> None:
                """Reject Origin and require auth before revealing the 405."""
                try:
                    if self.headers.get("Origin") is not None:
                        raise _generic_error(403)
                    if not self._authorized():
                        raise _generic_error(401)
                    raise _generic_error(405)
                except _HTTPError as exc:
                    self._reject(exc.status, exc.message)
                except Exception:
                    self._reject(int(HTTPStatus.INTERNAL_SERVER_ERROR), "internal error")

        return _Handler


# -- client ------------------------------------------------------------------------


class DaemonClient:
    """Stdlib HTTP client for the browser daemon. Never logs request bodies."""

    def __init__(
        self,
        connection_path: str | os.PathLike[str] | None = None,
        *,
        timeout: float = 30.0,
    ) -> None:
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise DaemonError("timeout must be a number")
        if not math.isfinite(timeout) or timeout <= 0:
            raise DaemonError("timeout must be finite and positive")
        self._info = read_connection_file(connection_path)
        self._timeout = timeout

    def create_session(
        self,
        url: str,
        goal: str,
        *,
        backend: str | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> SessionState:
        if not isinstance(url, str) or not url.strip() or len(url) > MAX_URL_CHARS:
            raise DaemonError("invalid url")
        if not isinstance(goal, str) or not goal.strip() or len(goal) > MAX_GOAL_CHARS:
            raise DaemonError("invalid goal")
        payload: dict[str, Any] = {"url": url, "goal": goal}
        if backend is not None:
            payload["backend"] = backend
        if options is not None:
            payload["options"] = dict(options)
        return self._state_from(self._request("POST", "/sessions", payload))

    def state(self, session_id: str | None = None) -> SessionState:
        path = "/sessions" if session_id is None else f"/sessions/{session_id}"
        return self._state_from(self._request("GET", path))

    def reply(self, text: str, session_id: str | None = None) -> SessionState:
        if not isinstance(text, str) or not text.strip() or len(text) > MAX_REPLY_CHARS:
            raise DaemonError("invalid reply text")
        if session_id is None:
            session_id = self.state().session_id
        return self._state_from(
            self._request("POST", f"/sessions/{session_id}/reply", {"text": text})
        )

    def stop(self, session_id: str | None = None) -> SessionState:
        if session_id is None:
            session_id = self.state().session_id
        return self._state_from(self._request("POST", f"/sessions/{session_id}/stop", {}))

    def shutdown(self) -> None:
        self._request("POST", "/shutdown", {})

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        body: bytes | None = None
        headers = {
            "Authorization": f"Bearer {self._info.token}",
            "Accept": "application/json",
        }
        if payload is not None:
            body = json.dumps(payload, **_CANONICAL_JSON_KWARGS).encode("utf-8")
            headers["Content-Type"] = "application/json"
            headers["Content-Length"] = str(len(body))
        connection = HTTPConnection("127.0.0.1", self._info.port, timeout=self._timeout)
        try:
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            raw = response.read(MAX_CLIENT_RESPONSE_BYTES + 1)
            status = response.status
        except OSError as exc:
            raise DaemonError("daemon is unreachable") from exc
        finally:
            connection.close()
        if len(raw) > MAX_CLIENT_RESPONSE_BYTES:
            raise DaemonError("daemon response is too large")
        if status < 200 or status >= 300:
            raise DaemonError(self._error_message(raw, status))
        try:
            envelope = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise DaemonError("daemon response is not valid") from None
        if (
            not isinstance(envelope, dict)
            or envelope.get("ok") is not True
            or "data" not in envelope
        ):
            raise DaemonError("daemon response is not valid")
        return envelope["data"]

    @staticmethod
    def _error_message(raw: bytes, status: int) -> str:
        fallback = f"daemon request failed with status {status}"
        try:
            envelope = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            return fallback
        if not isinstance(envelope, dict):
            return fallback
        error = envelope.get("error")
        if not isinstance(error, dict):
            return fallback
        message = error.get("message")
        if not isinstance(message, str) or not message or len(message) > 200:
            return fallback
        return message

    @staticmethod
    def _state_from(data: Any) -> SessionState:
        try:
            # JSON-mode validation: dict input carries ISO datetime strings and
            # list-shaped tuples that plain python-mode validation rejects.
            return SessionState.model_validate_json(json.dumps(data))
        except Exception:
            raise DaemonError("daemon returned an invalid session state") from None


__all__ = [
    "DEFAULT_MAX_BODY_BYTES",
    "BrowserControllerFactory",
    "BrowserDaemon",
    "ConnectionInfo",
    "DaemonClient",
    "DaemonControlledBrowser",
    "DaemonError",
    "default_connection_path",
    "read_connection_file",
    "write_connection_file",
]
