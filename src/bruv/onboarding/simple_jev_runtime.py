"""Managed Simple Jev runtime service.

Installs, launches, and controls an isolated local Simple Jev server inside a
bruv-owned data directory. The upstream checkout, virtual environment, model
cache, state files, and logs all live under one managed root so nothing leaks
into the bruv installation itself.

Subprocess execution, process identity, health polling, port probing, and
timing are injected through :class:`RuntimeDependencies`, so tests exercise
the full lifecycle without installing Simple Jev, downloading models, or
starting real processes.

Every failure raised here is an unpaid :class:`bruv.application.ConfigurationError`
subclass with a concrete recovery action.
"""

from __future__ import annotations

import ctypes
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from platformdirs import user_data_dir

from bruv.application import ConfigurationError

APP_NAME = "bruv"
MANAGED_DIR_NAME = "simple-jev"

#: Pinned upstream Simple Jev source revision (featherless-ai/simple-jev).
SOURCE_REVISION = "b02aa81c915a8193759b3cd33fef74721d6e005b"
UPSTREAM_URL = "https://github.com/featherless-ai/simple-jev.git"

DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000

MANIFEST_SCHEMA = 1
PID_SCHEMA = 1
HEALTH_PATH = "/health"
SERVER_SCRIPT_PARTS = ("hf-server", "hf_server.py")

REPAIR_ACTION = "Run `bruv setup simple-jev --repair` to rebuild the managed runtime state."
SETUP_ACTION = "Run `bruv setup simple-jev` to install the managed runtime."

_PY_312 = (3, 12)
_CUDA_PROBE_TIMEOUT_SECONDS = 600.0
_COMMAND_STDERR_EXCERPT_LIMIT = 600


# ---------------------------------------------------------------------------
# Errors


@dataclass(frozen=True, slots=True)
class SimpleJevRuntimeError(ConfigurationError):
    """Base for managed Simple Jev runtime failures. Always unpaid."""

    code: str = "simple_jev_runtime_error"


@dataclass(frozen=True, slots=True)
class ManifestError(SimpleJevRuntimeError):
    """Managed manifest is missing required fields or is unreadable."""

    code: str = "simple_jev_manifest_invalid"


@dataclass(frozen=True, slots=True)
class PidStateError(SimpleJevRuntimeError):
    """Managed PID metadata is unreadable or malformed."""

    code: str = "simple_jev_pid_invalid"


@dataclass(frozen=True, slots=True)
class NotInstalledError(SimpleJevRuntimeError):
    """Managed runtime has never been installed."""

    code: str = "simple_jev_not_installed"


@dataclass(frozen=True, slots=True)
class InstallError(SimpleJevRuntimeError):
    """Install or repair failed."""

    code: str = "simple_jev_install_failed"


@dataclass(frozen=True, slots=True)
class StartError(SimpleJevRuntimeError):
    """Server could not be started or did not become healthy in time."""

    code: str = "simple_jev_start_failed"


@dataclass(frozen=True, slots=True)
class StartupLockError(SimpleJevRuntimeError):
    """Another bruv process holds the managed startup lock."""

    code: str = "simple_jev_locked"


@dataclass(frozen=True, slots=True)
class UnsafeStopError(SimpleJevRuntimeError):
    """Stored PID could not be matched to the expected Simple Jev command."""

    code: str = "simple_jev_stop_refused"


@dataclass(frozen=True, slots=True)
class NotRunningError(SimpleJevRuntimeError):
    """Stop requested but no managed server is running."""

    code: str = "simple_jev_not_running"


@dataclass(frozen=True, slots=True)
class StopFailedError(SimpleJevRuntimeError):
    """Managed server survived termination attempts; PID state is preserved."""

    code: str = "simple_jev_stop_failed"


# ---------------------------------------------------------------------------
# Paths and settings


@dataclass(frozen=True, slots=True)
class SimpleJevPaths:
    """Filesystem layout of the managed runtime under one root."""

    root: Path

    @property
    def source(self) -> Path:
        return self.root / "source"

    @property
    def venv_dir(self) -> Path:
        return self.root / "venv"

    @property
    def venv_python(self) -> Path:
        if os.name == "nt":
            return self.venv_dir / "Scripts" / "python.exe"
        return self.venv_dir / "bin" / "python"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def pid_file(self) -> Path:
        return self.root / "simple-jev.pid"

    @property
    def lock_dir(self) -> Path:
        return self.root / "startup.lock"

    @property
    def log_file(self) -> Path:
        return self.root / "simple-jev.log"

    @property
    def model_cache(self) -> Path:
        return self.root / "huggingface"

    @property
    def server_script(self) -> Path:
        return self.source.joinpath(*SERVER_SCRIPT_PARTS)

    @classmethod
    def default(cls) -> "SimpleJevPaths":
        return cls(
            root=Path(user_data_dir(APP_NAME, appauthor=False)) / MANAGED_DIR_NAME
        )


@dataclass(frozen=True, slots=True)
class ManagedSimpleJevSettings:
    """Immutable defaults for the managed Simple Jev runtime.

    ``device`` accepts ``auto`` (probe managed torch and fall back to CPU),
    ``cpu``, or ``cuda``. ``dtype`` must be one of the upstream hf-server
    choices. Context and batch fields mirror upstream hf-server arguments.
    """

    model: str = DEFAULT_MODEL
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    device: str = "auto"
    dtype: str = "bfloat16"
    source_revision: str = SOURCE_REVISION
    upstream_url: str = UPSTREAM_URL
    context_length: int = 16384
    max_batch_size: int = 32
    max_batch_tokens: int = 32768
    max_request_branches: int = 100

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("model must be a non-empty model identifier")
        if self.dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError(
                f"dtype must be one of float32, float16, bfloat16; got {self.dtype!r}"
            )
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError(
                f"device must be one of auto, cpu, cuda; got {self.device!r}"
            )
        if not 1 <= self.port <= 65535:
            raise ValueError(f"port must be in 1..65535; got {self.port}")
        if self.context_length < 1 or self.max_batch_size < 1 or self.max_batch_tokens < 1:
            raise ValueError("context_length, max_batch_size, max_batch_tokens must be positive")
        if self.max_request_branches < 1:
            raise ValueError("max_request_branches must be positive")

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


# ---------------------------------------------------------------------------
# Manifest and PID records


@dataclass(frozen=True, slots=True)
class ManagedManifest:
    """Read-only view of the installed managed runtime state."""

    source_revision: str
    model: str
    host: str
    port: int
    device: str
    dtype: str
    context_length: int
    max_batch_size: int
    max_batch_tokens: int
    max_request_branches: int
    venv_python: str
    source_dir: str
    installed_at: str
    updated_at: str

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass(frozen=True, slots=True)
class ManagedPidRecord:
    """Read-only view of the persisted launch metadata."""

    pid: int
    started_at: str
    command: tuple[str, ...]
    log_file: str
    base_url: str


_MANIFEST_REQUIRED_KEYS = (
    "source_revision",
    "model",
    "host",
    "port",
    "device",
    "dtype",
    "context_length",
    "max_batch_size",
    "max_batch_tokens",
    "max_request_branches",
    "venv_python",
    "source_dir",
    "installed_at",
    "updated_at",
)


# ---------------------------------------------------------------------------
# Injectable runtime dependencies


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Outcome of one injected command run."""

    returncode: int
    stdout: str = ""
    stderr: str = ""


CommandRunner = Callable[..., CommandResult]
"""Run ``command`` with optional ``cwd`` and ``timeout`` keyword arguments."""

ProcessSpawner = Callable[..., int]
"""Spawn a detached ``command`` with ``log_path`` and ``env`` keyword arguments; return pid."""

AliveCheck = Callable[[int], bool]
CommandLineLookup = Callable[[int], "str | None"]
TerminateProcess = Callable[[int], None]
ForceKillProcess = Callable[[int], None]
HealthProbe = Callable[[str], bool]
PortProbe = Callable[[str, int], bool]
SleepFn = Callable[[float], None]
MonotonicFn = Callable[[], float]


def _default_run(
    command: Sequence[str],
    cwd: Path | str | None = None,
    timeout: float | None = None,
) -> CommandResult:
    completed = subprocess.run(
        list(command),
        cwd=None if cwd is None else str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return CommandResult(
        returncode=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
    )


def _default_spawn(
    command: Sequence[str],
    log_path: Path,
    env: dict[str, str] | None = None,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = open(log_path, "ab")
    try:
        if os.name == "nt":
            process = subprocess.Popen(
                list(command),
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=log_handle,
                close_fds=True,
                env=env,
                creationflags=(
                    subprocess.DETACHED_PROCESS
                    | subprocess.CREATE_NEW_PROCESS_GROUP
                    | subprocess.CREATE_NO_WINDOW
                ),
            )
        else:
            process = subprocess.Popen(
                list(command),
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=log_handle,
                close_fds=True,
                start_new_session=True,
                env=env,
            )
        return process.pid
    finally:
        log_handle.close()


def _default_process_alive(pid: int) -> bool:
    if os.name == "nt":
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return exit_code.value == STILL_ACTIVE
            return False
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _default_command_line(pid: int) -> str | None:
    """Best-effort process command line lookup across platforms.

    Returns ``None`` when the process does not exist or the command line
    cannot be read. Callers must treat ``None`` as "identity unknown" and
    refuse to signal the process.
    """
    cmdline_file = Path(f"/proc/{pid}/cmdline")
    if cmdline_file.is_file():
        try:
            raw = cmdline_file.read_bytes()
        except OSError:
            raw = b""
        if raw:
            return raw.replace(b"\x00", b" ").decode("utf-8", "replace").strip()
    if os.name == "nt":
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_Process "
                f"-Filter 'ProcessId = {pid}').CommandLine",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            value = (result.stdout or "").strip()
            return value or None
        return None
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "command="],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        value = (result.stdout or "").strip()
        return value or None
    return None


def _default_terminate(pid: int) -> None:
    """Graceful termination: SIGTERM on POSIX, TerminateProcess on Windows."""
    if os.name == "nt":
        os.kill(pid, signal.SIGTERM)
    else:
        os.kill(pid, signal.SIGTERM)


def _default_force_kill(pid: int) -> None:
    """Forcibly terminate. No-op escalation on Windows (terminate is hard)."""
    if os.name == "nt":
        return
    os.kill(pid, signal.SIGKILL)


def _default_health_probe(base_url: str) -> bool:
    try:
        response = httpx.get(base_url + HEALTH_PATH, timeout=2.0)
    except httpx.HTTPError:
        return False
    return response.status_code == 200


def _default_port_probe(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


@dataclass(frozen=True, slots=True)
class RuntimeDependencies:
    """Narrow injectable boundaries for subprocess, process, and timing I/O."""

    run: CommandRunner = _default_run
    spawn: ProcessSpawner = _default_spawn
    process_alive: AliveCheck = _default_process_alive
    command_line: CommandLineLookup = _default_command_line
    terminate: TerminateProcess = _default_terminate
    force_kill: ForceKillProcess = _default_force_kill
    health_probe: HealthProbe = _default_health_probe
    port_responding: PortProbe = _default_port_probe
    sleep: SleepFn = time.sleep
    monotonic: MonotonicFn = time.monotonic
    wall_time: Callable[[], float] = time.time

    def now(self) -> str:
        return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Result types


@dataclass(frozen=True, slots=True)
class InstallReport:
    """Outcome of a successful install or repair."""

    model: str
    source_revision: str
    python_version: str
    device: str
    dtype: str
    cuda_available: bool
    preliminary_cuda_detected: bool
    source_dir: str
    venv_python: str
    base_url: str


@dataclass(frozen=True, slots=True)
class RuntimeStatus:
    """Point-in-time managed runtime status."""

    installed: bool
    running: bool
    healthy: bool
    pid: int | None
    model: str | None
    device: str | None
    source_revision: str | None
    base_url: str | None
    log_file: str | None
    issues: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class StartOutcome:
    """Result of :meth:`ManagedSimpleJevRuntime.start` / ``ensure_running``."""

    pid: int
    base_url: str
    reused: bool
    waited_seconds: float


@dataclass(frozen=True, slots=True)
class StopOutcome:
    """Result of :meth:`ManagedSimpleJevRuntime.stop`."""

    pid: int | None
    stopped: bool
    stale: bool


# ---------------------------------------------------------------------------
# Helpers


def _excerpt(text: str, limit: int = _COMMAND_STDERR_EXCERPT_LIMIT) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[-limit:]


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` as JSON at ``path`` via temp file plus ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    try:
        dir_fd = os.open(path.parent, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    except OSError:
        pass
    finally:
        os.close(dir_fd)


def _preliminary_cuda_detected() -> bool:
    """Host-level NVIDIA hint for setup preview. Not authoritative.

    The authoritative CUDA check runs through the managed venv with
    ``torch.cuda.is_available()`` during install.
    """
    if shutil.which("nvidia-smi") is not None:
        return True
    return Path("/proc/driver/nvidia/version").is_file()


def _python_version_string() -> str:
    return ".".join(str(part) for part in sys.version_info[:3])


def _matches_identity(command_line: str, venv_python: Path) -> bool:
    python_token = str(venv_python)
    if os.name == "nt":
        haystack = command_line.lower()
        needle = python_token.lower()
    else:
        haystack = command_line
        needle = python_token
    return needle in haystack and "hf_server" in haystack


# ---------------------------------------------------------------------------
# Runtime service


class ManagedSimpleJevRuntime:
    """Install, launch, and control the managed local Simple Jev server."""

    def __init__(
        self,
        settings: ManagedSimpleJevSettings | None = None,
        *,
        paths: SimpleJevPaths | None = None,
        dependencies: RuntimeDependencies | None = None,
        startup_timeout: float = 600.0,
        stop_grace: float = 10.0,
        lock_stale_after: float = 180.0,
    ) -> None:
        self.settings = settings if settings is not None else ManagedSimpleJevSettings()
        self.paths = paths if paths is not None else SimpleJevPaths.default()
        self.dependencies = (
            dependencies if dependencies is not None else RuntimeDependencies()
        )
        self.startup_timeout = startup_timeout
        self.stop_grace = stop_grace
        self.lock_stale_after = lock_stale_after

    # -- manifest -----------------------------------------------------------

    def read_manifest(self) -> ManagedManifest | None:
        """Return the installed manifest, or ``None`` when not installed.

        Raises :class:`ManifestError` when the file exists but is malformed,
        so corrupted state never silently pretends to be healthy.
        """
        path = self.paths.manifest
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ManifestError(
                message=(
                    f"Managed Simple Jev manifest at {path} is unreadable: "
                    f"{exc.__class__.__name__}"
                ),
                action=REPAIR_ACTION,
            ) from exc
        if not isinstance(payload, dict) or payload.get("schema") != MANIFEST_SCHEMA:
            raise ManifestError(
                message=(
                    f"Managed Simple Jev manifest at {path} has an unknown or "
                    "missing schema marker."
                ),
                action=REPAIR_ACTION,
            )
        missing = [key for key in _MANIFEST_REQUIRED_KEYS if key not in payload]
        if missing:
            raise ManifestError(
                message=(
                    f"Managed Simple Jev manifest at {path} is missing required "
                    f"fields: {', '.join(missing)}."
                ),
                action=REPAIR_ACTION,
            )
        return ManagedManifest(
            source_revision=str(payload["source_revision"]),
            model=str(payload["model"]),
            host=str(payload["host"]),
            port=int(payload["port"]),
            device=str(payload["device"]),
            dtype=str(payload["dtype"]),
            context_length=int(payload["context_length"]),
            max_batch_size=int(payload["max_batch_size"]),
            max_batch_tokens=int(payload["max_batch_tokens"]),
            max_request_branches=int(payload["max_request_branches"]),
            venv_python=str(payload["venv_python"]),
            source_dir=str(payload["source_dir"]),
            installed_at=str(payload["installed_at"]),
            updated_at=str(payload["updated_at"]),
        )

    def _require_manifest(self) -> ManagedManifest:
        manifest = self.read_manifest()
        if manifest is None:
            raise NotInstalledError(
                message="Managed Simple Jev is not installed.",
                action=SETUP_ACTION,
            )
        return manifest

    def _write_manifest(self, manifest: ManagedManifest) -> None:
        _atomic_write_json(
            self.paths.manifest,
            {
                "schema": MANIFEST_SCHEMA,
                **{
                    key: getattr(manifest, key)
                    for key in ManagedManifest.__dataclass_fields__
                },
            },
        )

    # -- pid file -----------------------------------------------------------

    def _read_pid_record(self) -> ManagedPidRecord | None:
        path = self.paths.pid_file
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PidStateError(
                message=(
                    f"Managed Simple Jev PID file at {path} is unreadable: "
                    f"{exc.__class__.__name__}"
                ),
                action=REPAIR_ACTION,
            ) from exc
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != PID_SCHEMA
            or not isinstance(payload.get("pid"), int)
        ):
            raise PidStateError(
                message=f"Managed Simple Jev PID file at {path} is malformed.",
                action=REPAIR_ACTION,
            )
        command = payload.get("command")
        return ManagedPidRecord(
            pid=int(payload["pid"]),
            started_at=str(payload.get("started_at", "")),
            command=tuple(str(item) for item in command) if isinstance(command, list) else (),
            log_file=str(payload.get("log_file", str(self.paths.log_file))),
            base_url=str(payload.get("base_url", "")),
        )

    def _clear_pid(self) -> None:
        try:
            self.paths.pid_file.unlink()
        except FileNotFoundError:
            pass

    def _pid_is_ours(self, record: ManagedPidRecord) -> bool:
        """True only when the live process matches the expected Simple Jev command."""
        if not self.dependencies.process_alive(record.pid):
            return False
        command_line = self.dependencies.command_line(record.pid)
        if command_line is None:
            return False
        return _matches_identity(command_line, self.paths.venv_python)

    # -- install ------------------------------------------------------------

    def install(self, *, repair: bool = False) -> InstallReport:
        """Install or repair the managed runtime and write the manifest.

        Steps: validate the host Python, detect NVIDIA preliminarily, clone or
        fetch the pinned upstream source, create an isolated venv, install
        ``./hf-server`` editable, probe ``torch.cuda.is_available()`` through
        the managed Python, fall back to CPU safely when CUDA is unusable, and
        persist an atomic manifest. Repair rebuilds the venv and manifest.
        """
        self._validate_host_python()
        deps = self.dependencies
        preliminary_cuda = _preliminary_cuda_detected()
        if repair:
            self._repair_clear_broken_state()
        self._ensure_source()
        venv_python = self._ensure_venv(repair=repair)
        cuda_intent = self.settings.device == "cuda" or (
            self.settings.device == "auto" and preliminary_cuda
        )
        self._pip_install(venv_python, cuda_intent=cuda_intent)
        cuda_available = self._probe_cuda(venv_python)
        device = self._resolve_device(cuda_available)
        dtype = self._resolve_dtype(device)
        manifest = ManagedManifest(
            source_revision=self.settings.source_revision,
            model=self.settings.model,
            host=self.settings.host,
            port=self.settings.port,
            device=device,
            dtype=dtype,
            context_length=self.settings.context_length,
            max_batch_size=self.settings.max_batch_size,
            max_batch_tokens=self.settings.max_batch_tokens,
            max_request_branches=self.settings.max_request_branches,
            venv_python=str(self.paths.venv_python),
            source_dir=str(self.paths.source),
            installed_at=deps.now(),
            updated_at=deps.now(),
        )
        self._write_manifest(manifest)
        return InstallReport(
            model=manifest.model,
            source_revision=manifest.source_revision,
            python_version=_python_version_string(),
            device=device,
            dtype=dtype,
            cuda_available=cuda_available,
            preliminary_cuda_detected=preliminary_cuda,
            source_dir=str(self.paths.source),
            venv_python=str(self.paths.venv_python),
            base_url=manifest.base_url,
        )

    def repair(self) -> InstallReport:
        """Alias for :meth:`install` with repair enabled."""
        return self.install(repair=True)

    def _validate_host_python(self) -> None:
        if sys.version_info < _PY_312:
            current = _python_version_string()
            raise InstallError(
                message=(
                    "Managed Simple Jev requires Python 3.12 or newer; the "
                    f"running interpreter is {current}."
                ),
                action=(
                    "Install Python 3.12 or newer, reinstall bruv into it, and "
                    "run `bruv setup simple-jev` again."
                ),
            )

    def _git(self) -> str:
        git = shutil.which("git")
        if git is None:
            raise InstallError(
                message="git is required to install the managed Simple Jev source.",
                action="Install git and run `bruv setup simple-jev` again.",
            )
        return git

    def _run_checked(
        self,
        command: Sequence[str],
        *,
        cwd: Path | None = None,
        timeout: float | None = None,
        failure_message: str,
    ) -> CommandResult:
        result = self.dependencies.run(command, cwd=cwd, timeout=timeout)
        if result.returncode != 0:
            raise InstallError(
                message=(
                    f"{failure_message} (exit {result.returncode}): "
                    f"{_excerpt(result.stderr or result.stdout)}"
                ),
                action=REPAIR_ACTION,
            )
        return result

    def _ensure_source(self) -> None:
        git = self._git()
        source = self.paths.source
        source.parent.mkdir(parents=True, exist_ok=True)
        revision = self.settings.source_revision
        if not source.exists():
            self._run_checked(
                [git, "clone", self.settings.upstream_url, str(source)],
                failure_message="Could not clone the Simple Jev upstream repository",
            )
            self._run_checked(
                [git, "-C", str(source), "checkout", "--detach", revision],
                failure_message=f"Could not check out pinned revision {revision}",
            )
            self._verify_checked_out_revision()
            return
        if not (source / ".git").is_dir():
            raise InstallError(
                message=(
                    f"Managed source directory {source} exists but is not a git "
                    "checkout."
                ),
                action=(
                    f"Remove {source} and run `bruv setup simple-jev --repair`."
                ),
            )
        # Refresh the checkout to the pinned revision. Fetching the exact
        # revision first keeps traffic small; fall back to a full fetch when
        # the server refuses direct SHA fetches.
        fetch_pin = self.dependencies.run(
            [git, "-C", str(source), "fetch", "origin", revision],
            timeout=300.0,
        )
        if fetch_pin.returncode != 0:
            self._run_checked(
                [git, "-C", str(source), "fetch", "origin"],
                failure_message="Could not fetch Simple Jev upstream source",
                timeout=600.0,
            )
        self._run_checked(
            [git, "-C", str(source), "checkout", "--detach", revision],
            failure_message=f"Could not check out pinned revision {revision}",
        )
        self._verify_checked_out_revision()

    def _verify_checked_out_revision(self) -> None:
        """Confirm the checkout HEAD equals the pinned source revision."""
        git = self._git()
        revision = self.settings.source_revision
        head = self._run_checked(
            [git, "-C", str(self.paths.source), "rev-parse", "HEAD"],
            failure_message="Could not verify the checked-out Simple Jev revision",
        )
        checked_out = head.stdout.strip()
        if checked_out != revision:
            raise InstallError(
                message=(
                    f"Simple Jev source is at revision {checked_out!r}, expected "
                    f"{revision!r}."
                ),
                action=REPAIR_ACTION,
            )

    def _ensure_venv(self, *, repair: bool) -> Path:
        venv_python = self.paths.venv_python
        if repair and self.paths.venv_dir.is_dir():
            try:
                shutil.rmtree(self.paths.venv_dir)
            except OSError as exc:
                raise InstallError(
                    message=(
                        "Could not remove the previous managed virtual "
                        f"environment at {self.paths.venv_dir}: "
                        f"{exc.__class__.__name__}."
                    ),
                    action=(
                        f"Remove {self.paths.venv_dir} manually (close any "
                        "running managed server first), then run "
                        "`bruv setup simple-jev --repair`."
                    ),
                ) from exc
        if not venv_python.is_file():
            self._run_checked(
                [sys.executable, "-m", "venv", str(self.paths.venv_dir)],
                failure_message="Could not create the managed Simple Jev virtual environment",
                timeout=300.0,
            )
        if not venv_python.is_file():
            raise InstallError(
                message=(
                    f"Managed virtual environment was created but {venv_python} "
                    "is missing."
                ),
                action=REPAIR_ACTION,
            )
        return venv_python

    def _pip_install(self, venv_python: Path, *, cuda_intent: bool) -> None:
        """Install the hf-server package into the managed venv.

        Without CUDA intent, bootstrap a CPU-only torch wheel from the
        pytorch CPU index first so the resolver never pulls the default
        (often CUDA-mismatched) torch build. With CUDA intent, let pip's
        normal resolver install torch with the rest of hf-server. All pip
        commands run with ``--no-cache-dir`` so bruv never leaves a
        persistent pip cache behind.
        """
        if cuda_intent:
            self._run_checked(
                [
                    str(venv_python),
                    "-m",
                    "pip",
                    "install",
                    "--no-cache-dir",
                    "-e",
                    "./hf-server",
                ],
                cwd=self.paths.source,
                timeout=1800.0,
                failure_message="Could not install the Simple Jev hf-server package",
            )
            return
        self._run_checked(
            [
                str(venv_python),
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                "torch>=2.6",
                "--index-url",
                "https://download.pytorch.org/whl/cpu",
            ],
            cwd=self.paths.source,
            timeout=1800.0,
            failure_message="Could not install the CPU-only torch bootstrap package",
        )
        self._run_checked(
            [
                str(venv_python),
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                "-e",
                "./hf-server",
            ],
            cwd=self.paths.source,
            timeout=1800.0,
            failure_message="Could not install the Simple Jev hf-server package",
        )

    def _probe_cuda(self, venv_python: Path) -> bool:
        result = self.dependencies.run(
            [
                str(venv_python),
                "-c",
                "import torch; print(int(torch.cuda.is_available()))",
            ],
            timeout=_CUDA_PROBE_TIMEOUT_SECONDS,
        )
        if result.returncode != 0:
            raise InstallError(
                message=(
                    "The managed Simple Jev Python could not run torch "
                    f"({_excerpt(result.stderr or result.stdout)})."
                ),
                action=REPAIR_ACTION,
            )
        availability = result.stdout.strip()
        if availability not in {"0", "1"}:
            raise InstallError(
                message=(
                    "The managed torch CUDA probe returned unexpected output: "
                    f"{_excerpt(result.stdout)}."
                ),
                action=REPAIR_ACTION,
            )
        return availability == "1"

    def _resolve_device(self, cuda_available: bool) -> str:
        device = self.settings.device
        if device == "cpu":
            return "cpu"
        if device == "cuda" and not cuda_available:
            return "cpu"
        if device == "auto":
            return "cuda" if cuda_available else "cpu"
        return device

    def _resolve_dtype(self, device: str) -> str:
        """Resolve the launch dtype from the resolved device.

        CPU always runs ``float32`` because upstream hf-server weights do not
        support half precision on CPU. CUDA uses the requested dtype
        (default ``bfloat16``).
        """
        if device == "cpu":
            return "float32"
        return self.settings.dtype

    def _repair_clear_broken_state(self) -> None:
        """Clear malformed or dead PID state and a stale startup lock.

        Repair removes a PID file only when it is unreadable/malformed or its
        recorded process is no longer alive. A well-formed record whose
        process is still alive is never cleared or signaled: repair must not
        touch a running server or an unrelated live process.
        """
        deps = self.dependencies
        try:
            record = self._read_pid_record()
        except PidStateError:
            self._clear_pid()
        else:
            if record is not None and not deps.process_alive(record.pid):
                self._clear_pid()
        lock_dir = self.paths.lock_dir
        if lock_dir.is_dir():
            holder_pid = self._lock_holder_pid(lock_dir)
            holder_age = self._lock_holder_age(lock_dir)
            stale = (
                (holder_pid is not None and not deps.process_alive(holder_pid))
                or holder_age is None
                or holder_age > self.lock_stale_after
            )
            if stale:
                shutil.rmtree(lock_dir, ignore_errors=True)

    # -- status -------------------------------------------------------------

    def status(self) -> RuntimeStatus:
        """Return point-in-time managed runtime status without mutating state."""
        manifest = self.read_manifest()
        if manifest is None:
            return RuntimeStatus(
                installed=False,
                running=False,
                healthy=False,
                pid=None,
                model=None,
                device=None,
                source_revision=None,
                base_url=None,
                log_file=str(self.paths.log_file),
                issues=("Managed Simple Jev is not installed.",),
            )
        issues: list[str] = []
        pid_record: ManagedPidRecord | None = None
        try:
            pid_record = self._read_pid_record()
        except PidStateError:
            issues.append(
                f"Managed PID file at {self.paths.pid_file} is malformed; run "
                "`bruv setup simple-jev --repair` to clear it (stop refuses to "
                "signal an unreadable PID record)."
            )
        running = False
        healthy = False
        if pid_record is not None:
            running = self._pid_is_ours(pid_record)
            if not running and self.dependencies.process_alive(pid_record.pid):
                issues.append(
                    f"PID file points at process {pid_record.pid}, which is not "
                    "the managed Simple Jev command; bruv will not signal it. "
                    "Verify the process and, once it is safe, remove "
                    f"{self.paths.pid_file} manually."
                )
            elif not self.dependencies.process_alive(pid_record.pid):
                issues.append(
                    f"Stale PID file: process {pid_record.pid} is not running; "
                    "it will be cleaned up on the next start."
                )
            else:
                healthy = self.dependencies.health_probe(
                    pid_record.base_url or f"http://{manifest.host}:{manifest.port}"
                )
        return RuntimeStatus(
            installed=True,
            running=running,
            healthy=healthy,
            pid=pid_record.pid if pid_record is not None else None,
            model=manifest.model,
            device=manifest.device,
            source_revision=manifest.source_revision,
            base_url=f"http://{manifest.host}:{manifest.port}",
            log_file=str(self.paths.log_file),
            issues=tuple(issues),
        )

    # -- start --------------------------------------------------------------

    def start(self) -> StartOutcome:
        """Start the managed server, or reuse an already healthy instance.

        Cleans up stale PID files, diagnoses occupied ports, and guards the
        launch with an atomic directory lock so two bruv processes never race
        to start the same server.
        """
        manifest = self._require_manifest()
        deps = self.dependencies
        base_url = manifest.base_url
        existing = self._read_pid_record()
        if existing is not None:
            if self._pid_is_ours(existing):
                waited = self._wait_until_healthy(base_url, self.startup_timeout)
                if self.dependencies.health_probe(base_url):
                    return StartOutcome(
                        pid=existing.pid,
                        base_url=base_url,
                        reused=True,
                        waited_seconds=waited,
                    )
                raise StartError(
                    message=(
                        "Managed Simple Jev process "
                        f"{existing.pid} is running but did not become healthy "
                        f"after {waited:.0f} seconds."
                    ),
                    action=(
                        f"Inspect the managed log at {self.paths.log_file}. If "
                        "the server is stuck, run `bruv serve simple-jev stop` "
                        "and then `bruv serve simple-jev start`."
                    ),
                )
            if deps.process_alive(existing.pid):
                command_line = deps.command_line(existing.pid)
                raise StartError(
                    message=(
                        "PID file at "
                        f"{self.paths.pid_file} points at process "
                        f"{existing.pid}, which is not the managed Simple Jev "
                        "command."
                    ),
                    action=(
                        "Confirm the process is not needed and stop it manually "
                        "if it is yours, then remove "
                        f"{self.paths.pid_file} and run "
                        "`bruv serve simple-jev start` again. bruv never signals "
                        "or removes a PID record whose process is alive."
                    ),
                    details={"pid": existing.pid, "command_line": command_line},
                )
            # Stale PID from a crashed or killed server.
            self._clear_pid()
        if deps.port_responding(manifest.host, manifest.port):
            reused = self._reuse_healthy_managed(base_url)
            if reused is not None:
                return reused
            raise StartError(
                message=(
                    f"Port {manifest.port} on {manifest.host} is already in use "
                    "by another process; bruv did not start Simple Jev."
                ),
                action=(
                    "Stop the process using the port, or point bruv at the "
                    "existing server by setting SIMPLE_JEV_BASE_URL, or change "
                    "the managed port."
                ),
            )
        with self._startup_lock():
            if deps.port_responding(manifest.host, manifest.port):
                reused = self._reuse_healthy_managed(base_url)
                if reused is not None:
                    return reused
                raise StartError(
                    message=(
                        f"Port {manifest.port} on {manifest.host} became busy "
                        "while waiting to start; bruv did not start Simple Jev."
                    ),
                    action=(
                        "Another bruv or Simple Jev process took the port; run "
                        "`bruv serve simple-jev status` and retry."
                    ),
                )
            command = self._server_command(manifest)
            pid = deps.spawn(command, self.paths.log_file, env=self._server_env())
            self._write_pid_record(pid, command, base_url)
            waited = self._wait_until_healthy(base_url, self.startup_timeout)
        if self.dependencies.health_probe(base_url):
            return StartOutcome(
                pid=pid,
                base_url=base_url,
                reused=False,
                waited_seconds=waited,
            )
        alive = deps.process_alive(pid)
        if not alive:
            self._clear_pid()
        raise StartError(
            message=(
                f"Managed Simple Jev (pid {pid}) did not become healthy after "
                f"{waited:.0f} seconds"
                + (" and the process exited." if not alive else ".")
            ),
            action=(
                f"Inspect the managed log at {self.paths.log_file}. "
                + (
                    "Then run `bruv serve simple-jev start` again."
                    if not alive
                    else "If the server is stuck, run `bruv serve simple-jev stop`."
                )
            ),
            details={"log_file": str(self.paths.log_file)},
        )

    def _server_command(self, manifest: ManagedManifest) -> tuple[str, ...]:
        """Build the upstream hf-server command with matching arguments."""
        script = self.paths.server_script
        if not script.is_file():
            raise NotInstalledError(
                message=(
                    f"Upstream server script is missing at {script}; the managed "
                    "source install is incomplete."
                ),
                action=REPAIR_ACTION,
            )
        return (
            str(self.paths.venv_python),
            str(script),
            "--model",
            manifest.model,
            "--device",
            manifest.device,
            "--dtype",
            manifest.dtype,
            "--max-model-len",
            str(manifest.context_length),
            "--max-batch-size",
            str(manifest.max_batch_size),
            "--max-batch-tokens",
            str(manifest.max_batch_tokens),
            "--max-request-branches",
            str(manifest.max_request_branches),
            "--host",
            manifest.host,
            "--port",
            str(manifest.port),
        )

    def _server_env(self) -> dict[str, str]:
        """Child environment isolates model downloads under the managed root."""
        env = dict(os.environ)
        env["HF_HOME"] = str(self.paths.model_cache)
        return env

    def _write_pid_record(self, pid: int, command: Sequence[str], base_url: str) -> None:
        _atomic_write_json(
            self.paths.pid_file,
            {
                "schema": PID_SCHEMA,
                "pid": pid,
                "started_at": self.dependencies.now(),
                "command": list(command),
                "log_file": str(self.paths.log_file),
                "base_url": base_url,
            },
        )

    def _reuse_healthy_managed(self, base_url: str) -> StartOutcome | None:
        """Return a reused outcome only for a provably managed healthy server.

        The responding port is adopted only when the persisted PID record
        identifies a live process whose command line matches the managed
        Simple Jev command. Without that identity evidence the caller must
        treat the port as occupied: bruv never adopts or signals an
        unrelated process.
        """
        deps = self.dependencies
        if not deps.health_probe(base_url):
            return None
        try:
            record = self._read_pid_record()
        except PidStateError:
            return None
        if record is None or not self._pid_is_ours(record):
            return None
        return StartOutcome(
            pid=record.pid,
            base_url=base_url,
            reused=True,
            waited_seconds=0.0,
        )

    def _wait_until_healthy(self, base_url: str, timeout: float) -> float:
        """Poll ``/health`` until healthy or ``timeout`` elapses.

        Returns the actual elapsed wait in seconds, including on timeout, so
        outcomes report real elapsed time rather than the nominal budget.
        """
        deps = self.dependencies
        started = deps.monotonic()
        deadline = started + timeout
        while True:
            if deps.health_probe(base_url):
                return max(0.0, deps.monotonic() - started)
            remaining = deadline - deps.monotonic()
            if remaining <= 0:
                return max(0.0, deps.monotonic() - started)
            deps.sleep(min(1.0, remaining))

    @contextmanager
    def _startup_lock(self) -> Iterator[None]:
        """Atomic directory lock preventing duplicate simultaneous launches."""
        deps = self.dependencies
        lock_dir = self.paths.lock_dir
        deadline = deps.monotonic() + self.startup_timeout
        while True:
            try:
                lock_dir.mkdir(parents=True, exist_ok=False)
                break
            except FileExistsError:
                holder_age = self._lock_holder_age(lock_dir)
                holder_pid = self._lock_holder_pid(lock_dir)
                stale = (
                    (holder_pid is not None and not deps.process_alive(holder_pid))
                    or holder_age is None
                    or holder_age > self.lock_stale_after
                )
                if stale:
                    shutil.rmtree(lock_dir, ignore_errors=True)
                    continue
                if deps.monotonic() >= deadline:
                    raise StartupLockError(
                        message=(
                            "Another bruv process holds the managed Simple Jev "
                            f"startup lock at {lock_dir}."
                        ),
                        action=(
                            "Wait for the other operation to finish. If no bruv "
                            f"process is running, remove {lock_dir} and retry."
                        ),
                        details={"holder_pid": holder_pid},
                    )
                deps.sleep(0.2)
        try:
            _atomic_write_json(
                lock_dir / "holder.json",
                {"pid": os.getpid(), "acquired_at": deps.now()},
            )
            yield
        finally:
            shutil.rmtree(lock_dir, ignore_errors=True)

    def _lock_holder_pid(self, lock_dir: Path) -> int | None:
        holder = lock_dir / "holder.json"
        try:
            payload = json.loads(holder.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        pid = payload.get("pid") if isinstance(payload, dict) else None
        return int(pid) if isinstance(pid, int) else None

    def _lock_holder_age(self, lock_dir: Path) -> float | None:
        try:
            return max(0.0, self.dependencies.wall_time() - lock_dir.stat().st_mtime)
        except OSError:
            return None

    # -- stop ---------------------------------------------------------------

    def stop(self) -> StopOutcome:
        """Stop the managed server, refusing unless the PID identity matches.

        The stored PID must be live and its command line must contain the
        managed venv Python and the upstream ``hf_server`` script. Anything
        else refuses to signal and reports an actionable error.
        """
        record = self._read_pid_record()
        if record is None:
            raise NotRunningError(
                message="No managed Simple Jev server is running.",
                action="Run `bruv serve simple-jev start` to start it.",
            )
        deps = self.dependencies
        if not deps.process_alive(record.pid):
            self._clear_pid()
            return StopOutcome(pid=record.pid, stopped=False, stale=True)
        command_line = deps.command_line(record.pid)
        if command_line is None:
            raise UnsafeStopError(
                message=(
                    f"Could not read the command line of process {record.pid}; "
                    "bruv refuses to signal a process it cannot identify."
                ),
                action=(
                    "Verify process "
                    f"{record.pid} manually and stop it yourself, then remove "
                    f"{self.paths.pid_file} and run `bruv serve simple-jev start`."
                ),
                details={"pid": record.pid},
            )
        if not _matches_identity(command_line, self.paths.venv_python):
            raise UnsafeStopError(
                message=(
                    f"Process {record.pid} does not match the managed Simple Jev "
                    "command; bruv refuses to signal unknown processes."
                ),
                action=(
                    "Confirm the process owner and stop it manually if it is "
                    f"yours, then remove {self.paths.pid_file} and run "
                    "`bruv serve simple-jev start`."
                ),
                details={"pid": record.pid, "command_line": command_line},
            )
        deps.terminate(record.pid)
        deadline = deps.monotonic() + self.stop_grace
        while deps.process_alive(record.pid) and deps.monotonic() < deadline:
            deps.sleep(0.2)
        if deps.process_alive(record.pid):
            deps.force_kill(record.pid)
            deadline = deps.monotonic() + self.stop_grace
            while deps.process_alive(record.pid) and deps.monotonic() < deadline:
                deps.sleep(0.2)
        if deps.process_alive(record.pid):
            raise StopFailedError(
                message=(
                    f"Managed Simple Jev process {record.pid} is still alive "
                    "after terminate and force-kill grace periods; bruv did "
                    "not stop it and kept the PID file."
                ),
                action=(
                    f"Inspect process {record.pid} manually and kill it if it "
                    f"is safe to do so, then run `bruv serve simple-jev stop` "
                    "again or remove "
                    f"{self.paths.pid_file} once the process is gone."
                ),
                details={"pid": record.pid},
            )
        self._clear_pid()
        return StopOutcome(pid=record.pid, stopped=True, stale=False)

    # -- ensure_running -----------------------------------------------------

    def ensure_running(self) -> StartOutcome:
        """Idempotent entry point for the backend factory and CLI setup.

        Reuses a healthy managed server, starts a stopped one, and raises an
        actionable unpaid error when the runtime is missing or broken.
        """
        manifest = self._require_manifest()
        status = self.status()
        if status.running and status.healthy:
            if status.pid is None:
                raise SimpleJevRuntimeError(
                    message=(
                        "Managed Simple Jev status reported a running healthy "
                        "server without a PID; managed state is inconsistent."
                    ),
                    action=REPAIR_ACTION,
                )
            return StartOutcome(
                pid=status.pid,
                base_url=manifest.base_url,
                reused=True,
                waited_seconds=0.0,
            )
        return self.start()


__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_MODEL",
    "DEFAULT_PORT",
    "InstallError",
    "InstallReport",
    "ManagedManifest",
    "ManagedPidRecord",
    "ManagedSimpleJevRuntime",
    "ManagedSimpleJevSettings",
    "ManifestError",
    "NotInstalledError",
    "NotRunningError",
    "PidStateError",
    "RuntimeDependencies",
    "RuntimeStatus",
    "SERVER_SCRIPT_PARTS",
    "SOURCE_REVISION",
    "SimpleJevPaths",
    "SimpleJevRuntimeError",
    "StartError",
    "StartOutcome",
    "StartupLockError",
    "StopFailedError",
    "StopOutcome",
    "UnsafeStopError",
    "UPSTREAM_URL",
]