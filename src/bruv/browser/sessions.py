"""Atomic persisted session state for browser automation.

Session state files hold immutable, text-free ``SessionState`` metadata under
a per-user browser/sessions directory. Supplied text (goal text, user replies)
is never persisted; callers must pass it via ``supplied_texts`` so the store
can refuse to write if it ever appears in the payload.
"""

from __future__ import annotations

import os
import stat
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Final

import platformdirs
from pydantic import ValidationError

from bruv.browser.artifacts import validate_session_id
from bruv.browser.models import SessionState, assert_no_supplied_text

MAX_STATE_FILE_BYTES: Final[int] = 2 * 1024 * 1024


class SessionStoreError(Exception):
    """Raised when session state cannot be safely stored or read.

    Messages are intentionally generic so they are safe to surface without
    leaking file payloads, state contents, or exception details.
    """


class SessionStoreNotFoundError(SessionStoreError):
    """Raised when a requested session state file does not exist."""


def default_session_dir() -> Path:
    """Default session-state root under the bruv user-data directory."""
    return Path(platformdirs.user_data_dir("bruv", appauthor=False)) / "browser" / "sessions"


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


_OPEN_NOFOLLOW: Final[int] = getattr(os, "O_NOFOLLOW", 0)
_OPEN_NONBLOCK: Final[int] = getattr(os, "O_NONBLOCK", 0)


def _read_regular_file(path: Path) -> bytes:
    """Bounded, no-follow read of one regular state file. Fail closed.

    Rejects symlinks up front (cross-platform), opens without following where
    supported, and verifies a regular file via ``fstat``. Directories, FIFOs,
    and other non-regular files raise instead of blocking or being followed.
    """
    try:
        if path.is_symlink():
            raise SessionStoreError("session state file is not a regular file")
    except OSError:
        raise SessionStoreError("session state file could not be inspected") from None
    flags = os.O_RDONLY | _OPEN_NOFOLLOW | _OPEN_NONBLOCK
    try:
        fd = os.open(path, flags)
    except FileNotFoundError:
        raise SessionStoreNotFoundError("session state not found") from None
    except OSError:
        raise SessionStoreError("session state could not be read") from None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise SessionStoreError("session state file is not a regular file")
        if info.st_size > MAX_STATE_FILE_BYTES:
            raise SessionStoreError("session state file is too large")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_STATE_FILE_BYTES:
                raise SessionStoreError("session state file is too large")
            chunks.append(chunk)
    except SessionStoreError:
        raise
    except OSError:
        raise SessionStoreError("session state could not be read") from None
    finally:
        os.close(fd)
    return b"".join(chunks)


class SessionStore:
    """Atomic, fail-closed store for immutable SessionState metadata."""

    def __init__(self, root: str | os.PathLike[str] | None = None) -> None:
        self.root = Path(root) if root is not None else default_session_dir()
        try:
            self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        except OSError:
            raise SessionStoreError("session state directory could not be prepared") from None
        _apply_private_mode(self.root, is_dir=True)

    def _state_path(self, session_id: str) -> Path:
        try:
            validate_session_id(session_id)
        except ValueError:
            raise SessionStoreError("invalid session ID") from None
        return self.root / f"{session_id}.json"

    def save(
        self,
        state: SessionState,
        *,
        supplied_texts: tuple[str, ...] = (),
    ) -> None:
        """Atomically persist canonical JSON state; never persist supplied text.

        Raises SessionStoreError instead of writing when supplied text would be
        captured in the serialized state.
        """
        try:
            assert_no_supplied_text(state, supplied_texts)
        except ValueError:
            raise SessionStoreError("state would capture supplied text; write refused") from None

        path = self._state_path(state.session_id)
        payload = state.model_dump_json(indent=2) + "\n"

        # Unique O_EXCL temp file in the same directory: no fixed-name temp
        # symlink or clobber window. mkstemp creates the file 0600 on POSIX.
        try:
            fd, temp_name = tempfile.mkstemp(
                prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
            )
        except OSError:
            raise SessionStoreError("session state could not be saved") from None
        temp_path = Path(temp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
        except OSError:
            with suppress(OSError):
                temp_path.unlink()
            raise SessionStoreError("session state could not be saved") from None
        except BaseException:
            with suppress(OSError):
                temp_path.unlink()
            raise
        _apply_private_mode(path, is_dir=False)
        _fsync_directory(self.root)

    def _parse(self, path: Path, raw: bytes) -> SessionState:
        try:
            state = SessionState.model_validate_json(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValidationError):
            raise SessionStoreError("session state is not valid") from None
        # Canonical check: re-serialization must reproduce the stored bytes.
        canonical = state.model_dump_json(indent=2) + "\n"
        if canonical.encode("utf-8") != raw:
            raise SessionStoreError("session state file is not canonical")
        if state.session_id != path.stem:
            raise SessionStoreError("session state session ID does not match file")
        return state

    def load(self, session_id: str) -> SessionState:
        """Load and strictly validate one session state; fail closed."""
        path = self._state_path(session_id)
        raw = _read_regular_file(path)
        return self._parse(path, raw)

    def list_states(self) -> tuple[SessionState, ...]:
        """All valid session states, newest-updated first, ID tie break.

        Fails closed: any corrupt, oversized, or noncanonical state raises.
        """
        states: list[SessionState] = []
        try:
            entries = list(self.root.iterdir())
        except OSError:
            raise SessionStoreError("session states could not be listed") from None
        for entry in entries:
            if not entry.name.endswith(".json"):
                continue
            raw = _read_regular_file(entry)
            states.append(self._parse(entry, raw))
        states.sort(key=lambda s: (s.updated_at, s.session_id), reverse=True)
        return tuple(states)

    def latest(self) -> SessionState | None:
        """Most recently updated session state, or None when none exist."""
        states = self.list_states()
        return states[0] if states else None


__all__ = [
    "MAX_STATE_FILE_BYTES",
    "SessionStore",
    "SessionStoreError",
    "SessionStoreNotFoundError",
    "default_session_dir",
]
