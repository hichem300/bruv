"""Session artifact storage for browser automation.

Session-specific, safe paths under a configured or default user-data artifact
directory. The transcript is action-only safe metadata: it never records page
text, goal text, reply text, URLs with query strings, or typed values.
Screenshots, traces, and videos may capture private page content; see
``PRIVACY_WARNING``.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from bruv.browser.actions import origin_of
from bruv.browser.models import SessionArtifacts
from bruv.config import default_browser_artifact_dir

PRIVACY_WARNING: Final[str] = (
    "Browser artifacts may capture private page content: screenshots, traces, "
    "and videos record what was visibly on the page. Store and share them "
    "with care."
)


def privacy_warning() -> str:
    """Return the standard privacy warning for browser artifacts."""
    return PRIVACY_WARNING


_SESSION_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

# Conservative safe single-basename pattern for optional artifact file names.
_SAFE_BASENAME_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def validate_session_id(session_id: str) -> str:
    """Validate a session ID as a single safe path component."""
    if ".." in session_id or not _SESSION_ID_PATTERN.match(session_id):
        raise ValueError(
            "session ID must be 1-64 characters of letters, digits, dot, dash, "
            "or underscore, starting with a letter or digit"
        )
    return session_id


def default_artifact_dir() -> Path:
    """Return the configured default artifact root for browser sessions."""
    return default_browser_artifact_dir()


def _validate_file_name(name: str, *, enabled: bool, label: str) -> str:
    """Validate an optional artifact file name as a safe single basename."""
    if not enabled:
        return name
    if ".." in name or "/" in name or "\\" in name or not _SAFE_BASENAME_PATTERN.match(name):
        raise ValueError(f"{label} must be a safe file name with no path separators")
    return name


@dataclass(frozen=True, slots=True)
class ArtifactOptions:
    """Optional artifact switches. Trace/video capture is off by default."""

    trace: bool = False
    video: bool = False
    trace_file_name: str = "trace.zip"
    video_file_name: str = "video.webm"

    def __post_init__(self) -> None:
        _validate_file_name(self.trace_file_name, enabled=self.trace, label="trace_file_name")
        _validate_file_name(self.video_file_name, enabled=self.video, label="video_file_name")


def _make_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    return path


class TranscriptWriter:
    """Append-only, action-only JSONL transcript with atomic replaces.

    Entries carry safe metadata only: timestamp, step, action kind, optional
    candidate ID, optional top-level origin, outcome, and an error class name.
    Callers must never pass page text, supplied text, or full URLs.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._entries: list[dict[str, str | int]] = []
        _make_directory(path.parent)
        self._flush()

    def record(
        self,
        *,
        step: int,
        kind: str,
        outcome: str,
        candidate_id: str | None = None,
        origin: str | None = None,
        error_kind: str | None = None,
    ) -> None:
        """Record one action attempt. Values must already be safe metadata."""
        entry: dict[str, str | int] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "step": step,
            "kind": kind,
            "outcome": outcome,
        }
        if candidate_id:
            entry["candidate_id"] = candidate_id
        if origin:
            entry["origin"] = origin
        if error_kind:
            entry["error"] = error_kind
        self._entries.append(entry)
        self._flush()

    def _flush(self) -> None:
        payload = "".join(
            json.dumps(entry, separators=(",", ":"), ensure_ascii=False) + "\n"
            for entry in self._entries
        )
        temp_path = self._path.with_name(self._path.name + ".tmp")
        fd = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
        os.replace(temp_path, self._path)


def safe_origin(url: str) -> str:
    """Return the top-level origin of a URL, or an empty string if invalid."""
    try:
        return origin_of(url)
    except ValueError:
        return ""


class ArtifactStore:
    """Session-scoped artifact paths under a configured or default root."""

    def __init__(
        self,
        session_id: str,
        artifact_dir: Path | None = None,
        options: ArtifactOptions | None = None,
    ) -> None:
        self.session_id = validate_session_id(session_id)
        self.options = options or ArtifactOptions()
        root = Path(artifact_dir) if artifact_dir is not None else default_artifact_dir()
        self.root = root / self.session_id

    def prepare(self) -> SessionArtifacts:
        """Create the restrictive session directory and declare safe paths."""
        _make_directory(self.root)
        trace_path = str(self.root / self.options.trace_file_name) if self.options.trace else None
        video_path = str(self.root / self.options.video_file_name) if self.options.video else None
        return SessionArtifacts(
            screenshot_path=str(self.root / "final-screenshot.png"),
            transcript_path=str(self.root / "transcript.jsonl"),
            trace_path=trace_path,
            video_path=video_path,
        )

    def transcript(self) -> TranscriptWriter:
        return TranscriptWriter(self.root / "transcript.jsonl")


__all__ = [
    "ArtifactOptions",
    "ArtifactStore",
    "PRIVACY_WARNING",
    "TranscriptWriter",
    "default_artifact_dir",
    "privacy_warning",
    "safe_origin",
    "validate_session_id",
]
