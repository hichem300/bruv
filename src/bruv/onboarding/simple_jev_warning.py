"""Deterministic known non-discriminating combo warning for Simple Jev.

Covers the confirmed combination: managed Simple Jev backend serving Noul
questions with model ``Qwen/Qwen3.5-0.8B``. This model/mode pair is known to
produce non-discriminating answers. The warning is advisory only: it never
changes results, is non-fatal, and fires once per managed runtime root.
Library use stays silent unless a warning sink is configured.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import IO

KNOWN_NON_DISCRIMINATING_MODEL = "Qwen/Qwen3.5-0.8B"
WARNING_MARKER_NAME = "nondiscriminating-warning.shown"

WARNING_MESSAGE = (
    "warning: Simple Jev with model Qwen/Qwen3.5-0.8B in Noul mode is known "
    "to produce non-discriminating answers; results may not separate inputs. "
    "Consider a different model or backend."
)

WarningSink = Callable[[str], None]


def is_known_non_discriminating(model: str, mode: str = "noul") -> bool:
    """Return True when the model/mode pair matches the known bad combo."""
    return mode.lower() == "noul" and model == KNOWN_NON_DISCRIMINATING_MODEL


def marker_path(root: os.PathLike[str] | str) -> os.PathLike[str] | str:
    """Return the once-only warning marker file path inside ``root``."""
    return os.path.join(os.fspath(root), WARNING_MARKER_NAME)


def claim_marker(root: os.PathLike[str] | str) -> bool:
    """Atomically create the once-only marker. True when claimed by us.

    Returns False when the marker already exists (warning already shown) or
    when creation fails (fail-open: caller may then warn again).
    """
    path = marker_path(root)
    try:
        os.makedirs(os.fspath(root), exist_ok=True)
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    except OSError:
        # Fail-open: persistence must never suppress the warning.
        return True
    try:
        os.write(fd, b"shown\n")
    finally:
        os.close(fd)
    return True


def emit_first_use_warning(
    *,
    model: str,
    mode: str,
    root: os.PathLike[str] | str,
    sink: WarningSink | None,
) -> bool:
    """Show the warning once via ``sink`` when the combo matches.

    Silent (returns False) when the combo does not match, the marker already
    exists, or ``sink`` is None. Never raises, never changes results.
    """
    if sink is None or not is_known_non_discriminating(model, mode):
        return False
    if not claim_marker(root):
        return False
    sink(WARNING_MESSAGE)
    return True


def stderr_sink(stream: IO[str] | None = None) -> WarningSink:
    """Return a sink writing to ``stream`` (default stderr)."""
    import sys

    target = stream if stream is not None else sys.stderr

    def _write(message: str) -> None:
        target.write(message if message.endswith("\n") else message + "\n")

    return _write


__all__ = [
    "KNOWN_NON_DISCRIMINATING_MODEL",
    "WARNING_MARKER_NAME",
    "WARNING_MESSAGE",
    "WarningSink",
    "claim_marker",
    "emit_first_use_warning",
    "is_known_non_discriminating",
    "marker_path",
    "stderr_sink",
]