"""Stable domain validation issue primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

IssuePathPart: TypeAlias = str | int
IssuePath: TypeAlias = tuple[IssuePathPart, ...]


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    code: str
    path: IssuePath
    message: str
