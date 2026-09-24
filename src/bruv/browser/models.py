"""Immutable browser session contracts.

Supplied text (goal text, user replies) never lives in these models. Session
state persists references and safe metadata only; raw text stays in the live
process memory of the daemon/controller and is discarded after use.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Literal, TypeAlias

from pydantic import Field

from bruv.domain.questions import CanonicalModel, NonBlankString

BrowserStatus: TypeAlias = Literal[
    "running", "needs_text", "needs_review", "done", "failed", "stopped"
]

BrowserActionKind: TypeAlias = Literal[
    "click", "type", "navigate", "scroll", "wait", "request_text", "request_review"
]

BrowserReviewKind: TypeAlias = Literal[
    "login_submit",
    "file_transfer",
    "payment",
    "destructive",
    "public_posting",
    "message_submit",
    "permission_grant",
    "captcha",
]

PendingRequestKind: TypeAlias = Literal["text", "review"]

ExecutionOutcome: TypeAlias = Literal["ok", "failed"]

MAX_SESSION_ELEMENTS = 500
MAX_SESSION_ACTIONS = 1000


def _utc_now() -> datetime:
    return datetime.now(UTC)


class ElementFingerprint(CanonicalModel):
    """Stable, page-derived identity for one observed element."""

    fingerprint_id: NonBlankString
    tag: str = Field(min_length=1, max_length=32)
    role: str | None = Field(default=None, max_length=64)
    text_length: int = Field(ge=0, le=1_000_000)
    attributes_hash: str = Field(min_length=1, max_length=128)
    is_sensitive: bool = False


class TrustedCandidate(CanonicalModel):
    """One trusted, DOM-derived action candidate with an opaque ID."""

    candidate_id: NonBlankString = Field(max_length=64)
    kind: BrowserActionKind
    fingerprint: ElementFingerprint
    description: str = Field(max_length=200)
    needs_review_kind: BrowserReviewKind | None = None


class PendingRequest(CanonicalModel):
    """A paused request to the human. Only refs and metadata persist."""

    request_id: NonBlankString = Field(max_length=64)
    kind: PendingRequestKind
    prompt_ref: NonBlankString = Field(max_length=128)
    candidate_id: str | None = Field(default=None, max_length=64)
    review_kind: BrowserReviewKind | None = None
    created_at: datetime = Field(default_factory=_utc_now)


class Observation(CanonicalModel):
    """Bounded snapshot metadata. Page text persists as a ref, never inline."""

    observation_id: NonBlankString = Field(max_length=64)
    url: NonBlankString = Field(max_length=2048)
    title: str = Field(max_length=512)
    page_text_ref: NonBlankString = Field(max_length=128)
    element_count: int = Field(ge=0, le=1_000_000)
    elements: tuple[ElementFingerprint, ...] = Field(max_length=MAX_SESSION_ELEMENTS)
    captured_at: datetime = Field(default_factory=_utc_now)


class BrowserGoal(CanonicalModel):
    """Goal identity. The goal text itself is referenced, never stored."""

    goal_id: NonBlankString = Field(max_length=64)
    goal_ref: NonBlankString = Field(max_length=128)
    initial_url: NonBlankString = Field(max_length=2048)
    created_at: datetime = Field(default_factory=_utc_now)


class SessionArtifacts(CanonicalModel):
    """Paths of artifacts written for one session. No page content here."""

    screenshot_path: str | None = Field(default=None, max_length=1024)
    transcript_path: str | None = Field(default=None, max_length=1024)
    trace_path: str | None = Field(default=None, max_length=1024)
    video_path: str | None = Field(default=None, max_length=1024)


class ExecutedAction(CanonicalModel):
    """Record of one executed or rejected action. Action-only, text-free."""

    step: int = Field(ge=1, le=1_000_000)
    kind: BrowserActionKind
    description: str = Field(max_length=200)
    outcome: ExecutionOutcome
    timestamp: datetime = Field(default_factory=_utc_now)


class SessionState(CanonicalModel):
    """Persisted session metadata. Supplied text must never appear here."""

    session_id: NonBlankString = Field(max_length=64)
    status: BrowserStatus
    goal: BrowserGoal
    backend: NonBlankString = Field(max_length=64)
    origin_allowlist: tuple[Annotated[str, Field(max_length=2048)], ...] = ()
    observation: Observation | None = None
    pending_request: PendingRequest | None = None
    executed_actions: tuple[ExecutedAction, ...] = Field(default=(), max_length=MAX_SESSION_ACTIONS)
    artifacts: SessionArtifacts | None = None
    failure_reason: str | None = Field(default=None, max_length=200)
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)


def assert_no_supplied_text(state: SessionState, supplied_texts: tuple[str, ...]) -> None:
    """Guardrail helper: supplied text must not be persisted in the model.

    Structural safeguard: these models simply have no fields that could carry
    supplied text, so any leak would require a new text-bearing field. This
    helper is a defensive second layer over that structural guarantee.

    Conservative heuristic, applied recursively to serialized *string values*
    only (never JSON keys, so short words like ``goal`` or ``goal_ref`` cannot
    collide with field names):

    - any nonempty supplied string that exactly equals a serialized string
      value is rejected, regardless of length;
    - substring containment is rejected only for supplied strings of length
      >= 8, balancing leak detection against false positives for common short
      words inside safe refs and URLs.

    Error messages never include the supplied text itself.
    """

    def _collect_strings(value: object) -> None:
        if isinstance(value, str):
            strings.append(value)
        elif isinstance(value, dict):
            for item in value.values():
                _collect_strings(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                _collect_strings(item)

    strings: list[str] = []
    _collect_strings(state.model_dump(mode="json"))
    for text in supplied_texts:
        if not text:
            continue
        if len(text) < 8:
            if any(value == text for value in strings):
                raise ValueError("supplied text must never be persisted in session models")
        elif any(text in value for value in strings):
            raise ValueError("supplied text must never be persisted in session models")


__all__ = [
    "BrowserActionKind",
    "BrowserReviewKind",
    "BrowserGoal",
    "BrowserStatus",
    "ElementFingerprint",
    "ExecutedAction",
    "ExecutionOutcome",
    "MAX_SESSION_ACTIONS",
    "MAX_SESSION_ELEMENTS",
    "Observation",
    "PendingRequest",
    "PendingRequestKind",
    "SessionArtifacts",
    "SessionState",
    "TrustedCandidate",
    "assert_no_supplied_text",
]
