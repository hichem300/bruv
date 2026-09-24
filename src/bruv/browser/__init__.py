"""Backend-agnostic browser automation built on canonical Noul/Choice."""

from bruv.browser.actions import BrowserAction, origin_of
from bruv.browser.models import (
    BrowserGoal,
    BrowserReviewKind,
    BrowserStatus,
    ElementFingerprint,
    ExecutedAction,
    Observation,
    PendingRequest,
    SessionArtifacts,
    SessionState,
    TrustedCandidate,
    assert_no_supplied_text,
)

__all__ = [
    "BrowserAction",
    "BrowserGoal",
    "BrowserReviewKind",
    "BrowserStatus",
    "ElementFingerprint",
    "ExecutedAction",
    "Observation",
    "PendingRequest",
    "SessionArtifacts",
    "SessionState",
    "TrustedCandidate",
    "assert_no_supplied_text",
    "origin_of",
]
