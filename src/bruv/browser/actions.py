"""Validated browser action schemas.

Every field is bounded and strict. Actions carry only opaque candidate IDs,
refs, and safe metadata. The model never supplies URLs, selectors, text
payloads, or code; typed text resolves from refs at execution time only.
"""

from __future__ import annotations

from typing import Annotated, Literal, TypeAlias
from urllib.parse import urlsplit

from pydantic import Field, field_validator

from bruv.domain.questions import CanonicalModel, NonBlankString

MAX_WAIT_SECONDS = 10.0
MAX_SCROLL_AMOUNT = 3000


class ClickAction(CanonicalModel):
    kind: Literal["click"] = "click"
    candidate_id: NonBlankString = Field(max_length=64)


class TypeAction(CanonicalModel):
    kind: Literal["type"] = "type"
    candidate_id: NonBlankString = Field(max_length=64)
    # Reference into memory-only text (goal data or user reply). Never raw text.
    text_ref: NonBlankString = Field(max_length=128)


class NavigateAction(CanonicalModel):
    kind: Literal["navigate"] = "navigate"
    candidate_id: NonBlankString = Field(max_length=64)
    # Trusted internal target resolved from a runtime observation. Never
    # supplied by the model; validated against navigation_targets at execution.
    url: NonBlankString = Field(max_length=2048)

    @field_validator("url")
    @classmethod
    def _http_url_only(cls, value: str) -> str:
        stripped = value.strip()
        parts = urlsplit(stripped)
        if parts.scheme not in ("http", "https") or not parts.netloc:
            raise ValueError("navigation URLs must be absolute http(s) URLs with a hostname")
        try:
            hostname = parts.hostname
        except ValueError as exc:
            raise ValueError(
                "navigation URLs must be absolute http(s) URLs with a hostname"
            ) from exc
        if not hostname:
            raise ValueError("navigation URLs must be absolute http(s) URLs with a hostname")
        return value


class ScrollAction(CanonicalModel):
    kind: Literal["scroll"] = "scroll"
    direction: Literal["up", "down"]
    amount: int = Field(ge=1, le=MAX_SCROLL_AMOUNT)


class WaitAction(CanonicalModel):
    kind: Literal["wait"] = "wait"
    seconds: float = Field(gt=0, le=MAX_WAIT_SECONDS)


class RequestTextAction(CanonicalModel):
    kind: Literal["request_text"] = "request_text"
    prompt_ref: NonBlankString = Field(max_length=128)
    reason: str = Field(max_length=200)


class RequestReviewAction(CanonicalModel):
    kind: Literal["request_review"] = "request_review"
    candidate_id: NonBlankString = Field(max_length=64)
    review_kind: Literal[
        "login_submit",
        "file_transfer",
        "payment",
        "destructive",
        "public_posting",
        "message_submit",
        "permission_grant",
        "captcha",
    ]


BrowserAction: TypeAlias = Annotated[
    ClickAction
    | TypeAction
    | NavigateAction
    | ScrollAction
    | WaitAction
    | RequestTextAction
    | RequestReviewAction,
    Field(discriminator="kind"),
]


def origin_of(url: str) -> str:
    """Return scheme://host[:port] for an http(s) URL, lowercased, no path."""
    stripped = url.strip()
    lowered = stripped.lower()
    if lowered.startswith("https://"):
        scheme, rest = "https://", stripped[len("https://") :]
    elif lowered.startswith("http://"):
        scheme, rest = "http://", stripped[len("http://") :]
    else:
        raise ValueError(f"not an http(s) URL: {url!r}")
    authority = rest.split("/")[0].split("?")[0].split("#")[0]
    if not authority:
        raise ValueError(f"URL has no host: {url!r}")
    return scheme + authority.lower()


__all__ = [
    "ClickAction",
    "MAX_SCROLL_AMOUNT",
    "MAX_WAIT_SECONDS",
    "NavigateAction",
    "RequestReviewAction",
    "RequestTextAction",
    "ScrollAction",
    "TypeAction",
    "WaitAction",
    "BrowserAction",
    "origin_of",
]
