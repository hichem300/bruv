"""Runtime observation building: raw DOM metadata to bounded observations.

This module is the only place where raw, page-extracted metadata is held in
runtime memory. It converts unbounded DOM metadata into:

- a bounded, immutable Task 1 ``Observation`` summary (page text persists only
  as a deterministic hash ref, never inline),
- trusted ``TrustedCandidate`` entries with opaque deterministic IDs and
  structural fingerprints (no raw selectors),
- runtime-only metadata (bounded page text, resolved navigation targets) that
  the later driver/executor needs. These runtime fields are used to build a
  bounded backend input and are never persisted; model-visible output carries
  candidate IDs only.

Supplied text (goal text, user replies) never passes through here.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final
from urllib.parse import urljoin, urlsplit

from bruv.browser.actions import origin_of
from bruv.browser.models import (
    BrowserActionKind,
    BrowserReviewKind,
    ElementFingerprint,
    Observation,
    TrustedCandidate,
)

# Input types treated as editable text fields (request_text candidates).
_EDITABLE_INPUT_TYPES: Final[frozenset[str]] = frozenset(
    {
        "text",
        "email",
        "tel",
        "search",
        "number",
        "url",
        "date",
        "time",
        "datetime-local",
        "month",
        "week",
    }
)

# Input types that act as click controls.
_BUTTON_INPUT_TYPES: Final[frozenset[str]] = frozenset(
    {"button", "submit", "reset", "image", "checkbox", "radio"}
)

# Excluded input types: hidden, credential, and file transfer surfaces.
_EXCLUDED_INPUT_TYPES: Final[frozenset[str]] = frozenset({"hidden", "password", "file"})

# Sensitive autocomplete categories: payment card fields and one-time codes.
_SENSITIVE_AUTOCOMPLETE_PREFIXES: Final[tuple[str, ...]] = ("cc-",)
_SENSITIVE_AUTOCOMPLETE_VALUES: Final[frozenset[str]] = frozenset(
    {"one-time-code", "one-time-code-paragraph", "current-password", "new-password"}
)

# Non-secret structural attributes included in the fingerprint hash. Input
# values, styles, and free-form attributes are deliberately excluded.
_FINGERPRINT_ATTRIBUTES: Final[frozenset[str]] = frozenset(
    {"id", "name", "aria-label", "download", "target", "formaction", "list"}
)

# Stable synthetic candidate IDs for scroll/wait; exported so callers never
# hard-code magic IDs. Positions are fixed by the bounded specs below.
SCROLL_DOWN_CANDIDATE_ID: Final[str] = "s0000"
SCROLL_UP_CANDIDATE_ID: Final[str] = "s0001"
WAIT_CANDIDATE_ID: Final[str] = "s0002"

# Deterministic priority classes; lower sorts first, then original index.
_PRIORITY_EDITABLE: Final[int] = 0
_PRIORITY_CONTROL: Final[int] = 1
_PRIORITY_LINK: Final[int] = 2

_REVIEW_TEXT_PATTERNS: Final[tuple[tuple[BrowserReviewKind, tuple[str, ...]], ...]] = (
    (
        "captcha",
        (
            "captcha",
            "recaptcha",
            "hcaptcha",
            "i'm not a robot",
            "im not a robot",
            "verify you are human",
        ),
    ),
    (
        "payment",
        (
            "pay now",
            "checkout",
            "complete purchase",
            "place order",
            "confirm payment",
            "buy now",
            "subscribe now",
        ),
    ),
    (
        "login_submit",
        (
            "sign in",
            "log in",
            "log-in",
            "login",
            "sign-in",
        ),
    ),
    (
        "file_transfer",
        (
            "upload",
            "download",
            "import file",
            "export file",
        ),
    ),
    (
        "destructive",
        (
            "delete",
            "remove",
            "permanently",
            "erase",
            "cancel account",
            "reset all",
        ),
    ),
    (
        "public_posting",
        (
            "post",
            "publish",
            "tweet",
            "share publicly",
            "add comment",
            "post comment",
        ),
    ),
    (
        "permission_grant",
        (
            "allow",
            "grant",
            "enable notifications",
            "allow camera",
            "allow microphone",
            "allow location",
        ),
    ),
    (
        "message_submit",
        (
            "send message",
            "send",
            "submit",
        ),
    ),
)

# Review-kind precedence when several patterns match one element.
_REVIEW_PRECEDENCE: Final[tuple[BrowserReviewKind, ...]] = (
    "captcha",
    "payment",
    "login_submit",
    "file_transfer",
    "destructive",
    "public_posting",
    "permission_grant",
    "message_submit",
)


@dataclass(frozen=True, slots=True)
class RawDomElement:
    """Page-extracted metadata for one DOM element. Runtime memory only.

    ``attributes`` must contain page-derived structural metadata only; input
    values and other secrets must never be placed here.
    """

    index: int
    tag: str
    role: str | None = None
    text: str = ""
    input_type: str | None = None
    autocomplete: str | None = None
    href: str | None = None
    visible: bool = True
    disabled: bool = False
    attributes: Mapping[str, str] | None = None


@dataclass(frozen=True, slots=True)
class RuntimeObservation:
    """Runtime-only observation. Never persisted.

    ``page_text`` and candidate descriptions are runtime metadata used to
    build the bounded backend input; they are not part of persisted state and
    never reach persisted models. Only ``summary`` (an immutable Task 1
    ``Observation``) and ``candidates`` are safe for persistence;
    model-visible output carries candidate IDs only.
    """

    summary: Observation
    candidates: tuple[TrustedCandidate, ...]
    page_text: str
    navigation_targets: tuple[tuple[str, str], ...]
    # Runtime-only mapping of candidate ID to the raw extraction index of its
    # element in the page snapshot. Used by the executor to re-locate the live
    # element after fingerprint validation. Never persisted.
    candidate_elements: tuple[tuple[str, int], ...] = ()

    def target_for(self, candidate_id: str) -> str | None:
        """Return the resolved absolute URL for a navigate candidate, if any."""
        for cid, url in self.navigation_targets:
            if cid == candidate_id:
                return url
        return None

    def raw_index_for(self, candidate_id: str) -> int | None:
        """Return the raw extraction index for a candidate, if any.

        Synthetic candidates (scroll/wait) have no backing element and
        therefore no raw index.
        """
        for cid, raw_index in self.candidate_elements:
            if cid == candidate_id:
                return raw_index
        return None


def _bounded_text(value: str, limit: int) -> str:
    collapsed = " ".join(value.split())
    return collapsed[:limit]


def _page_text_ref(page_text: str) -> str:
    digest = hashlib.sha256(page_text.encode("utf-8")).hexdigest()
    return f"sha256:{digest[:32]}"


def _normalized(value: str | None) -> str:
    if value is None:
        return ""
    return " ".join(value.split()).strip().lower()


def _is_sensitive_autocomplete(autocomplete: str | None) -> bool:
    lowered = _normalized(autocomplete)
    if not lowered:
        return False
    if lowered in _SENSITIVE_AUTOCOMPLETE_VALUES:
        return True
    return lowered.startswith(_SENSITIVE_AUTOCOMPLETE_PREFIXES)


def _resolve_href(href: str, base_url: str) -> str:
    return urljoin(base_url, href.strip())


def _classify_input(input_type: str | None) -> str:
    """Classify an <input> type: editable, control, or other."""
    lowered = _normalized(input_type)
    if not lowered:
        return "editable"  # <input> without type defaults to text.
    if lowered in _EXCLUDED_INPUT_TYPES:
        return "excluded"
    if lowered in _EDITABLE_INPUT_TYPES:
        return "editable"
    if lowered in _BUTTON_INPUT_TYPES:
        return "control"
    return "control"


def _match_review_kind(
    text: str,
    *,
    input_type: str | None,
    download_flag: bool,
) -> BrowserReviewKind | None:
    lowered = _normalized(text)
    matched: set[BrowserReviewKind] = set()
    if download_flag:
        matched.add("file_transfer")
    if _normalized(input_type) == "reset":
        matched.add("destructive")
    for kind, patterns in _REVIEW_TEXT_PATTERNS:
        for pattern in patterns:
            if pattern in lowered:
                matched.add(kind)
                break
    for kind in _REVIEW_PRECEDENCE:
        if kind in matched:
            return kind
    return None


@dataclass(frozen=True, slots=True)
class _KeptElement:
    """Internal pairing of a raw element with its resolved link target."""

    raw: RawDomElement
    kind_class: str  # "editable", "control", or "link"
    resolved_href: str | None


class ObservationBuilder:
    """Deterministic converter of raw DOM metadata into bounded observations.

    Configured once per session with the bounded limits from ``AppConfig``
    (``browser_observation_max_elements``, ``browser_page_text_max_chars``) and
    the normalized origin allowlist (``browser_allowed_origins``). One ``build``
    call produces the observation for one page snapshot.
    """

    def __init__(
        self,
        max_elements: int,
        max_page_text_chars: int,
        allowed_origins: tuple[str, ...],
    ) -> None:
        if max_elements < 1:
            raise ValueError("max_elements must be at least 1")
        if max_page_text_chars < 1:
            raise ValueError("max_page_text_chars must be at least 1")
        self._max_elements = max_elements
        self._max_page_text_chars = max_page_text_chars
        self._allowed_origins: frozenset[str] = frozenset(
            origin.strip().lower() for origin in allowed_origins if origin.strip()
        )

    def build(
        self,
        url: str,
        title: str,
        page_text: str,
        elements: Sequence[RawDomElement],
    ) -> RuntimeObservation:
        """Build a bounded runtime observation from one page snapshot.

        Deterministic: same inputs yield the same summary, candidates, and
        runtime metadata. Review-kind candidates are returned but never
        executed here.
        """
        parts = urlsplit(url.strip())
        if parts.scheme not in ("http", "https") or not parts.netloc:
            raise ValueError(f"observation requires an absolute http(s) URL: {url!r}")
        base_url = url.strip()
        run_origin = origin_of(base_url)
        allowed_origins = frozenset({run_origin}) | self._allowed_origins

        bounded_page_text = _bounded_text(page_text, self._max_page_text_chars)

        kept = self._filter_and_rank(elements, base_url, allowed_origins)
        kept = kept[: self._max_elements]

        password_present = any(
            _normalized(element.input_type) == "password"
            for element in elements
            if element.visible and not element.disabled
        )

        fingerprints: list[ElementFingerprint] = []
        candidates: list[TrustedCandidate] = []
        navigation_targets: list[tuple[str, str]] = []
        candidate_elements: list[tuple[str, int]] = []
        seen_fingerprints: set[str] = set()

        for position, item in enumerate(kept):
            fingerprint = self._fingerprint(item.raw, seen_fingerprints)
            fingerprints.append(fingerprint)
            candidate_id = f"c{position:04d}"
            candidate = self._candidate(item, candidate_id, fingerprint, password_present)
            candidates.append(candidate)
            candidate_elements.append((candidate_id, item.raw.index))
            if item.kind_class == "link":
                navigation_targets.append((candidate_id, item.resolved_href or ""))

        candidates.extend(_synthetic_candidates())

        observation_seed = f"{base_url}|{_page_text_ref(bounded_page_text)}"
        summary = Observation(
            observation_id=f"obs-{hashlib.sha256(observation_seed.encode()).hexdigest()[:16]}",
            url=base_url,
            title=_bounded_text(title, 512),
            page_text_ref=_page_text_ref(bounded_page_text),
            element_count=len(fingerprints),
            elements=tuple(fingerprints),
        )

        return RuntimeObservation(
            summary=summary,
            candidates=tuple(candidates),
            page_text=bounded_page_text,
            navigation_targets=tuple(navigation_targets),
            candidate_elements=tuple(candidate_elements),
        )

    def _filter_and_rank(
        self,
        elements: Sequence[RawDomElement],
        base_url: str,
        allowed_origins: frozenset[str],
    ) -> list[_KeptElement]:
        kept: list[_KeptElement] = []
        for element in elements:
            if not element.visible or element.disabled:
                continue
            tag = _normalized(element.tag) or "unknown"
            input_class = _classify_input(element.input_type)
            autocomplete_sensitive = _is_sensitive_autocomplete(element.autocomplete)

            if tag == "input":
                if input_class == "excluded":
                    continue
                if autocomplete_sensitive and input_class == "editable":
                    continue
                kind_class = "editable" if input_class == "editable" else "control"
            elif tag == "textarea":
                if autocomplete_sensitive:
                    continue
                kind_class = "editable"
            elif tag in ("button", "select"):
                kind_class = "control"
            elif tag == "a":
                href = element.href
                if not href or not href.strip():
                    continue
                scheme = urlsplit(href.strip()).scheme.lower()
                if scheme not in ("http", "https", ""):
                    continue  # javascript:, mailto:, data:, and friends.
                resolved = _resolve_href(href, base_url)
                try:
                    target_origin = origin_of(resolved)
                except ValueError:
                    continue
                if target_origin not in allowed_origins:
                    continue
                kept.append(_KeptElement(raw=element, kind_class="link", resolved_href=resolved))
                continue
            else:
                role = _normalized(element.role)
                if role in ("button", "link", "tab", "menuitem"):
                    kind_class = "control"
                else:
                    continue  # Plain text and non-actionable nodes are dropped.
            kept.append(_KeptElement(raw=element, kind_class=kind_class, resolved_href=None))

        priority = {
            "editable": _PRIORITY_EDITABLE,
            "control": _PRIORITY_CONTROL,
            "link": _PRIORITY_LINK,
        }
        kept.sort(key=lambda item: (priority[item.kind_class], item.raw.index))
        return kept

    def _fingerprint(self, raw: RawDomElement, seen: set[str]) -> ElementFingerprint:
        tag = _normalized(raw.tag) or "unknown"
        role = _normalized(raw.role) or None
        text_length = len(_normalized(raw.text))
        input_type = _normalized(raw.input_type) or None
        autocomplete = _normalized(raw.autocomplete) or None
        attributes = raw.attributes or {}
        attr_parts = sorted(
            f"{key.lower()}={value}"
            for key, value in attributes.items()
            if key.lower() in _FINGERPRINT_ATTRIBUTES
        )
        canonical = "|".join(
            [
                tag,
                role or "",
                str(text_length),
                input_type or "",
                autocomplete or "",
                _normalized(raw.href),
                ";".join(attr_parts),
            ]
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        fingerprint_id = f"el-{digest[:16]}"
        if fingerprint_id in seen:
            fingerprint_id = f"el-{digest[:16]}-{len(seen)}"
        seen.add(fingerprint_id)
        return ElementFingerprint(
            fingerprint_id=fingerprint_id,
            tag=tag,
            role=role,
            text_length=text_length,
            attributes_hash=f"sha256:{digest[:32]}",
            is_sensitive=False,
        )

    def _candidate(
        self,
        item: _KeptElement,
        candidate_id: str,
        fingerprint: ElementFingerprint,
        password_present: bool,
    ) -> TrustedCandidate:
        raw = item.raw
        text = _bounded_text(raw.text, 120)
        download_flag = (raw.attributes or {}).get("download") is not None
        review_text = f"{text} {(raw.attributes or {}).get('aria-label') or ''}"
        review_input_type: str | None = raw.input_type

        if item.kind_class == "link":
            kind: BrowserActionKind = "navigate"
            needs_review = _match_review_kind(
                review_text, input_type=None, download_flag=download_flag
            )
            description = f"link: {text}" if text else "link"
        elif item.kind_class == "editable":
            kind = "request_text"
            needs_review = None
            label = _bounded_text(
                (raw.attributes or {}).get("aria-label")
                or (raw.attributes or {}).get("name")
                or text,
                80,
            )
            description = f"input {_normalized(raw.input_type) or 'text'}: {label}"
        else:
            kind = "click"
            needs_review = _match_review_kind(
                review_text, input_type=review_input_type, download_flag=download_flag
            )
            control_type = _normalized(review_input_type)
            is_submit_control = control_type == "submit" or (
                control_type == "" and _normalized(raw.tag) == "button"
            )
            if needs_review is None and is_submit_control:
                needs_review = "login_submit" if password_present else "message_submit"
            description = f"button: {text}" if text else "button"

        description = _bounded_text(description, 200)
        if not description:
            description = item.kind_class
        return TrustedCandidate(
            candidate_id=candidate_id,
            kind=kind,
            fingerprint=fingerprint,
            description=description,
            needs_review_kind=needs_review,
        )


def _synthetic_fingerprint(fingerprint_id: str, tag: str) -> ElementFingerprint:
    digest = hashlib.sha256(fingerprint_id.encode("utf-8")).hexdigest()
    return ElementFingerprint(
        fingerprint_id=fingerprint_id,
        tag=tag,
        role=None,
        text_length=0,
        attributes_hash=f"sha256:{digest[:32]}",
        is_sensitive=False,
    )


def _synthetic_candidates() -> list[TrustedCandidate]:
    """Bounded scroll and wait candidates. Goal completion is Noul; no finish."""
    specs: list[tuple[str, str, str]] = [
        ("scroll-down", SCROLL_DOWN_CANDIDATE_ID, "scroll down the page"),
        ("scroll-up", SCROLL_UP_CANDIDATE_ID, "scroll up the page"),
        ("wait", WAIT_CANDIDATE_ID, "wait briefly for the page to settle"),
    ]
    candidates: list[TrustedCandidate] = []
    for name, candidate_id, description in specs:
        candidates.append(
            TrustedCandidate(
                candidate_id=candidate_id,
                kind="scroll" if name.startswith("scroll") else "wait",
                fingerprint=_synthetic_fingerprint(f"syn-{name}", "synthetic"),
                description=description,
                needs_review_kind=None,
            )
        )
    return candidates


__all__ = [
    "SCROLL_DOWN_CANDIDATE_ID",
    "SCROLL_UP_CANDIDATE_ID",
    "WAIT_CANDIDATE_ID",
    "ObservationBuilder",
    "RawDomElement",
    "RuntimeObservation",
]
