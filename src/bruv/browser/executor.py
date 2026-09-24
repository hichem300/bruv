"""Synchronous Playwright executor for trusted browser candidates.

Security model:

- Playwright is an optional dependency, lazily imported so ``bruv`` imports
  work without the ``browser`` extra.
- Only fixed, authored locator logic runs here. Model-generated JavaScript,
  caller-supplied selectors, arbitrary evaluation, shell, and file
  upload/download are never supported.
- Trusted candidate kinds (click, type, navigate, scroll, wait) execute only
  after immediate re-observation confirms the selected candidate ID and its
  ``ElementFingerprint`` still match the live page. Otherwise a dedicated
  stale-action error is raised before acting.
- Review-gated candidates never execute; they raise a dedicated
  review-required error.
- Navigation targets must appear in the runtime observation's
  ``navigation_targets`` and their top-level origin must be the run origin or
  a configured allowed origin. After any action that navigates or redirects,
  the resulting top-level origin is enforced again, failing closed.
- Supplied text (goal text, user replies) resolves in memory only, never
  appears in exceptions, descriptions, transcripts, reprs, or persisted
  models, and is discarded immediately after use.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Final

from bruv.browser.actions import (
    MAX_SCROLL_AMOUNT,
    MAX_WAIT_SECONDS,
    ClickAction,
    NavigateAction,
    RequestReviewAction,
    RequestTextAction,
    ScrollAction,
    TypeAction,
    WaitAction,
    origin_of,
)
from bruv.browser.artifacts import (
    ArtifactOptions,
    ArtifactStore,
    TranscriptWriter,
    safe_origin,
)
from bruv.browser.models import SessionArtifacts, TrustedCandidate
from bruv.browser.observation import ObservationBuilder, RawDomElement, RuntimeObservation

_ACTION_TIMEOUT_MS: Final[int] = int(MAX_WAIT_SECONDS * 1000)
_NAVIGATION_TIMEOUT_MS: Final[int] = 30_000

# Fixed, authored selector set for bounded interactive-element extraction.
_INTERACTIVE_SELECTORS: Final[tuple[str, ...]] = (
    "a[href]",
    "button",
    "input",
    "textarea",
    "select",
    "[role=button]",
    "[role=link]",
    "[role=tab]",
    "[role=menuitem]",
)
_SCAN_CAP_PER_SELECTOR: Final[int] = 400

# Fixed, authored per-element metadata extraction. Structural attributes only;
# input values are deliberately never read.
_EXTRACT_ELEMENT_SCRIPT: Final[str] = """el => {
    const attributes = {};
    for (const name of ['id', 'name', 'aria-label', 'download', 'target', 'formaction', 'list']) {
        const value = el.getAttribute(name);
        if (value !== null) attributes[name] = value;
    }
    return {
        tag: (el.tagName || 'unknown').toLowerCase(),
        role: el.getAttribute('role'),
        text: (String(el.innerText || '')).slice(0, 400),
        input_type: el.getAttribute('type'),
        autocomplete: el.getAttribute('autocomplete'),
        href: el.getAttribute('href'),
        visible: !!(el.offsetWidth || el.offsetHeight || el.getClientRects().length),
        disabled: el.disabled === true || el.getAttribute('aria-disabled') === 'true',
        attributes: attributes,
    };
}"""

# Fixed, authored selector for bounded page body text.
_BODY_SELECTOR: Final[str] = "body"

_SENSITIVE_INPUT_TYPES: Final[frozenset[str]] = frozenset({"password", "file"})
_SENSITIVE_AUTOCOMPLETE_EXACT: Final[frozenset[str]] = frozenset(
    {
        "one-time-code",
        "one-time-code-paragraph",
        "current-password",
        "new-password",
    }
)
_SENSITIVE_AUTOCOMPLETE_PREFIX: Final[str] = "cc-"


class BrowserError(Exception):
    """Base class for browser executor failures. Messages are text-safe."""


class BrowserDependencyError(BrowserError):
    """Playwright is not installed. Install the browser extra to continue."""


class StaleActionError(BrowserError):
    """The selected candidate is absent or changed on the live page."""


class ReviewRequiredError(BrowserError):
    """A review-gated candidate was offered for execution; it never executes."""


class DisallowedNavigationError(BrowserError):
    """A navigation target was not an observed trusted target or allowed origin."""


def _require_sync_playwright() -> Callable[[], object]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise BrowserDependencyError(
            "Playwright is required for browser automation. "
            "Install it with: pip install 'bruv[browser]'"
        ) from exc
    return sync_playwright


def _normalized(value: str | None) -> str:
    if value is None:
        return ""
    return " ".join(value.split()).strip().lower()


def _is_sensitive_field(raw: RawDomElement) -> bool:
    if _normalized(raw.input_type) in _SENSITIVE_INPUT_TYPES:
        return True
    autocomplete = _normalized(raw.autocomplete)
    if not autocomplete:
        return False
    return autocomplete in _SENSITIVE_AUTOCOMPLETE_EXACT or autocomplete.startswith(
        _SENSITIVE_AUTOCOMPLETE_PREFIX
    )


class BrowserExecutor:
    """Owns one isolated Chromium session and executes trusted candidates.

    Constructed with the session's ``ObservationBuilder`` (bounded limits and
    origin allowlist) and the run's initial URL. The run origin plus the
    configured allowed origins form the fixed navigation allowlist. All public
    calls must come from the thread that called ``start()``; the later daemon
    owns that thread exclusively. No browser profile persists: the context
    runs from a temporary directory removed on ``close()``.
    """

    def __init__(
        self,
        *,
        session_id: str,
        initial_url: str,
        builder: ObservationBuilder,
        allowed_origins: tuple[str, ...] = (),
        artifact_dir: Path | None = None,
        artifact_options: ArtifactOptions | None = None,
        headless: bool = True,
    ) -> None:
        self._initial_url = initial_url
        self._builder = builder
        self._headless = headless
        self._artifact_store = ArtifactStore(
            session_id, artifact_dir=artifact_dir, options=artifact_options
        )
        self._allowed_origins: frozenset[str] = frozenset(
            origin.strip().lower() for origin in allowed_origins if origin.strip()
        )
        self._owner_thread: int | None = None
        self._playwright: object | None = None
        self._browser: object | None = None
        self._context: object | None = None
        self._page: object | None = None
        self._closed = False
        self._step = 0
        self._observation: RuntimeObservation | None = None
        self._raw_elements: tuple[RawDomElement, ...] = ()
        self._element_locs: tuple[tuple[str, int], ...] = ()
        self._transcript: TranscriptWriter | None = None
        self._session_artifacts: SessionArtifacts | None = None

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        """Start Playwright and an isolated temporary browser context.

        ``new_context`` provides throwaway isolated state: no persistent
        profile exists and nothing about the session outlives ``close()``.
        """
        if self._playwright is not None:
            raise BrowserError("executor is already started")
        sync_playwright = _require_sync_playwright()
        self._owner_thread = threading.get_ident()
        self._session_artifacts = self._artifact_store.prepare()
        self._transcript = self._artifact_store.transcript()
        try:
            playwright = sync_playwright().start()
        except Exception:
            self._cleanup_playwright()
            raise
        self._playwright = playwright
        options = self._artifact_store.options
        video_dir = str(self._artifact_store.root) if options.video else None
        try:
            self._browser = playwright.chromium.launch(headless=self._headless)
            self._context = self._browser.new_context(record_video_dir=video_dir)
            if options.trace:
                self._context.tracing.start(screenshots=True, snapshots=True, sources=False)
            self._page = self._context.new_page()
            self._page.goto(
                self._initial_url,
                wait_until="domcontentloaded",
                timeout=_NAVIGATION_TIMEOUT_MS,
            )
            self._enforce_origin(self._page)
            self._refresh_observation()
        except Exception:
            self._save_final_screenshot()
            self.close()
            raise

    def close(self) -> None:
        """Close page, context, Playwright, and the temporary profile.

        Written artifacts are preserved. Safe to call more than once.
        """
        if self._closed:
            return
        self._closed = True
        self._save_final_screenshot()
        context = self._context
        browser = self._browser
        self._page = None
        self._context = None
        self._browser = None
        if context is not None:
            with suppress(Exception):
                if self._artifact_store.options.trace:
                    artifacts = self._session_artifacts
                    trace_path = artifacts.trace_path if artifacts else None
                    if trace_path:
                        context.tracing.stop(path=trace_path)
                context.close()
        if self._artifact_store.options.video:
            self._finalize_video()
        if browser is not None:
            with suppress(Exception):
                browser.close()
        self._cleanup_playwright()
        self._observation = None
        self._raw_elements = ()
        self._element_locs = ()

    def __enter__(self) -> BrowserExecutor:
        if self._playwright is None:
            self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    # -- observation -----------------------------------------------------------

    @property
    def session_artifacts(self) -> SessionArtifacts | None:
        """Declared artifact paths for this session, if ``start()`` prepared them.

        Safe metadata (paths only); None before start or when preparation failed.
        """
        return self._session_artifacts

    def observe(self) -> RuntimeObservation:
        """Re-observe the live page and return the bounded runtime observation."""
        self._ensure_usable()
        self._refresh_observation()
        return self._require_observation()

    # -- execution -----------------------------------------------------------

    def execute(
        self,
        action: ClickAction
        | TypeAction
        | NavigateAction
        | ScrollAction
        | WaitAction
        | RequestTextAction
        | RequestReviewAction,
        *,
        text_resolver: Callable[[str], str | None] | None = None,
    ) -> RuntimeObservation:
        """Execute one trusted action and return the fresh runtime observation.

        ``text_resolver`` resolves ``TypeAction.text_ref`` against in-memory
        text and is never stored. ``request_text`` and ``request_review``
        actions, and any review-gated candidate, are never auto-executed.
        """
        self._ensure_usable()
        if isinstance(action, (RequestTextAction, RequestReviewAction)):
            raise ReviewRequiredError(
                f"{action.kind} actions pause for a human and are never auto-executed"
            )
        self._step += 1
        try:
            if isinstance(action, ClickAction):
                self._execute_click(action)
            elif isinstance(action, TypeAction):
                self._execute_type(action, text_resolver)
            elif isinstance(action, NavigateAction):
                self._execute_navigate(action)
            elif isinstance(action, ScrollAction):
                self._execute_scroll(action)
            elif isinstance(action, WaitAction):
                self._execute_wait(action)
            else:  # pragma: no cover - exhaustive over the action union
                raise BrowserError(f"unsupported action kind: {type(action).__name__}")
        except BrowserError as exc:
            self._record(action, outcome="failed", error_kind=type(exc).__name__)
            raise
        self._record(action, outcome="ok")
        self._refresh_observation()
        return self._require_observation()

    def type_text(self, candidate_id: str, text: str) -> RuntimeObservation:
        """Type caller-supplied in-memory text into a trusted text candidate.

        The text exists only within this call frame and is discarded after
        use. Sensitive fields are rejected even when called directly.
        """
        self._ensure_usable()
        self._step += 1
        try:
            self._perform_type(candidate_id, text)
        except BrowserError as exc:
            self._record_kind("type", outcome="failed", error_kind=type(exc).__name__)
            raise
        self._record_kind("type", outcome="ok")
        self._refresh_observation()
        return self._require_observation()

    # -- action internals --------------------------------------------------------

    def _execute_click(self, action: ClickAction) -> None:
        page = self._require_page()
        self._validate_candidate(action.candidate_id, expected_kind="click")
        locator = self._locator_for_candidate(action.candidate_id)
        try:
            locator.click(timeout=_ACTION_TIMEOUT_MS)
        except Exception as exc:
            raise BrowserError("click on the candidate failed") from exc
        self._enforce_origin(page)

    def _execute_type(
        self,
        action: TypeAction,
        text_resolver: Callable[[str], str | None] | None,
    ) -> None:
        if text_resolver is None:
            raise BrowserError("typing requires a text resolver for the supplied text_ref")
        text = text_resolver(action.text_ref)
        if text is None:
            raise BrowserError("text reference could not be resolved")
        try:
            self._perform_type(action.candidate_id, text)
        finally:
            del text  # discard the local text reference immediately

    def _perform_type(self, candidate_id: str, text: str) -> None:
        self._validate_candidate(candidate_id, expected_kind="request_text")
        raw = self._raw_elements[self._raw_index(candidate_id)]
        if _is_sensitive_field(raw):
            raise BrowserError("typing into sensitive fields is not supported")
        locator = self._locator_for_candidate(candidate_id)
        try:
            locator.fill(text, timeout=_ACTION_TIMEOUT_MS)
        except Exception as exc:
            raise BrowserError("typing into the candidate failed") from exc

    def _execute_navigate(self, action: NavigateAction) -> None:
        page = self._require_page()
        # Re-observe immediately; the action must select a navigate candidate
        # that still exists, and its freshly observed trusted target must be
        # exactly the requested URL. No URL membership across candidates.
        self._validate_candidate(action.candidate_id, expected_kind="navigate")
        observation = self._require_observation()
        trusted_target = observation.target_for(action.candidate_id)
        if trusted_target is None or trusted_target != action.url:
            raise DisallowedNavigationError(
                "navigation target does not match the candidate's trusted target"
            )
        try:
            target_origin = origin_of(action.url)
        except ValueError as exc:
            raise DisallowedNavigationError(
                "navigation target is not an absolute http(s) URL"
            ) from exc
        if target_origin not in self._navigation_allowlist():
            raise DisallowedNavigationError("navigation target origin is not allowed")
        try:
            page.goto(action.url, wait_until="domcontentloaded", timeout=_NAVIGATION_TIMEOUT_MS)
        except Exception as exc:
            raise BrowserError("navigation to the trusted target failed") from exc
        self._enforce_origin(page)

    def _execute_scroll(self, action: ScrollAction) -> None:
        page = self._require_page()
        amount = min(action.amount, MAX_SCROLL_AMOUNT)
        delta = -amount if action.direction == "up" else amount
        page.mouse.wheel(0, delta)

    def _execute_wait(self, action: WaitAction) -> None:
        page = self._require_page()
        page.wait_for_timeout(int(min(action.seconds, MAX_WAIT_SECONDS) * 1000))

    # -- candidate validation --------------------------------------------------------

    def _navigation_allowlist(self) -> frozenset[str]:
        try:
            run_origin = origin_of(self._initial_url)
        except ValueError as exc:
            raise BrowserError("initial URL must be an absolute http(s) URL") from exc
        return frozenset({run_origin}) | self._allowed_origins

    def _validate_candidate(self, candidate_id: str, expected_kind: str) -> TrustedCandidate:
        """Re-observe and confirm candidate ID, fingerprint, and kind still match.

        Raises the dedicated stale-action error before any side effect, and the
        dedicated review-required error for review-gated candidates.
        """
        previous = self._require_observation()
        fresh = self.observe()
        previous_fingerprint = None
        for candidate in previous.candidates:
            if candidate.candidate_id == candidate_id:
                previous_fingerprint = candidate.fingerprint
                break
        current = None
        for candidate in fresh.candidates:
            if candidate.candidate_id == candidate_id:
                current = candidate
                break
        if current is None or previous_fingerprint is None:
            raise StaleActionError("selected candidate is no longer present on the page")
        if current.fingerprint != previous_fingerprint:
            raise StaleActionError("selected candidate changed on the page")
        if current.kind != expected_kind:
            raise StaleActionError("selected candidate no longer has the expected kind")
        if current.needs_review_kind is not None:
            raise ReviewRequiredError(
                "review-gated candidate requires human review and never auto-executes"
            )
        if fresh.raw_index_for(candidate_id) is None:
            raise StaleActionError("selected candidate has no live element")
        return current

    def _raw_index(self, candidate_id: str) -> int:
        observation = self._require_observation()
        raw_index = observation.raw_index_for(candidate_id)
        if raw_index is None or raw_index >= len(self._raw_elements):
            raise StaleActionError("selected candidate has no live element")
        return raw_index

    def _locator_for_candidate(self, candidate_id: str) -> object:
        page = self._require_page()
        raw_index = self._raw_index(candidate_id)
        selector, group_index = self._element_locs[raw_index]
        return page.locator(selector).nth(group_index)

    # -- observation extraction ---------------------------------------------------

    def _refresh_observation(self) -> None:
        page = self._require_page()
        raw_elements, element_locs = self._extract_elements(page)
        url = page.url
        try:
            title = page.title()
        except Exception:
            title = ""
        try:
            page_text = page.locator(_BODY_SELECTOR).inner_text(timeout=_ACTION_TIMEOUT_MS)
        except Exception:
            page_text = ""
        self._raw_elements = tuple(raw_elements)
        self._element_locs = tuple(element_locs)
        self._observation = self._builder.build(url, title, page_text, raw_elements)

    def _extract_elements(self, page: object) -> tuple[list[RawDomElement], list[tuple[str, int]]]:
        """Bounded extraction of interactive elements via fixed authored queries.

        Returns raw elements (with executor-assigned extraction indices) plus a
        parallel (selector, group index) list so a validated candidate can be
        re-located with Playwright locator APIs only.
        """
        elements: list[RawDomElement] = []
        locations: list[tuple[str, int]] = []
        seen_keys: set[tuple[object, ...]] = set()
        for selector in _INTERACTIVE_SELECTORS:
            locator = page.locator(selector)
            count = 0
            with suppress(Exception):
                count = min(locator.count(), _SCAN_CAP_PER_SELECTOR)
            for group_index in range(count):
                data = self._extract_element_data(locator, group_index)
                if data is None:
                    continue
                key = (
                    data.get("tag"),
                    data.get("role"),
                    data.get("text"),
                    data.get("input_type"),
                    data.get("autocomplete"),
                    data.get("href"),
                    tuple(sorted((data.get("attributes") or {}).items())),
                )
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                elements.append(
                    RawDomElement(
                        index=len(elements),
                        tag=str(data.get("tag") or "unknown"),
                        role=data.get("role"),
                        text=str(data.get("text") or ""),
                        input_type=data.get("input_type"),
                        autocomplete=data.get("autocomplete"),
                        href=data.get("href"),
                        visible=bool(data.get("visible", True)),
                        disabled=bool(data.get("disabled", False)),
                        attributes=data.get("attributes") or {},
                    )
                )
                locations.append((selector, group_index))
        return elements, locations

    def _extract_element_data(self, locator: object, group_index: int) -> dict | None:
        """Run the fixed extraction script on one element; None on failure."""
        with suppress(Exception):
            data = locator.nth(group_index).evaluate(_EXTRACT_ELEMENT_SCRIPT)
            if isinstance(data, dict):
                return data
        return None

    # -- guards and bookkeeping ------------------------------------------------------

    def _ensure_usable(self) -> None:
        if self._closed or self._playwright is None or self._page is None:
            raise BrowserError("executor has not been started or is closed")
        if self._owner_thread != threading.get_ident():
            raise BrowserError("browser executor calls must come from the owning thread")

    def _enforce_origin(self, page: object) -> None:
        """Fail closed if the resulting top-level origin is not allowed."""
        current_url = page.url
        try:
            current_origin = origin_of(current_url)
        except ValueError as exc:
            raise BrowserError("page left the allowed origins after the action") from exc
        if current_origin not in self._navigation_allowlist():
            raise BrowserError("page navigated to a disallowed origin; failing closed")

    def _require_observation(self) -> RuntimeObservation:
        if self._observation is None:
            raise BrowserError("no observation available; call start() first")
        return self._observation

    def _require_page(self) -> object:
        """Return the live page, failing closed when unusable."""
        self._ensure_usable()
        page = self._page
        if page is None:
            raise BrowserError("executor has not been started or is closed")
        return page

    def _record(
        self,
        action: ClickAction
        | TypeAction
        | NavigateAction
        | ScrollAction
        | WaitAction
        | RequestTextAction
        | RequestReviewAction,
        *,
        outcome: str,
        error_kind: str | None = None,
    ) -> None:
        self._record_kind(
            action.kind,
            outcome=outcome,
            candidate_id=getattr(action, "candidate_id", None),
            url=action.url if isinstance(action, NavigateAction) else None,
            error_kind=error_kind,
        )

    def _record_kind(
        self,
        kind: str,
        *,
        outcome: str,
        candidate_id: str | None = None,
        url: str | None = None,
        error_kind: str | None = None,
    ) -> None:
        if self._transcript is None:
            return
        # Only safe metadata is recorded: no page text, supplied text, or
        # full URLs (navigate records the top-level origin only).
        self._transcript.record(
            step=self._step,
            kind=kind,
            outcome=outcome,
            candidate_id=candidate_id,
            origin=safe_origin(url) if url else None,
            error_kind=error_kind,
        )

    def _finalize_video(self) -> None:
        """Best-effort rename of the recorded video to the declared path."""
        artifacts = self._session_artifacts
        if artifacts is None or not artifacts.video_path:
            return
        root = self._artifact_store.root
        recordings: list = []
        with suppress(Exception):
            recordings = sorted(root.glob("*.webm"))
        if not recordings:
            return
        with suppress(Exception):
            recordings[0].replace(artifacts.video_path)
            return  # best effort; the recording stays in the session folder

    def _save_final_screenshot(self) -> None:
        page = self._page
        artifacts = self._session_artifacts
        if page is None or artifacts is None or not artifacts.screenshot_path:
            return
        with suppress(Exception):
            page.screenshot(path=artifacts.screenshot_path, full_page=False)

    def _cleanup_playwright(self) -> None:
        playwright = self._playwright
        self._playwright = None
        if playwright is not None:
            with suppress(Exception):
                playwright.stop()


__all__ = [
    "BrowserDependencyError",
    "BrowserError",
    "BrowserExecutor",
    "DisallowedNavigationError",
    "ReviewRequiredError",
    "StaleActionError",
]
