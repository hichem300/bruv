"""Executor behavior tests with fake Playwright objects. No real browser."""

from __future__ import annotations

import sys
import threading
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import bruv.browser.executor as executor_module
from bruv.browser.actions import ClickAction, NavigateAction
from bruv.browser.executor import (
    BrowserDependencyError,
    BrowserError,
    BrowserExecutor,
    DisallowedNavigationError,
    ReviewRequiredError,
    StaleActionError,
)
from bruv.browser.models import ElementFingerprint, Observation, TrustedCandidate
from bruv.browser.observation import RuntimeObservation

BASE = "https://example.com/start"
LINK_URL = "https://example.com/page2"
TRANSCRIPT = "transcript.jsonl"

PAGE_ELEMENTS = {
    "a[href]": [{"tag": "a", "href": "/page2", "text": "Go", "visible": True}],
    "button": [{"tag": "button", "text": "Save", "visible": True}],
    "input": [{"tag": "input", "input_type": "text", "visible": True}],
}

# (candidate_id, kind, raw extraction index); raw order: a=0, button=1, input=2.
SPECS = [
    ("c-nav", "navigate", 0),
    ("c-btn", "click", 1),
    ("c-txt", "request_text", 2),
]


def fingerprint(cid: str, tag: str = "x") -> ElementFingerprint:
    return ElementFingerprint(
        fingerprint_id=f"fp-{cid}", tag=tag, text_length=0, attributes_hash=f"h-{cid}"
    )


def make_observation(
    specs: list[tuple[str, str, int]],
    navigation_targets: tuple[tuple[str, str], ...] = (("c-nav", LINK_URL),),
) -> RuntimeObservation:
    candidates = tuple(
        TrustedCandidate(candidate_id=cid, kind=kind, fingerprint=fingerprint(cid), description=cid)
        for cid, kind, _ in specs
    )
    candidate_elements = tuple((cid, raw_index) for cid, _, raw_index in specs)
    summary = Observation(
        observation_id="obs-1",
        url=BASE,
        title="t",
        page_text_ref="sha256:abc",
        element_count=len(specs),
        elements=(),
    )
    return RuntimeObservation(
        summary=summary,
        candidates=candidates,
        page_text="p",
        navigation_targets=navigation_targets,
        candidate_elements=candidate_elements,
    )


def with_changes(observation: RuntimeObservation, cid: str, **changes) -> RuntimeObservation:
    candidates = tuple(
        c if c.candidate_id != cid else c.model_copy(update=changes) for c in observation.candidates
    )
    return RuntimeObservation(
        summary=observation.summary,
        candidates=candidates,
        page_text=observation.page_text,
        navigation_targets=observation.navigation_targets,
        candidate_elements=observation.candidate_elements,
    )


class QueueBuilder:
    """Returns observations in order, then repeats the last one."""

    def __init__(self, observations: list[RuntimeObservation]) -> None:
        self._observations = observations

    def build(self, url, title, page_text, elements) -> RuntimeObservation:
        if len(self._observations) > 1:
            return self._observations.pop(0)
        return self._observations[0]


class FakeSlot:
    def __init__(self, data: dict, clicks: list[bool], fills: list[str]) -> None:
        self._data = data
        self._clicks = clicks
        self._fills = fills

    def evaluate(self, _script: str) -> dict:
        return dict(self._data)

    def click(self, timeout: int | None = None) -> None:
        self._clicks.append(True)

    def fill(self, text: str, timeout: int | None = None) -> None:
        self._fills.append(text)


class FakeLocator:
    def __init__(
        self, elements: list[dict], clicks: list[bool], fills: list[str], body_text: str = ""
    ) -> None:
        self._elements = elements
        self._clicks = clicks
        self._fills = fills
        self._body_text = body_text

    def count(self) -> int:
        return len(self._elements)

    def nth(self, index: int) -> FakeSlot:
        return FakeSlot(self._elements[index], self._clicks, self._fills)

    def inner_text(self, timeout: int | None = None) -> str:
        return self._body_text


class FakePage:
    def __init__(self, elements_by_selector: dict[str, list[dict]], url: str = BASE) -> None:
        self.url = url
        self.gotos: list[str] = []
        self.clicks: list[bool] = []
        self.fills: list[str] = []
        self.mouse = SimpleNamespace(wheel=lambda x, y: None)
        self._elements = elements_by_selector

    def title(self) -> str:
        return "Page"

    def locator(self, selector: str) -> FakeLocator:
        if selector == "body":
            return FakeLocator([], self.clicks, self.fills, body_text="page body")
        return FakeLocator(self._elements.get(selector, []), self.clicks, self.fills)

    def goto(self, url: str, wait_until: str | None = None, timeout: int | None = None) -> None:
        self.gotos.append(url)
        self.url = url

    def wait_for_timeout(self, ms: int) -> None:
        pass

    def screenshot(self, path=None, full_page: bool = False) -> None:
        pass


class FakeContext:
    tracing = SimpleNamespace(start=lambda **kw: None, stop=lambda **kw: None)

    def __init__(self, page: FakePage) -> None:
        self._page = page

    def new_page(self) -> FakePage:
        return self._page

    def close(self) -> None:
        pass


class FakeBrowser:
    def __init__(self, page: FakePage) -> None:
        self._page = page

    def new_context(self, record_video_dir=None) -> FakeContext:
        return FakeContext(self._page)

    def close(self) -> None:
        pass


class FakeSyncPlaywright:
    def __init__(self, page: FakePage) -> None:
        self._page = page

    def __call__(self) -> FakeSyncPlaywright:
        return self

    def start(self) -> FakeSyncPlaywright:
        return self

    def stop(self) -> None:
        pass

    @property
    def chromium(self):
        return SimpleNamespace(launch=lambda headless: FakeBrowser(self._page))


def make_started_executor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    builder: QueueBuilder,
    page: FakePage | None = None,
) -> tuple[BrowserExecutor, FakePage]:
    page = page or FakePage(PAGE_ELEMENTS)
    fake = FakeSyncPlaywright(page)
    monkeypatch.setattr(executor_module, "_require_sync_playwright", lambda: fake)
    executor = BrowserExecutor(
        session_id="s1",
        initial_url=BASE,
        builder=builder,
        artifact_dir=tmp_path / "art",
    )
    executor.start()
    return executor, page


def transcript_text(tmp_path) -> str:
    return (tmp_path / "art" / "s1" / TRANSCRIPT).read_text()


def test_dependency_error_is_actionable_without_playwright(monkeypatch, tmp_path) -> None:
    monkeypatch.setitem(sys.modules, "playwright", None)
    monkeypatch.setitem(sys.modules, "playwright.sync_api", None)
    executor = BrowserExecutor(
        session_id="s1",
        initial_url=BASE,
        builder=QueueBuilder([make_observation(SPECS)]),
        artifact_dir=tmp_path / "art",
    )
    with pytest.raises(BrowserDependencyError, match="pip install"):
        executor.start()


def test_cross_thread_call_fails_closed(monkeypatch, tmp_path) -> None:
    executor, _ = make_started_executor(
        monkeypatch, tmp_path, QueueBuilder([make_observation(SPECS)])
    )
    errors: list[Exception] = []

    def run() -> None:
        try:
            executor.observe()
        except BrowserError as exc:
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()
    assert errors and "owning thread" in str(errors[0])


def test_closed_executor_fails_closed(monkeypatch, tmp_path) -> None:
    executor, _ = make_started_executor(
        monkeypatch, tmp_path, QueueBuilder([make_observation(SPECS)])
    )
    executor.close()
    with pytest.raises(BrowserError, match="closed"):
        executor.observe()


def test_stale_fingerprint_fails_before_click(monkeypatch, tmp_path) -> None:
    obs1 = make_observation(SPECS)
    obs2 = with_changes(
        obs1,
        "c-btn",
        fingerprint=ElementFingerprint(
            fingerprint_id="fp-other", tag="x", text_length=0, attributes_hash="h-other"
        ),
    )
    executor, page = make_started_executor(monkeypatch, tmp_path, QueueBuilder([obs1, obs2]))
    with pytest.raises(StaleActionError):
        executor.execute(ClickAction(candidate_id="c-btn"))
    assert not page.clicks


def test_missing_candidate_fails_before_click(monkeypatch, tmp_path) -> None:
    obs1 = make_observation(SPECS)
    obs2 = make_observation([spec for spec in SPECS if spec[0] != "c-btn"])
    executor, page = make_started_executor(monkeypatch, tmp_path, QueueBuilder([obs1, obs2]))
    with pytest.raises(StaleActionError):
        executor.execute(ClickAction(candidate_id="c-btn"))
    assert not page.clicks


def test_review_gated_candidate_fails_before_click(monkeypatch, tmp_path) -> None:
    obs1 = make_observation(SPECS)
    obs2 = with_changes(obs1, "c-btn", needs_review_kind="payment")
    executor, page = make_started_executor(monkeypatch, tmp_path, QueueBuilder([obs1, obs2]))
    with pytest.raises(ReviewRequiredError):
        executor.execute(ClickAction(candidate_id="c-btn"))
    assert not page.clicks


def test_sensitive_direct_typing_rejected_without_leak(monkeypatch, tmp_path) -> None:
    page = FakePage(
        {**PAGE_ELEMENTS, "input": [{"tag": "input", "input_type": "password", "visible": True}]}
    )
    executor, _ = make_started_executor(
        monkeypatch, tmp_path, QueueBuilder([make_observation(SPECS)]), page=page
    )
    with pytest.raises(BrowserError, match="sensitive"):
        executor.type_text("c-txt", "SECRET-PW")
    assert "SECRET-PW" not in transcript_text(tmp_path)


def test_navigate_requires_candidate_id() -> None:
    with pytest.raises(ValidationError):
        NavigateAction(kind="navigate", url=LINK_URL)


def test_navigate_exact_fresh_target(monkeypatch, tmp_path) -> None:
    executor, page = make_started_executor(
        monkeypatch, tmp_path, QueueBuilder([make_observation(SPECS)])
    )
    page.gotos.clear()
    executor.execute(NavigateAction(candidate_id="c-nav", url=LINK_URL))
    assert page.gotos == [LINK_URL]
    assert page.url == LINK_URL


def test_navigate_query_mismatch_fails_before_goto(monkeypatch, tmp_path) -> None:
    executor, page = make_started_executor(
        monkeypatch, tmp_path, QueueBuilder([make_observation(SPECS)])
    )
    page.gotos.clear()
    with pytest.raises(DisallowedNavigationError):
        executor.execute(NavigateAction(candidate_id="c-nav", url=LINK_URL + "?utm_source=x"))
    assert not page.gotos
    assert "utm_source" not in transcript_text(tmp_path)


def test_navigate_disallowed_origin_fails_before_goto(monkeypatch, tmp_path) -> None:
    obs = make_observation(SPECS, navigation_targets=(("c-nav", "https://evil.com/x"),))
    executor, page = make_started_executor(monkeypatch, tmp_path, QueueBuilder([obs]))
    page.gotos.clear()
    with pytest.raises(DisallowedNavigationError):
        executor.execute(NavigateAction(candidate_id="c-nav", url="https://evil.com/x"))
    assert not page.gotos
