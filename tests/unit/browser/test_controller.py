"""Unit tests for the browser controller with fake driver and executor."""

from __future__ import annotations

import threading

from bruv.browser.controller import BrowserController
from bruv.browser.models import (
    ElementFingerprint,
    ExecutedAction,
    Observation,
    SessionArtifacts,
    TrustedCandidate,
)
from bruv.browser.observation import RuntimeObservation


class FakeDriver:
    """Programmed driver fake: queued goal results and candidate selections."""

    backend_name = "fake"

    def __init__(
        self,
        goal_results: list[object],
        selections: list[object] | None = None,
    ) -> None:
        self.goal_results = list(goal_results)
        self.selections = list(selections or [])

    def goal_complete(self, goal_text: str, observation: RuntimeObservation, recent: tuple) -> bool:
        item = self.goal_results.pop(0)
        if isinstance(item, Exception):
            raise item
        assert isinstance(item, bool)
        return item

    def select_candidate(
        self, goal_text: str, observation: RuntimeObservation, recent: tuple, candidates: tuple
    ):
        item = self.selections.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeExecutor:
    def __init__(self, observation: RuntimeObservation, artifacts=None) -> None:
        self._observation = observation
        self._artifacts = artifacts
        self.started = False
        self.closed = False
        self.executed: list[object] = []
        self.typed: list[tuple[str, str]] = []

    def start(self) -> None:
        self.started = True

    def close(self) -> None:
        self.closed = True

    @property
    def session_artifacts(self):
        return self._artifacts

    def observe(self) -> RuntimeObservation:
        return self._observation

    def execute(self, action) -> None:
        self.executed.append(action)

    def type_text(self, candidate_id: str, text: str) -> RuntimeObservation:
        self.typed.append((candidate_id, text))
        return self._observation


def make_observation(candidates: tuple = ()) -> RuntimeObservation:
    summary = Observation(
        observation_id="obs-1",
        url="https://example.com/",
        title="Example",
        page_text_ref="sha256:abc",
        element_count=len(candidates),
        elements=(),
    )
    return RuntimeObservation(
        summary=summary,
        candidates=candidates,
        page_text="page text",
        navigation_targets=(),
    )


def candidate(cid: str, kind: str = "click", needs_review_kind=None):
    return TrustedCandidate(
        candidate_id=cid,
        kind=kind,  # type: ignore[arg-type]
        fingerprint=ElementFingerprint(
            fingerprint_id=cid,
            tag="button",
            text_length=1,
            attributes_hash="h",
        ),
        description=f"desc {cid}",
        needs_review_kind=needs_review_kind,
    )


def make_controller(driver, executor, **kwargs):
    defaults: dict = {
        "session_id": "s-1",
        "initial_url": "https://example.com/",
        "goal_text": "do the thing",
        "goal_ref": "goalref-1",
        "max_steps": 25,
    }
    defaults.update(kwargs)
    return BrowserController(driver=driver, executor=executor, **defaults)


def test_goal_already_complete_done_no_action_artifacts_close() -> None:
    executor = FakeExecutor(
        make_observation(), artifacts=SessionArtifacts(screenshot_path="artifacts/shot.png")
    )
    driver = FakeDriver(goal_results=[True])
    controller = make_controller(driver, executor)
    state = controller.run()
    assert state.status == "done"
    assert state.executed_actions == ()
    assert state.artifacts is not None and state.artifacts.screenshot_path == "artifacts/shot.png"
    assert executor.closed and executor.started
    assert "do the thing" not in state.model_dump_json()


def test_false_then_click_then_true_exactly_one_safe_action() -> None:
    executor = FakeExecutor(make_observation())
    driver = FakeDriver(goal_results=[False, True], selections=[candidate("c1")])
    controller = make_controller(driver, executor)
    state = controller.run()
    assert state.status == "done"
    assert len(state.executed_actions) == 1
    action = state.executed_actions[0]
    assert isinstance(action, ExecutedAction)
    assert action.kind == "click" and action.outcome == "ok"
    assert action.description == "click candidate c1"
    assert len(executor.executed) == 1


def test_review_gated_candidate_needs_review_executes_nothing() -> None:
    executor = FakeExecutor(make_observation())
    driver = FakeDriver(
        goal_results=[False], selections=[candidate("c9", needs_review_kind="payment")]
    )
    controller = make_controller(driver, executor)
    state = controller.run()
    assert state.status == "needs_review"
    assert executor.executed == []
    assert state.pending_request is not None
    assert state.pending_request.kind == "review"
    assert state.pending_request.review_kind == "payment"
    assert executor.closed


def test_request_text_flow_worker_thread_reply_resumes_and_text_never_persisted() -> None:
    executor = FakeExecutor(make_observation())
    driver = FakeDriver(
        goal_results=[False, True], selections=[candidate("t1", kind="request_text")]
    )
    states: list = []
    needs_text = threading.Event()

    def on_state(state) -> None:
        states.append(state)
        if state.status == "needs_text":
            needs_text.set()

    controller2 = BrowserController(
        session_id="s-2",
        initial_url="https://example.com/",
        goal_text="fill the form",
        goal_ref="goalref-2",
        driver=driver,
        executor=executor,
        max_steps=25,
        on_state=on_state,
    )
    worker = threading.Thread(target=controller2.run)
    worker.start()
    assert needs_text.wait(5.0), "controller never reached needs_text"
    request_id = controller2.state.pending_request.request_id
    controller2.reply(request_id, "typed reply SECRET")
    worker.join(timeout=5.0)
    assert not worker.is_alive()
    final = controller2.state
    assert final.status == "done"
    assert executor.typed == [("t1", "typed reply SECRET")]
    assert "typed reply SECRET" not in final.model_dump_json()


def test_stop_while_waiting_for_reply_wakes_worker_to_stopped() -> None:
    executor = FakeExecutor(make_observation())
    driver = FakeDriver(goal_results=[False], selections=[candidate("t1", kind="request_text")])
    needs_text = threading.Event()
    holder: dict = {}

    def on_state(state) -> None:
        if state.status == "needs_text":
            needs_text.set()

    controller = BrowserController(
        session_id="s-3",
        initial_url="https://example.com/",
        goal_text="fill the form",
        goal_ref="goalref-3",
        driver=driver,
        executor=executor,
        max_steps=25,
        on_state=on_state,
    )
    worker = threading.Thread(target=lambda: holder.update(state=controller.run()))
    worker.start()
    assert needs_text.wait(5.0)
    controller.stop()
    worker.join(timeout=5.0)
    assert not worker.is_alive()
    assert controller.state.status == "stopped"


def test_max_steps_limit_breach_fails_with_category_reason() -> None:
    executor = FakeExecutor(make_observation())
    driver = FakeDriver(goal_results=[False, False], selections=[candidate("c1")])
    controller = make_controller(driver, executor, max_steps=1)
    state = controller.run()
    assert state.status == "failed"
    assert state.failure_reason == "max_steps_exceeded"
