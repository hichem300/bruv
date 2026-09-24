"""Browser contract model tests: immutability and text-free persistence."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from bruv.browser.models import (
    BrowserGoal,
    ElementFingerprint,
    ExecutedAction,
    Observation,
    PendingRequest,
    SessionArtifacts,
    SessionState,
    TrustedCandidate,
    assert_no_supplied_text,
)

SUPPLIED_GOAL = "log in with hunter2 and buy the gift card"


def make_fingerprint(**overrides) -> ElementFingerprint:
    fields = {
        "fingerprint_id": "fp1",
        "tag": "button",
        "role": "button",
        "text_length": 12,
        "attributes_hash": "a" * 8,
    }
    fields.update(overrides)
    return ElementFingerprint(**fields)


def make_session_state(**overrides) -> SessionState:
    fields = {
        "session_id": "sess-1",
        "status": "running",
        "goal": BrowserGoal(
            goal_id="goal-1",
            goal_ref="mem:goal-1",
            initial_url="https://example.com/",
        ),
        "backend": "typesafe",
        "origin_allowlist": ("https://example.com",),
    }
    fields.update(overrides)
    return SessionState(**fields)


def test_statuses_are_exactly_the_six_defined() -> None:
    allowed = {"running", "needs_text", "needs_review", "done", "failed", "stopped"}
    for status in allowed:
        assert make_session_state(status=status).status == status
    for bad in ("paused", "pending", "complete", ""):
        with pytest.raises(ValidationError):
            make_session_state(status=bad)


def test_models_are_immutable() -> None:
    state = make_session_state()
    with pytest.raises(ValidationError):
        state.status = "done"
    fingerprint = make_fingerprint()
    with pytest.raises(ValidationError):
        fingerprint.text_length = 5
    action = ExecutedAction(step=1, kind="click", description="click button", outcome="ok")
    with pytest.raises(ValidationError):
        action.step = 2


def test_nested_collections_are_frozen_snapshots() -> None:
    state = make_session_state(
        executed_actions=(
            ExecutedAction(step=1, kind="click", description="click button", outcome="ok"),
        )
    )
    assert isinstance(state.executed_actions, tuple)
    with pytest.raises(ValidationError):
        state.executed_actions = ("nope",)  # type: ignore[assignment]


def test_session_state_persists_only_refs_not_supplied_text() -> None:
    state = make_session_state(
        pending_request=PendingRequest(request_id="r1", kind="text", prompt_ref="mem:p1")
    )
    payload = state.model_dump_json()
    assert SUPPLIED_GOAL not in payload
    assert state.goal.goal_ref == "mem:goal-1"


def test_executed_action_dump_has_no_text_payload() -> None:
    action = ExecutedAction(step=2, kind="type", description="type into search box", outcome="ok")
    assert SUPPLIED_GOAL not in action.model_dump_json()
    assert "text_ref" not in action.model_dump()


def test_assert_no_supplied_text_guardrail() -> None:
    state = make_session_state()
    assert_no_supplied_text(state, (SUPPLIED_GOAL,))
    with pytest.raises(ValueError, match="supplied text"):
        # A model that somehow carried supplied text must be rejected loudly.
        leaked = make_session_state(failure_reason=SUPPLIED_GOAL[:100])
        assert_no_supplied_text(leaked, (SUPPLIED_GOAL[:100],))


def test_observation_is_bounded() -> None:
    with pytest.raises(ValidationError):
        Observation(
            observation_id="obs-1",
            url="https://example.com/",
            title="Example",
            page_text_ref="mem:t1",
            element_count=3,
            elements=tuple(make_fingerprint(fingerprint_id=f"fp{i}") for i in range(501)),
        )


def test_candidate_description_is_bounded() -> None:
    with pytest.raises(ValidationError):
        TrustedCandidate(
            candidate_id="c1",
            kind="click",
            fingerprint=make_fingerprint(),
            description="x" * 201,
        )


def test_artifacts_hold_paths_only() -> None:
    artifacts = SessionArtifacts(screenshot_path="/data/shot.png", transcript_path="/data/t.txt")
    payload = artifacts.model_dump_json()
    assert SUPPLIED_GOAL not in payload
    assert artifacts.video_path is None


def test_failure_reason_is_bounded() -> None:
    with pytest.raises(ValidationError):
        make_session_state(failure_reason="x" * 201)
