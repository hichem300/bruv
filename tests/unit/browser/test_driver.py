"""Unit tests for the browser driver over a tiny queued fake backend."""

from __future__ import annotations

import pytest

from bruv.browser.driver import BrowserBackendDriver, BrowserDriverError
from bruv.browser.models import ElementFingerprint, Observation, TrustedCandidate
from bruv.browser.observation import RuntimeObservation
from bruv.config import AppConfig
from bruv.domain.questions import ChoiceQuestion
from bruv.domain.requests import DecisionRequest
from bruv.domain.results import (
    AbstainAnswer,
    ChoiceAnswer,
    DecisionResult,
    NoulAnswer,
)
from bruv.domain.validation import BackendCapabilities

SECREET_REPLY = "SUPER_SECREET_REPLY_TEXT"


class FakeBackend:
    """Queued DecisionBackend fake. No network, no provider."""

    def __init__(
        self,
        capabilities: BackendCapabilities,
        results: list[DecisionResult],
    ) -> None:
        self.capabilities = capabilities
        self._results = list(results)
        self.requests: list[DecisionRequest] = []

    def evaluate(self, request: DecisionRequest) -> DecisionResult:
        self.requests.append(request)
        if not self._results:
            raise AssertionError("unexpected extra backend call")
        return self._results.pop(0)


def capabilities(
    *,
    backend: str = "typesafe",
    calibrated: bool = True,
    explicit_abstention: bool = False,
    supported_total_candidates: frozenset[int] | None = None,
) -> BackendCapabilities:
    return BackendCapabilities(
        backend=backend,
        question_types=frozenset({"noul", "choice"}),
        calibrated=calibrated,
        allows_json_state=True,
        explicit_abstention=explicit_abstention,
        supported_total_candidates=supported_total_candidates,
    )


def noul_result(
    answer: NoulAnswer | ChoiceAnswer | AbstainAnswer,
    *,
    backend: str = "typesafe",
    calibrated: bool = True,
    question_id: str = "goal_complete",
    extra: dict[str, object] | None = None,
) -> DecisionResult:
    answers: dict[str, object] = {question_id: answer}
    if extra:
        answers.update(extra)
    return DecisionResult(
        backend=backend,
        model="m",
        calibrated=calibrated,
        answers=answers,  # type: ignore[arg-type]
    )


def make_observation(
    candidates: tuple[TrustedCandidate, ...] = (),
    navigation_targets: tuple[tuple[str, str], ...] = (),
) -> RuntimeObservation:
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
        navigation_targets=navigation_targets,
    )


def candidate(cid: str, kind: str = "click") -> TrustedCandidate:
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
    )


def make_driver(
    backend: FakeBackend,
    *,
    config: AppConfig | None = None,
) -> BrowserBackendDriver:
    cfg = config or AppConfig(backend=backend.capabilities.backend)
    return BrowserBackendDriver(backend=backend, config=cfg)


def test_backend_name_mismatch_fails_closed_at_construction() -> None:
    backend = FakeBackend(capabilities(backend="needle"), [])
    with pytest.raises(BrowserDriverError):
        make_driver(backend, config=AppConfig(backend="typesafe"))


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        (NoulAnswer(value=True), True),
        (NoulAnswer(value=False), False),
        (NoulAnswer(noul=0.6), True),
        (NoulAnswer(noul=0.4), False),
    ],
)
def test_noul_bool_and_calibrated_threshold(answer: NoulAnswer, expected: bool) -> None:
    backend = FakeBackend(capabilities(), [noul_result(answer)])
    driver = make_driver(backend)
    assert (
        driver.goal_complete("goal", make_observation(), ()) is expected  # type: ignore[func-returns-value]
    )


def test_uncalibrated_probability_only_runs_exactly_one_choice_fallback() -> None:
    caps = capabilities(calibrated=False)
    results = [
        noul_result(NoulAnswer(noul=0.9), calibrated=False),
        noul_result(
            ChoiceAnswer(choice="complete"),
            calibrated=False,
            question_id="goal_complete_fallback",
        ),
    ]
    backend = FakeBackend(caps, results)
    driver = make_driver(backend)
    assert driver.goal_complete("goal", make_observation(), ()) is True
    assert len(backend.requests) == 2
    assert "goal_complete" not in backend.requests[1].questions
    assert isinstance(backend.requests[1].questions["goal_complete_fallback"], ChoiceQuestion)


@pytest.mark.parametrize(
    "result",
    [
        noul_result(AbstainAnswer(reason="insufficient_evidence", source_question_type="noul")),
        noul_result(ChoiceAnswer(choice="complete")),
        noul_result(NoulAnswer(value=True), question_id="wrong_question_id"),
    ],
    ids=["abstain", "wrong-type", "malformed-question-id"],
)
def test_malformed_wrong_and_abstain_answers_fail_closed(result: DecisionResult) -> None:
    backend = FakeBackend(capabilities(), [result])
    driver = make_driver(backend)
    with pytest.raises(BrowserDriverError):
        driver.goal_complete("goal", make_observation(), ())


def test_unknown_fallback_choice_fails_closed() -> None:
    caps = capabilities(calibrated=False)
    results = [
        noul_result(NoulAnswer(noul=0.9), calibrated=False),
        noul_result(
            ChoiceAnswer(choice="not-a-real-option"),
            calibrated=False,
            question_id="goal_complete_fallback",
        ),
    ]
    driver = make_driver(FakeBackend(caps, results))
    with pytest.raises(BrowserDriverError):
        driver.goal_complete("goal", make_observation(), ())


def test_batch_uses_opaque_ids_and_returns_chosen_candidate_id() -> None:
    batch = (candidate("c1"), candidate("c2"))
    backend = FakeBackend(
        capabilities(),
        [noul_result(ChoiceAnswer(choice="c2"), question_id="choose_candidate")],
    )
    driver = make_driver(backend)
    chosen = driver.choose_batch("goal", make_observation(batch), (), batch)
    assert chosen == "c2"
    question = backend.requests[0].questions["choose_candidate"]
    assert isinstance(question, ChoiceQuestion)
    assert set(question.criteria) == {"c1", "c2"}


def test_unsupported_batch_size_rejected_before_backend_io() -> None:
    caps = capabilities(supported_total_candidates=frozenset({3}))
    backend = FakeBackend(caps, [])
    driver = make_driver(backend)
    batch = (candidate("c1"), candidate("c2"))
    with pytest.raises(BrowserDriverError):
        driver.choose_batch("goal", make_observation(batch), (), batch)
    assert backend.requests == []
    with pytest.raises(BrowserDriverError):
        driver.select_candidate("goal", make_observation(batch), (), batch)
    assert backend.requests == []


def test_recursive_selection_returns_one_candidate() -> None:
    batch = (candidate("c1"), candidate("c2"), candidate("c3"))
    results = [
        noul_result(ChoiceAnswer(choice="c1"), question_id="choose_candidate"),
        noul_result(ChoiceAnswer(choice="c3"), question_id="choose_candidate"),
    ]
    # Only batch size 2 supported: forces 2-vs-1 then head-to-head recursion.
    driver = make_driver(
        FakeBackend(capabilities(supported_total_candidates=frozenset({2})), results)
    )
    winner = driver.select_candidate("goal", make_observation(batch), (), batch)
    assert isinstance(winner, TrustedCandidate)
    assert winner.candidate_id == "c3"
    assert len(driver._backend.requests) == 2  # noqa: SLF001


def test_request_state_is_bounded_and_safe() -> None:
    nav = (candidate("nav1", kind="navigate"),)
    observation = make_observation(
        nav, navigation_targets=(("nav1", "https://secret-target.example/path"),)
    )
    backend = FakeBackend(capabilities(), [noul_result(NoulAnswer(value=True))])
    driver = make_driver(backend)
    driver.goal_complete("goal", observation, ())
    state_json = str(backend.requests[0].state)
    assert "navigation_targets" not in state_json
    assert "secret-target.example" not in state_json
    assert SECREET_REPLY not in state_json


def test_result_backend_mismatch_rejected() -> None:
    results = [noul_result(NoulAnswer(value=True), backend="someone-else")]
    driver = make_driver(FakeBackend(capabilities(), results))
    with pytest.raises(BrowserDriverError):
        driver.goal_complete("goal", make_observation(), ())
