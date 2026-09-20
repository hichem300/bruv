from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from bruv.domain.questions import ChoiceQuestion, NoulQuestion, ScoreQuestion
from bruv.domain.requests import DecisionRequest
from bruv.domain.validation import BackendCapabilities, ValidationResult, validate_request


def capabilities(
    backend: str = "simple-jev",
    *,
    question_types: frozenset[str] = frozenset({"noul", "choice", "score"}),
    allows_json_state: bool = True,
) -> BackendCapabilities:
    return BackendCapabilities(
        backend=backend,  # type: ignore[arg-type]
        question_types=question_types,
        calibrated=backend == "typesafe",
        allows_json_state=allows_json_state,
    )


def request_with(state: object, question: object | None = None) -> DecisionRequest:
    selected = question or NoulQuestion(instructions="Refund?")
    return DecisionRequest(
        state=state,  # type: ignore[arg-type]
        questions={"check": selected},  # type: ignore[dict-item]
    )


@pytest.mark.parametrize("state", [True, False, 0, 3, 1.5])
def test_simple_jev_rejects_top_level_boolean_or_number_state(state: object) -> None:
    result = validate_request(request_with(state), capabilities())

    assert result.valid is False
    assert result.issues == (result.issues[0],)
    assert result.issues[0].code == "simple_jev_scalar_state_not_supported"
    assert result.issues[0].path == ("state",)
    assert result.issues[0].message == ("Simple Jev state cannot be a top-level boolean or number.")


@pytest.mark.parametrize("state", [None, "ticket", [1, True], {"priority": 3}])
def test_simple_jev_accepts_supported_top_level_state(state: object) -> None:
    result = validate_request(request_with(state), capabilities())

    assert result == ValidationResult(valid=True, issues=())


def test_backend_without_json_state_accepts_text_only() -> None:
    result = validate_request(
        request_with({"ticket": "refund"}),
        capabilities(allows_json_state=False),
    )

    assert [issue.code for issue in result.issues] == ["json_state_not_supported"]


def test_typesafe_rejects_more_than_ten_score_levels_without_weakening_canonical_model() -> None:
    question = ScoreQuestion(
        instructions="Score it",
        criteria=[f"level-{index}" for index in range(11)],
    )
    request = request_with("ticket", question)

    result = validate_request(request, capabilities("typesafe"))

    assert result.valid is False
    assert result.issues[0].code == "typesafe_score_level_limit"
    assert result.issues[0].path == ("questions", "check", "criteria")
    assert result.issues[0].message == (
        "TypeSafe score criteria must contain between 2 and 10 levels."
    )


def test_typesafe_accepts_ten_score_levels() -> None:
    question = ScoreQuestion(
        instructions="Score it",
        criteria=[f"level-{index}" for index in range(10)],
    )

    assert validate_request(request_with("ticket", question), capabilities("typesafe")).valid


def test_unsupported_question_types_are_reported_in_sorted_question_order() -> None:
    request = DecisionRequest(
        state="ticket",
        questions={
            "z-question": ScoreQuestion(instructions="Score?", criteria=["low", "high"]),
            "a-question": NoulQuestion(instructions="Refund?"),
            "m-question": ChoiceQuestion(
                instructions="Route?",
                criteria={"billing": None, "other": None},
            ),
        },
    )

    result = validate_request(
        request,
        capabilities(question_types=frozenset({"choice"})),
    )

    assert [issue.path for issue in result.issues] == [
        ("questions", "a-question", "type"),
        ("questions", "z-question", "type"),
    ]
    assert all(issue.code == "question_type_not_supported" for issue in result.issues)


def test_simple_jev_rejects_bare_scalar_criteria_entries() -> None:
    request = request_with(
        "ticket",
        ChoiceQuestion(
            instructions="Route?",
            criteria={"billing": 1, "other": {"description": "another route"}},
        ),
    )

    result = validate_request(request, capabilities())

    assert [issue.code for issue in result.issues] == ["simple_jev_scalar_entry_not_supported"]
    assert result.issues[0].path == ("questions", "check", "criteria", "billing")


def test_capabilities_and_validation_results_are_immutable() -> None:
    backend_capabilities = capabilities()
    result = ValidationResult(valid=True, issues=())

    with pytest.raises(FrozenInstanceError):
        backend_capabilities.calibrated = True  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.valid = False  # type: ignore[misc]
