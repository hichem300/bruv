from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from bruv.domain.questions import ChoiceQuestion, NoulQuestion, ScoreQuestion
from bruv.domain.requests import DecisionRequest


@pytest.mark.parametrize(
    ("question_type", "criteria"),
    [
        ("noul", None),
        ("choice", {"yes": None, "no": None}),
        ("score", ["low", "high"]),
    ],
)
def test_question_instructions_must_be_nonblank(
    question_type: str,
    criteria: object,
) -> None:
    model = {
        "noul": NoulQuestion,
        "choice": ChoiceQuestion,
        "score": ScoreQuestion,
    }[question_type]

    with pytest.raises(ValidationError):
        model(instructions=" \t\n", criteria=criteria)


@pytest.mark.parametrize("size", [1, 51])
def test_choice_requires_between_two_and_fifty_options(size: int) -> None:
    criteria = {f"option-{index}": None for index in range(size)}

    with pytest.raises(ValidationError):
        ChoiceQuestion(instructions="Pick one", criteria=criteria)


@pytest.mark.parametrize("size", [2, 50])
def test_choice_accepts_canonical_bounds(size: int) -> None:
    criteria = {f"option-{index}": None for index in range(size)}

    question = ChoiceQuestion(instructions="Pick one", criteria=criteria)

    assert len(question.criteria) == size


def test_choice_option_keys_must_be_nonblank() -> None:
    with pytest.raises(ValidationError):
        ChoiceQuestion(instructions="Pick one", criteria={" ": None, "valid": None})


@pytest.mark.parametrize("size", [1, 51])
def test_score_requires_between_two_and_fifty_levels(size: int) -> None:
    with pytest.raises(ValidationError):
        ScoreQuestion(instructions="Rate it", criteria=list(range(size)))


@pytest.mark.parametrize("size", [2, 50])
def test_score_accepts_ordered_canonical_bounds(size: int) -> None:
    levels = [f"level-{index}" for index in range(size)]

    question = ScoreQuestion(instructions="Rate it", criteria=levels)

    assert question.criteria == levels


def test_noul_criteria_accepts_only_true_and_false_keys() -> None:
    question = NoulQuestion(
        instructions="Is it urgent?",
        criteria={"true": {"meaning": "urgent"}, "false": None},
    )
    assert set(question.criteria or {}) == {"true", "false"}

    with pytest.raises(ValidationError):
        NoulQuestion(instructions="Is it urgent?", criteria={"yes": None})  # type: ignore[dict-item]


def test_request_parses_discriminated_questions() -> None:
    request = DecisionRequest.model_validate(
        {
            "state": {"ticket": "refund"},
            "questions": {
                "binary": {"type": "noul", "instructions": "Refund?"},
                "route": {
                    "type": "choice",
                    "instructions": "Route?",
                    "criteria": {"billing": None, "other": None},
                },
                "priority": {
                    "type": "score",
                    "instructions": "Priority?",
                    "criteria": ["low", "high"],
                },
            },
        }
    )

    assert isinstance(request.questions["binary"], NoulQuestion)
    assert isinstance(request.questions["route"], ChoiceQuestion)
    assert isinstance(request.questions["priority"], ScoreQuestion)


def test_unknown_question_discriminator_is_rejected() -> None:
    with pytest.raises(ValidationError):
        DecisionRequest.model_validate(
            {
                "state": "ticket",
                "questions": {
                    "route": {"type": "unknown", "instructions": "Route?"},
                },
            }
        )


@pytest.mark.parametrize("question_id", ["", " \n"])
def test_question_ids_must_be_nonblank(question_id: str) -> None:
    with pytest.raises(ValidationError):
        DecisionRequest(
            state="ticket",
            questions={question_id: NoulQuestion(instructions="Refund?")},
        )


@pytest.mark.parametrize("size", [0, 257])
def test_request_requires_between_one_and_256_questions(size: int) -> None:
    questions = {f"question-{index}": NoulQuestion(instructions="Yes?") for index in range(size)}

    with pytest.raises(ValidationError):
        DecisionRequest(state="ticket", questions=questions)


@pytest.mark.parametrize("size", [1, 256])
def test_request_accepts_canonical_question_bounds(size: int) -> None:
    questions = {f"question-{index}": NoulQuestion(instructions="Yes?") for index in range(size)}

    request = DecisionRequest(state="ticket", questions=questions)

    assert len(request.questions) == size


def test_recursive_json_state_is_accepted() -> None:
    state: dict[str, Any] = {
        "ticket": {
            "messages": ["hello", 3, 2.5, True, None, {"tags": ["billing"]}],
        }
    }

    request = DecisionRequest(
        state=state,
        questions={"route": NoulQuestion(instructions="Refund?")},
    )

    assert request.state == state


@pytest.mark.parametrize(
    "invalid_state",
    [
        {"value": float("nan")},
        [float("inf")],
        {"value": float("-inf")},
        {"not": {"a", "json", "value"}},
        ("not", "a", "json", "array"),
        {1: "non-string key"},
    ],
)
def test_non_json_or_nonfinite_state_is_rejected(invalid_state: object) -> None:
    with pytest.raises(ValidationError):
        DecisionRequest(
            state=invalid_state,  # type: ignore[arg-type]
            questions={"route": NoulQuestion(instructions="Refund?")},
        )


def test_nonfinite_nested_question_criteria_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ChoiceQuestion(
            instructions="Route?",
            criteria={"billing": {"weight": float("nan")}, "other": None},
        )
