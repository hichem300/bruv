"""Field extraction tests."""

from __future__ import annotations

import pytest

from bruv.domain.results import ChoiceAnswer, DecisionResult, ScoreAnswer, Usage
from bruv.output.fields import FieldError, get_field


def _result() -> DecisionResult:
    return DecisionResult(
        backend="simple-jev",
        model="Qwen/Qwen3.5-0.8B",
        calibrated=False,
        answers={
            "route": ChoiceAnswer(
                type="choice",
                choice="billing",
                confidence=0.9,
                probabilities={"billing": 0.9, "other": 0.1},
            ),
            "urgency": ScoreAnswer(
                type="score",
                score=1.8,
                confidence=0.8,
                legend={"0": "low", "1": "medium", "2": "high"},
                probabilities={"0": 0.0, "1": 0.2, "2": 0.8},
            ),
        },
        usage=Usage(input_tokens=40, output_tokens=0),
    )


def test_top_level_field() -> None:
    assert get_field(_result(), "backend") == "simple-jev"
    assert get_field(_result(), "calibrated") is False


def test_choice_field() -> None:
    assert get_field(_result(), "answers.route.choice") == "billing"


def test_score_field() -> None:
    assert get_field(_result(), "answers.urgency.score") == 1.8


def test_usage_field() -> None:
    assert get_field(_result(), "usage.input_tokens") == 40


def test_unknown_top_level_rejected() -> None:
    with pytest.raises(FieldError):
        get_field(_result(), "secret")


def test_private_answer_field_rejected() -> None:
    with pytest.raises(FieldError):
        get_field(_result(), "answers.route.raw")


def test_missing_answer_rejected() -> None:
    with pytest.raises(FieldError):
        get_field(_result(), "answers.missing.choice")


def test_empty_path_rejected() -> None:
    with pytest.raises(FieldError):
        get_field(_result(), "")
