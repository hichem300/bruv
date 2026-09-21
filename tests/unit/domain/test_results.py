from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from bruv.domain.results import ChoiceAnswer, DecisionResult, NoulAnswer, ScoreAnswer


def test_result_parses_discriminated_answers_and_defaults_metadata() -> None:
    result = DecisionResult.model_validate(
        {
            "backend": "typesafe",
            "model": "jev-1.13.0",
            "calibrated": True,
            "answers": {
                "refund": {"type": "noul", "noul": 0.75},
                "route": {
                    "type": "choice",
                    "choice": "billing",
                    "confidence": 0.9,
                    "probabilities": {"billing": 0.9, "other": 0.1},
                },
                "urgency": {
                    "type": "score",
                    "score": 1.8,
                    "confidence": 0.8,
                    "legend": {"0": "low", "1": "medium", "2": "high"},
                    "probabilities": {"0": 0.0, "1": 0.2, "2": 0.8},
                },
            },
        }
    )

    assert isinstance(result.answers["refund"], NoulAnswer)
    assert isinstance(result.answers["route"], ChoiceAnswer)
    assert isinstance(result.answers["urgency"], ScoreAnswer)
    assert result.usage is None
    assert result.latency_ms is None
    assert result.request_id is None
    assert result.provider_metadata is None


def test_result_accepts_needle_confidence_only_answers() -> None:
    result = DecisionResult(
        backend="needle",
        model="Cactus-Compute/needle3",
        calibrated=True,
        answers={
            "refund": NoulAnswer(value=True, confidence=0.9),
            "route": ChoiceAnswer(choice="billing", confidence=0.8),
            "urgency": ScoreAnswer(
                score=2.0,
                confidence=0.7,
                legend={"0": "low", "1": "medium", "2": "high"},
            ),
        },
    )

    assert result.backend == "needle"
    expected_answers = {
        "refund": {"type": "noul", "value": True, "confidence": 0.9},
        "route": {"type": "choice", "choice": "billing", "confidence": 0.8},
        "urgency": {
            "type": "score",
            "score": 2.0,
            "confidence": 0.7,
            "legend": {"0": "low", "1": "medium", "2": "high"},
        },
    }
    assert result.model_dump()["answers"] == expected_answers
    assert json.loads(result.model_dump_json())["answers"] == expected_answers
    assert result.answers["route"].probabilities is None  # type: ignore[union-attr]
    assert result.answers["urgency"].probabilities is None  # type: ignore[union-attr]


def test_probability_noul_serializes_with_existing_shape() -> None:
    answer = NoulAnswer(noul=0.75)
    expected = {"type": "noul", "noul": 0.75}

    assert answer.model_dump() == expected
    assert json.loads(answer.model_dump_json()) == expected


def test_result_accepts_success_envelope_metadata() -> None:
    result = DecisionResult.model_validate(
        {
            "backend": "simple-jev",
            "model": "Qwen/Qwen3.5-0.8B",
            "calibrated": False,
            "answers": {},
            "usage": {"input_tokens": 40, "output_tokens": 0},
            "latency_ms": 12.5,
            "request_id": "request-1",
            "provider_metadata": {"calibration": "not_calibrated", "nested": [True, 1]},
        }
    )

    assert result.usage is not None
    assert result.usage.input_tokens == 40
    assert result.latency_ms == 12.5


@pytest.mark.parametrize("value", [-0.01, 1.01, float("nan"), float("inf")])
def test_noul_probability_must_be_finite_and_bounded(value: float) -> None:
    with pytest.raises(ValidationError):
        NoulAnswer(noul=value)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"confidence": 0.9},
        {"noul": 0.9, "value": True},
        {"noul": 0.9, "confidence": 0.9},
        {"noul": 0.9, "value": True, "confidence": 0.9},
    ],
)
def test_noul_rejects_empty_partial_and_mixed_modes(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match="noul, value with confidence, or value only"):
        NoulAnswer(**payload)  # type: ignore[arg-type]


def test_noul_value_only_mode_round_trips() -> None:
    answer = NoulAnswer(value=False)
    expected = {"type": "noul", "value": False}

    assert answer.model_dump() == expected
    assert json.loads(answer.model_dump_json()) == expected
    assert NoulAnswer.model_validate(expected) == answer


def test_choice_and_score_support_label_only_mode() -> None:
    choice = ChoiceAnswer(choice="billing")
    score = ScoreAnswer(score=1.5, legend={"0": "low", "1": "high"})

    assert choice.confidence is None
    assert score.confidence is None
    assert choice.model_dump() == {"type": "choice", "choice": "billing"}
    assert score.model_dump() == {
        "type": "score",
        "score": 1.5,
        "legend": {"0": "low", "1": "high"},
    }
    assert ChoiceAnswer.model_validate({"type": "choice", "choice": "billing"}) == choice
    assert (
        ScoreAnswer.model_validate(
            {"type": "score", "score": 1.5, "legend": {"0": "low", "1": "high"}}
        )
        == score
    )


def test_confidence_only_mode_round_trips_without_probabilities() -> None:
    choice = ChoiceAnswer(choice="billing", confidence=0.8)
    score = ScoreAnswer(
        score=2.0,
        confidence=0.7,
        legend={"0": "low", "1": "medium", "2": "high"},
    )

    assert choice.probabilities is None
    assert score.probabilities is None
    assert choice.model_dump() == {"type": "choice", "choice": "billing", "confidence": 0.8}
    assert score.model_dump() == {
        "type": "score",
        "score": 2.0,
        "confidence": 0.7,
        "legend": {"0": "low", "1": "medium", "2": "high"},
    }


def test_probabilities_without_confidence_are_rejected() -> None:
    with pytest.raises(ValidationError, match="confidence when probabilities are present"):
        ChoiceAnswer(choice="billing", probabilities={"billing": 1.0})
    with pytest.raises(ValidationError, match="confidence when probabilities are present"):
        ScoreAnswer(
            score=0.5,
            legend={"0": "low", "1": "high"},
            probabilities={"0": 0.5, "1": 0.5},
        )


@pytest.mark.parametrize("value", [-0.01, 1.01, float("nan"), float("-inf")])
def test_choice_probabilities_must_be_finite_and_bounded(value: float) -> None:
    with pytest.raises(ValidationError):
        ChoiceAnswer(
            choice="billing",
            confidence=0.9,
            probabilities={"billing": value},
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_score_must_be_finite(value: float) -> None:
    with pytest.raises(ValidationError):
        ScoreAnswer(
            score=value,
            confidence=0.8,
            legend={"0": "low", "1": "high"},
            probabilities={"0": 0.2, "1": 0.8},
        )


def test_confidence_must_be_probability() -> None:
    with pytest.raises(ValidationError):
        ScoreAnswer(
            score=0.5,
            confidence=float("nan"),
            legend={"0": "low", "1": "high"},
            probabilities={"0": 0.5, "1": 0.5},
        )


def test_choice_must_have_corresponding_probability() -> None:
    with pytest.raises(ValidationError):
        ChoiceAnswer(choice="billing", confidence=0.9, probabilities={"other": 1.0})


@pytest.mark.parametrize("answer_type", ["choice", "score"])
def test_probability_distributions_reject_materially_invalid_total(answer_type: str) -> None:
    if answer_type == "choice":
        with pytest.raises(ValidationError, match="must sum to 1 within 1e-05"):
            ChoiceAnswer(
                choice="billing",
                confidence=0.8,
                probabilities={"billing": 0.8, "other": 0.4},
            )
    else:
        with pytest.raises(ValidationError, match="must sum to 1 within 1e-05"):
            ScoreAnswer(
                score=0.5,
                confidence=0.8,
                legend={"0": "low", "1": "high"},
                probabilities={"0": 0.8, "1": 0.4},
            )


@pytest.mark.parametrize("answer_type", ["choice", "score"])
def test_probability_distributions_accept_six_decimal_rounding(answer_type: str) -> None:
    probabilities = {"0": 0.333333, "1": 0.333333, "2": 0.333333}

    if answer_type == "choice":
        answer = ChoiceAnswer(
            choice="0",
            confidence=0.5,
            probabilities=probabilities,
        )
    else:
        answer = ScoreAnswer(
            score=1.0,
            confidence=0.5,
            legend={"0": "low", "1": "medium", "2": "high"},
            probabilities=probabilities,
        )

    assert answer.probabilities == probabilities


def test_empty_probability_distributions_are_rejected() -> None:
    with pytest.raises(ValidationError):
        ChoiceAnswer(choice="billing", confidence=0.5, probabilities={})
    with pytest.raises(ValidationError):
        ScoreAnswer(score=0.5, confidence=0.5, legend={}, probabilities={})


def test_score_legend_and_probabilities_must_match() -> None:
    with pytest.raises(ValidationError):
        ScoreAnswer(
            score=0.5,
            confidence=0.5,
            legend={"0": "low", "1": "high"},
            probabilities={"0": 1.0},
        )


def test_nonfinite_provider_metadata_is_rejected() -> None:
    with pytest.raises(ValidationError):
        DecisionResult(
            backend="typesafe",
            model="jev-latest",
            calibrated=True,
            answers={},
            provider_metadata={"metric": [float("nan")]},
        )


@pytest.mark.parametrize("latency", [-1.0, float("nan"), float("inf")])
def test_latency_must_be_nonnegative_and_finite(latency: float) -> None:
    with pytest.raises(ValidationError):
        DecisionResult(
            backend="typesafe",
            model="jev-latest",
            calibrated=True,
            answers={},
            latency_ms=latency,
        )


def test_result_values_are_deeply_immutable() -> None:
    result = DecisionResult(
        backend="typesafe",
        model="jev-latest",
        calibrated=True,
        answers={
            "route": ChoiceAnswer(
                choice="billing",
                confidence=0.9,
                probabilities={"billing": 0.9, "other": 0.1},
            ),
            "score": ScoreAnswer(
                score=0.8,
                confidence=0.8,
                legend={"0": {"tags": ["low"]}, "1": "high"},
                probabilities={"0": 0.2, "1": 0.8},
            ),
        },
        provider_metadata={"trace": {"steps": ["classify"]}},
    )

    with pytest.raises(ValidationError):
        result.calibrated = False  # type: ignore[misc]
    with pytest.raises(TypeError, match="immutable"):
        result.answers.pop("route")
    with pytest.raises(TypeError, match="immutable"):
        result.answers["route"].probabilities["billing"] = 0.2  # type: ignore[union-attr]
    with pytest.raises(TypeError, match="immutable"):
        result.answers["score"].legend["0"]["tags"].append("changed")  # type: ignore[index, union-attr]
    with pytest.raises(TypeError, match="immutable"):
        result.provider_metadata["trace"]["steps"].append("changed")  # type: ignore[index, union-attr]

    assert result.model_dump()["provider_metadata"] == {"trace": {"steps": ["classify"]}}
