from __future__ import annotations

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
