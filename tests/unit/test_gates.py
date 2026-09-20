"""Gate and exit policy tests."""

from __future__ import annotations

import pytest

from bruv.application import AuthenticationError, ProviderResponseError
from bruv.domain.results import ChoiceAnswer, DecisionResult, ScoreAnswer, Usage
from bruv.exit_codes import (
    ABSTAINED,
    AUTHENTICATION,
    GATE_FAILED,
    PROVIDER_RESPONSE,
    SUCCESS,
    USAGE_OR_VALIDATION,
    CommandOutcome,
    exit_code_for,
)
from bruv.gates import GateOptions, apply_gate


def _score_result(score: float) -> DecisionResult:
    return DecisionResult(
        backend="typesafe",
        model="jev-latest",
        calibrated=True,
        answers={
            "urgency": ScoreAnswer(
                type="score",
                score=score,
                confidence=0.8,
                legend={"0": "low", "1": "high"},
                probabilities={"0": 0.4, "1": 0.6},
            )
        },
        usage=Usage(input_tokens=10, output_tokens=0),
    )


def test_fail_under_passes_when_above() -> None:
    outcome = apply_gate(
        _score_result(0.8), GateOptions(field="answers.urgency.score", fail_under=0.5)
    )
    assert outcome.gate_passed is True
    assert outcome.abstained is False


def test_fail_under_fails_when_below() -> None:
    outcome = apply_gate(
        _score_result(0.3), GateOptions(field="answers.urgency.score", fail_under=0.5)
    )
    assert outcome.gate_passed is False


def test_abstain_band_abstains() -> None:
    outcome = apply_gate(
        _score_result(0.5),
        GateOptions(field="answers.urgency.score", abstain_band=(0.4, 0.6)),
    )
    assert outcome.abstained is True


def test_no_gate_passes() -> None:
    outcome = apply_gate(_score_result(0.5), GateOptions())
    assert outcome.abstained is False
    assert outcome.gate_passed is None


def test_exit_code_success() -> None:
    assert exit_code_for(CommandOutcome()) == SUCCESS


def test_exit_code_gate_failed() -> None:
    assert exit_code_for(CommandOutcome(gate_passed=False)) == GATE_FAILED


def test_exit_code_abstained() -> None:
    assert exit_code_for(CommandOutcome(abstained=True)) == ABSTAINED


def test_exit_code_authentication_error_takes_precedence() -> None:
    outcome = CommandOutcome(
        error=AuthenticationError(message="x", paid_request=False), gate_passed=False
    )
    assert exit_code_for(outcome) == AUTHENTICATION


def test_exit_code_provider_response_error() -> None:
    outcome = CommandOutcome(error=ProviderResponseError(message="x", paid_request=True))
    assert exit_code_for(outcome) == PROVIDER_RESPONSE


def test_exit_code_validation_error() -> None:
    from bruv.application import RequestValidationError

    outcome = CommandOutcome(error=RequestValidationError(message="x", paid_request=False))
    assert exit_code_for(outcome) == USAGE_OR_VALIDATION


def test_field_must_be_numeric() -> None:
    result = DecisionResult(
        backend="simple-jev",
        model="m",
        calibrated=False,
        answers={
            "route": ChoiceAnswer(
                type="choice", choice="a", confidence=0.5, probabilities={"a": 1.0}
            )
        },
    )
    with pytest.raises(Exception):  # noqa: B017
        apply_gate(result, GateOptions(field="answers.route.choice", fail_under=0.5))
