"""JSON and terminal output determinism tests."""

from __future__ import annotations

import json

from bruv.application import ProviderResponseError
from bruv.domain.results import ChoiceAnswer, DecisionResult, NoulAnswer, ScoreAnswer, Usage
from bruv.output.json_output import render_error, render_json, render_success
from bruv.output.terminal import render_human_error, render_human_result


def _result(calibrated: bool = False) -> DecisionResult:
    return DecisionResult(
        backend="simple-jev",
        model="Qwen/Qwen3.5-0.8B",
        calibrated=calibrated,
        answers={
            "route": ChoiceAnswer(
                type="choice",
                choice="billing",
                confidence=0.9,
                probabilities={"billing": 0.9, "other": 0.1},
            )
        },
        usage=Usage(input_tokens=40, output_tokens=0),
    )


def test_render_json_is_sorted_single_line_terminated() -> None:
    text = render_json({"b": 1, "a": 2})
    assert text.endswith("\n")
    assert text.count("\n") == 1
    parsed = json.loads(text)
    assert list(parsed.keys()) == ["a", "b"]


def test_render_success_envelope_has_required_keys() -> None:
    text = render_success(_result())
    payload = json.loads(text)
    for key in ("backend", "model", "calibrated", "answers"):
        assert key in payload


def test_render_success_preserves_existing_provider_answer_shapes() -> None:
    result = DecisionResult(
        backend="typesafe",
        model="jev-1.13.0",
        calibrated=True,
        answers={
            "refund": NoulAnswer(noul=0.75),
            "route": ChoiceAnswer(
                choice="billing",
                confidence=0.9,
                probabilities={"billing": 0.9, "other": 0.1},
            ),
            "urgency": ScoreAnswer(
                score=1.8,
                confidence=0.8,
                legend={"0": "low", "1": "medium", "2": "high"},
                probabilities={"0": 0.0, "1": 0.2, "2": 0.8},
            ),
        },
    )

    assert json.loads(render_success(result)) == {
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
        "usage": None,
        "latency_ms": None,
        "request_id": None,
        "provider_metadata": None,
    }


def test_render_success_omits_needle_unavailable_answer_fields() -> None:
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

    assert json.loads(render_success(result)) == {
        "backend": "needle",
        "model": "Cactus-Compute/needle3",
        "calibrated": True,
        "answers": {
            "refund": {"type": "noul", "value": True, "confidence": 0.9},
            "route": {"type": "choice", "choice": "billing", "confidence": 0.8},
            "urgency": {
                "type": "score",
                "score": 2.0,
                "confidence": 0.7,
                "legend": {"0": "low", "1": "medium", "2": "high"},
            },
        },
        "usage": None,
        "latency_ms": None,
        "request_id": None,
        "provider_metadata": None,
    }


def test_render_error_envelope_has_paid_request() -> None:
    error = ProviderResponseError(message="bad", paid_request=True, action="retry")
    payload = json.loads(render_error(error))
    assert payload["ok"] is False
    assert payload["error"]["code"] == "provider_response_error"
    assert payload["error"]["paid_request"] is True


def test_human_result_includes_uncalibrated_for_simple_jev() -> None:
    text = render_human_result(_result(calibrated=False))
    assert text == (
        "backend: simple-jev\nmodel: Qwen/Qwen3.5-0.8B\ncalibration: uncalibrated\nroute: billing\n"
    )


def test_human_result_preserves_probability_answer_rendering() -> None:
    result = DecisionResult(
        backend="typesafe",
        model="jev-1.13.0",
        calibrated=True,
        answers={
            "refund": NoulAnswer(noul=0.75),
            "route": ChoiceAnswer(
                choice="billing",
                confidence=0.9,
                probabilities={"billing": 0.9, "other": 0.1},
            ),
            "urgency": ScoreAnswer(
                score=1.8,
                confidence=0.8,
                legend={"0": "low", "1": "medium", "2": "high"},
                probabilities={"0": 0.0, "1": 0.2, "2": 0.8},
            ),
        },
    )

    assert render_human_result(result) == (
        "backend: typesafe\n"
        "model: jev-1.13.0\n"
        "calibration: calibrated\n"
        "refund: 0.750\n"
        "route: billing\n"
        "urgency: 1.800\n"
    )


def test_human_result_renders_confidence_only_answers() -> None:
    result = DecisionResult(
        backend="needle",
        model="Cactus-Compute/needle3",
        calibrated=True,
        answers={
            "refund": NoulAnswer(value=False, confidence=0.9),
            "route": ChoiceAnswer(choice="billing", confidence=0.8),
            "urgency": ScoreAnswer(
                score=2.0,
                confidence=0.7,
                legend={"0": "low", "1": "medium", "2": "high"},
            ),
        },
    )

    assert render_human_result(result) == (
        "backend: needle\n"
        "model: Cactus-Compute/needle3\n"
        "calibration: calibrated\n"
        "refund: false (confidence: 0.900)\n"
        "route: billing (confidence: 0.800)\n"
        "urgency: 2.000 (confidence: 0.700)\n"
    )


def test_human_result_includes_calibrated_for_typesafe() -> None:
    text = render_human_result(_result(calibrated=True))
    assert "calibrated" in text


def test_human_error_stable() -> None:
    error = ProviderResponseError(message="bad", paid_request=True, action="retry")
    text = render_human_error(error)
    assert text.startswith("error: provider_response_error")
