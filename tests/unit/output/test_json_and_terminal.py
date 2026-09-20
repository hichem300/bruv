"""JSON and terminal output determinism tests."""

from __future__ import annotations

import json

from bruv.application import ProviderResponseError
from bruv.domain.results import ChoiceAnswer, DecisionResult, Usage
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


def test_render_error_envelope_has_paid_request() -> None:
    error = ProviderResponseError(message="bad", paid_request=True, action="retry")
    payload = json.loads(render_error(error))
    assert payload["ok"] is False
    assert payload["error"]["code"] == "provider_response_error"
    assert payload["error"]["paid_request"] is True


def test_human_result_includes_uncalibrated_for_simple_jev() -> None:
    text = render_human_result(_result(calibrated=False))
    assert "uncalibrated" in text


def test_human_result_includes_calibrated_for_typesafe() -> None:
    text = render_human_result(_result(calibrated=True))
    assert "calibrated" in text


def test_human_error_stable() -> None:
    error = ProviderResponseError(message="bad", paid_request=True, action="retry")
    text = render_human_error(error)
    assert text.startswith("error: provider_response_error")
