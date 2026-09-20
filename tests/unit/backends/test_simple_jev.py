"""Unit tests for the Simple Jev adapter HTTP and mapping behavior."""

from __future__ import annotations

import httpx
import pytest

from bruv.application import BackendUnavailableError, ProviderResponseError
from bruv.backends.simple_jev import SimpleJevAdapter
from bruv.domain.questions import ChoiceQuestion, NoulQuestion, ScoreQuestion
from bruv.domain.requests import DecisionRequest
from tests.contract._shared import mixed_request, mixed_response

BASE_URL = "http://127.0.0.1:8000"
MODEL = "Qwen/Qwen3.5-0.8B"


def _client(handler: httpx.MockTransport) -> httpx.Client:
    return httpx.Client(transport=handler, follow_redirects=False, verify=True)


def _adapter(handler: httpx.MockTransport) -> SimpleJevAdapter:
    return SimpleJevAdapter(base_url=BASE_URL, model=MODEL, client=_client(handler))


def test_payload_uses_request_model_when_provided() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["body"] = json.loads(request.read())
        return httpx.Response(200, json=mixed_response())

    adapter = _adapter(httpx.MockTransport(handler))
    request = mixed_request().model_copy(update={"model": "custom-model"})
    adapter.evaluate(request)
    assert captured["body"]["model"] == "custom-model"


def test_payload_uses_adapter_model_when_request_omits_it() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["body"] = json.loads(request.read())
        return httpx.Response(200, json=mixed_response())

    _adapter(httpx.MockTransport(handler)).evaluate(mixed_request())
    assert captured["body"]["model"] == MODEL


def test_payload_shape_matches_upstream_schema() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["body"] = json.loads(request.read())
        return httpx.Response(200, json=mixed_response())

    _adapter(httpx.MockTransport(handler)).evaluate(mixed_request())
    body = captured["body"]
    assert set(body) >= {"model", "state", "messages", "questions", "options", "tools"}
    assert body["options"] == {"raw_logits": False}
    assert body["tools"] is None
    questions = body["questions"]
    assert questions["route"]["type"] == "choice"
    assert questions["urgency"]["type"] == "score"
    assert questions["asks_for_refund"]["type"] == "noul"
    assert questions["asks_for_refund"]["criteria"] is None


def test_missing_answer_is_paid_provider_error() -> None:
    response = mixed_response()
    del response["answers"]["urgency"]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=response)

    with pytest.raises(ProviderResponseError) as exc_info:
        _adapter(httpx.MockTransport(handler)).evaluate(mixed_request())
    assert exc_info.value.paid_request is True


def test_malformed_answer_is_paid_provider_error() -> None:
    response = mixed_response()
    response["answers"]["route"] = {"type": "choice"}  # missing fields

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=response)

    with pytest.raises(ProviderResponseError):
        _adapter(httpx.MockTransport(handler)).evaluate(mixed_request())


def test_422_validation_error_is_not_paid() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"detail": "bad request"})

    with pytest.raises(ProviderResponseError) as exc_info:
        _adapter(httpx.MockTransport(handler)).evaluate(mixed_request())
    assert exc_info.value.paid_request is False


def test_429_queue_full_is_backend_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"detail": "Scoring queue is full"})

    with pytest.raises(BackendUnavailableError) as exc_info:
        _adapter(httpx.MockTransport(handler)).evaluate(mixed_request())
    assert exc_info.value.paid_request is False


def test_500_server_error_is_paid_provider_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=b"boom")

    with pytest.raises(ProviderResponseError) as exc_info:
        _adapter(httpx.MockTransport(handler)).evaluate(mixed_request())
    assert exc_info.value.paid_request is True


def test_network_error_is_backend_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route")

    with pytest.raises(BackendUnavailableError):
        _adapter(httpx.MockTransport(handler)).evaluate(mixed_request())


def test_invalid_usage_payload_is_paid_provider_error() -> None:
    response = mixed_response()
    response["usage"] = {"input_tokens": -1}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=response)

    with pytest.raises(ProviderResponseError):
        _adapter(httpx.MockTransport(handler)).evaluate(mixed_request())


def test_missing_model_identifier_is_paid_provider_error() -> None:
    response = mixed_response()
    response["model"] = "   "

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=response)

    with pytest.raises(ProviderResponseError):
        _adapter(httpx.MockTransport(handler)).evaluate(mixed_request())


def test_non_positive_timeout_rejected() -> None:
    with pytest.raises(ValueError):
        SimpleJevAdapter(
            base_url=BASE_URL,
            model=MODEL,
            client=_client(httpx.MockTransport(lambda r: httpx.Response(200))),
            timeout_seconds=0,
        )


def test_score_legend_keys_must_match_probabilities() -> None:
    response = mixed_response()
    response["answers"]["urgency"]["legend"]["3"] = "critical"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=response)

    with pytest.raises(ProviderResponseError):
        _adapter(httpx.MockTransport(handler)).evaluate(mixed_request())


def test_single_choice_request_maps() -> None:
    response = {
        "model": MODEL,
        "answers": {
            "route": {
                "type": "choice",
                "choice": "billing",
                "confidence": 0.9,
                "probabilities": {"billing": 0.9, "other": 0.1},
            }
        },
        "usage": {"input_tokens": 40, "output_tokens": 0},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=response)

    request = DecisionRequest(
        state="help",
        questions={
            "route": ChoiceQuestion(
                instructions="Route?", criteria={"sales": None, "billing": None}
            )
        },
    )
    result = _adapter(httpx.MockTransport(handler)).evaluate(request)
    assert result.answers["route"].choice == "billing"


def test_noul_only_request_maps() -> None:
    response = {
        "model": MODEL,
        "answers": {"q": {"type": "noul", "noul": 0.4}},
        "usage": {"input_tokens": 10, "output_tokens": 0},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=response)

    request = DecisionRequest(state="x", questions={"q": NoulQuestion(instructions="Yes?")})
    result = _adapter(httpx.MockTransport(handler)).evaluate(request)
    assert result.answers["q"].noul == pytest.approx(0.4)


def test_score_only_request_maps() -> None:
    response = {
        "model": MODEL,
        "answers": {
            "s": {
                "type": "score",
                "score": 0.5,
                "confidence": 0.6,
                "legend": {"0": "low", "1": "high"},
                "probabilities": {"0": 0.5, "1": 0.5},
            }
        },
        "usage": {"input_tokens": 5, "output_tokens": 0},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=response)

    request = DecisionRequest(
        state="x", questions={"s": ScoreQuestion(instructions="Rate?", criteria=["low", "high"])}
    )
    result = _adapter(httpx.MockTransport(handler)).evaluate(request)
    assert result.answers["s"].score == pytest.approx(0.5)
