"""Unit tests for the TypeSafe adapter mapping behavior."""

from __future__ import annotations

import pytest

from bruv.application import ProviderResponseError
from bruv.backends.typesafe import TypeSafeJevAdapter
from bruv.domain.questions import ChoiceQuestion, NoulQuestion, ScoreQuestion
from bruv.domain.requests import DecisionRequest
from tests.contract.test_typesafe_contract import (
    FakeAnswer,
    FakeClient,
    FakeResponse,
    _fake_response,
)


def _backend(
    response: FakeResponse | None = None, *, raise_with: BaseException | None = None
) -> TypeSafeJevAdapter:
    client = FakeClient(response, raise_with=raise_with)
    return TypeSafeJevAdapter(client, model="jev-latest")


def test_request_model_overrides_adapter_model() -> None:
    client = FakeClient(_fake_response())
    backend = TypeSafeJevAdapter(client, model="jev-latest")
    request = DecisionRequest(
        state="x",
        model="custom-model",
        questions={"q": NoulQuestion(instructions="Yes?")},
    )
    response = FakeResponse(
        model="custom-model",
        answers={"q": FakeAnswer(type="noul", noul=0.3)},
        usage=type("U", (), {"input_tokens": 1, "output_tokens": 0})(),
    )
    client = FakeClient(response)
    backend = TypeSafeJevAdapter(client, model="jev-latest")
    backend.evaluate(request)
    assert client.last_model == "custom-model"


def test_missing_answer_is_paid_provider_error() -> None:
    response = FakeResponse(
        model="jev-latest",
        answers={},
        usage=type("U", (), {"input_tokens": 1, "output_tokens": 0})(),
    )
    with pytest.raises(ProviderResponseError) as exc_info:
        _backend(response).evaluate(
            DecisionRequest(state="x", questions={"q": NoulQuestion(instructions="Yes?")})
        )
    assert exc_info.value.paid_request is True


def test_malformed_answer_is_paid_provider_error() -> None:
    response = FakeResponse(
        model="jev-latest",
        answers={"q": FakeAnswer(type="choice")},  # missing fields
        usage=type("U", (), {"input_tokens": 1, "output_tokens": 0})(),
    )
    with pytest.raises(ProviderResponseError):
        _backend(response).evaluate(
            DecisionRequest(
                state="x",
                questions={
                    "q": ChoiceQuestion(instructions="Route?", criteria={"a": None, "b": None})
                },
            )
        )


def test_blank_model_is_paid_provider_error() -> None:
    response = FakeResponse(
        model="  ",
        answers={"q": FakeAnswer(type="noul", noul=0.1)},
        usage=type("U", (), {"input_tokens": 1, "output_tokens": 0})(),
    )
    with pytest.raises(ProviderResponseError):
        _backend(response).evaluate(
            DecisionRequest(state="x", questions={"q": NoulQuestion(instructions="Yes?")})
        )


def test_score_answer_maps_legend_and_probabilities() -> None:
    response = FakeResponse(
        model="jev-latest",
        answers={
            "s": FakeAnswer(
                type="score",
                score=1.2,
                confidence=0.7,
                legend={"0": "low", "1": "high"},
                probabilities={"0": 0.4, "1": 0.6},
            )
        },
        usage=type("U", (), {"input_tokens": 1, "output_tokens": 0})(),
    )
    result = _backend(response).evaluate(
        DecisionRequest(
            state="x",
            questions={"s": ScoreQuestion(instructions="Rate?", criteria=["low", "high"])},
        )
    )
    assert result.answers["s"].score == pytest.approx(1.2)
    assert result.answers["s"].legend.keys() == result.answers["s"].probabilities.keys()


def test_noul_answer_maps() -> None:
    response = FakeResponse(
        model="jev-latest",
        answers={"q": FakeAnswer(type="noul", noul=0.66)},
        usage=type("U", (), {"input_tokens": 1, "output_tokens": 0})(),
    )
    result = _backend(response).evaluate(
        DecisionRequest(state="x", questions={"q": NoulQuestion(instructions="Yes?")})
    )
    assert result.answers["q"].noul == pytest.approx(0.66)


def test_request_id_included_when_present() -> None:
    response = _fake_response()
    result = _backend(response).evaluate(
        DecisionRequest(
            state="x",
            questions={
                "route": ChoiceQuestion(instructions="Route?", criteria={"a": None, "b": None})
            },
        )
    )
    # _fake_response answers include route
    assert result.request_id == "req-123"


def test_invalid_usage_is_paid_provider_error() -> None:
    response = FakeResponse(
        model="jev-latest",
        answers={"q": FakeAnswer(type="noul", noul=0.1)},
        usage=type("U", (), {"input_tokens": -5, "output_tokens": 0})(),
    )
    with pytest.raises(ProviderResponseError):
        _backend(response).evaluate(
            DecisionRequest(state="x", questions={"q": NoulQuestion(instructions="Yes?")})
        )
