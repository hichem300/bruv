"""TypeSafe backend contract: canonical mapping and calibration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pytest

from bruv.backends.typesafe import TypeSafeJevAdapter
from bruv.domain.results import ChoiceAnswer, NoulAnswer, ScoreAnswer
from tests.contract._shared import mixed_request, mixed_response
from tests.contract.backend_contract import assert_backend_contract


@dataclass(frozen=True, slots=True)
class FakeAnswer:
    type: str
    noul: float | None = None
    choice: str | None = None
    confidence: float | None = None
    probabilities: dict[str, float] | None = None
    score: float | None = None
    legend: dict[str, str] | None = None


@dataclass(frozen=True, slots=True)
class FakeUsage:
    input_tokens: int | None
    output_tokens: int | None


@dataclass(frozen=True, slots=True)
class FakeResponse:
    model: str
    answers: Mapping[str, FakeAnswer]
    usage: FakeUsage
    request_id: str | None = None


def _fake_response() -> FakeResponse:
    src = mixed_response()
    answers: dict[str, FakeAnswer] = {}
    for key, ans in src["answers"].items():
        answers[key] = FakeAnswer(
            type=ans["type"],
            noul=ans.get("noul"),
            choice=ans.get("choice"),
            confidence=ans.get("confidence"),
            probabilities=ans.get("probabilities"),
            score=ans.get("score"),
            legend=ans.get("legend"),
        )
    usage = src["usage"]
    return FakeResponse(
        model=src["model"],
        answers=answers,
        usage=FakeUsage(input_tokens=usage["input_tokens"], output_tokens=usage["output_tokens"]),
        request_id="req-123",
    )


class FakeClient:
    def __init__(
        self, response: FakeResponse | None = None, *, raise_with: BaseException | None = None
    ) -> None:
        self.calls = 0
        self.last_state: Any = None
        self.last_questions: Any = None
        self.last_model: str | None = None
        self._response = response
        self._raise_with = raise_with

    def system_one(
        self,
        *,
        state: Any,
        questions: Mapping[str, Any],
        model: str | None = None,
    ) -> FakeResponse:
        self.calls += 1
        self.last_state = state
        self.last_questions = questions
        self.last_model = model
        if self._raise_with is not None:
            raise self._raise_with
        assert self._response is not None
        return self._response


def test_typesafe_satisfies_backend_contract() -> None:
    client = FakeClient(_fake_response())
    backend = TypeSafeJevAdapter(client, model="jev-latest")
    result = assert_backend_contract(backend, mixed_request())
    assert result.calibrated is True
    assert result.backend == "typesafe"
    assert result.request_id == "req-123"
    assert isinstance(result.answers["route"], ChoiceAnswer)
    assert isinstance(result.answers["urgency"], ScoreAnswer)
    assert isinstance(result.answers["asks_for_refund"], NoulAnswer)


def test_typesafe_always_reports_calibrated() -> None:
    client = FakeClient(_fake_response())
    backend = TypeSafeJevAdapter(client, model="jev-latest")
    result = backend.evaluate(mixed_request())
    assert result.calibrated is True


def test_typesafe_passes_provider_questions_and_model() -> None:
    client = FakeClient(_fake_response())
    backend = TypeSafeJevAdapter(client, model="jev-latest")
    request = mixed_request().model_copy(update={"model": "jev-1.13.0"})
    backend.evaluate(request)
    assert client.calls == 1
    assert client.last_state == request.state
    assert client.last_model == "jev-1.13.0"
    assert set(client.last_questions) == set(request.questions)
    assert client.last_questions["route"].type == "choice"
    assert client.last_questions["urgency"].type == "score"
    assert client.last_questions["asks_for_refund"].type == "noul"


def test_typesafe_missing_credentials_map_to_authentication() -> None:
    from bruv.application import AuthenticationError

    client = FakeClient(
        raise_with=AuthenticationError(message="missing key", paid_request=False, action="set key")
    )
    backend = TypeSafeJevAdapter(client, model="jev-latest")
    with pytest.raises(AuthenticationError) as exc_info:
        backend.evaluate(mixed_request())
    assert exc_info.value.paid_request is False


def test_typesafe_timeout_maps_to_backend_unavailable() -> None:
    from bruv.application import BackendUnavailableError

    client = FakeClient(
        raise_with=BackendUnavailableError(message="timed out", paid_request=False, action="retry")
    )
    backend = TypeSafeJevAdapter(client, model="jev-latest")
    with pytest.raises(BackendUnavailableError):
        backend.evaluate(mixed_request())


def test_typesafe_malformed_response_maps_to_provider_error() -> None:
    from bruv.application import ProviderResponseError

    response = _fake_response()
    malformed = FakeResponse(
        model=response.model,
        answers={},
        usage=response.usage,
        request_id=response.request_id,
    )
    client = FakeClient(malformed)
    backend = TypeSafeJevAdapter(client, model="jev-latest")
    with pytest.raises(ProviderResponseError) as exc_info:
        backend.evaluate(mixed_request())
    assert exc_info.value.paid_request is True


def test_typesafe_errors_do_not_leak_secret() -> None:
    secret = "bruv_test_secret_DO_NOT_PRINT"  # noqa: S105

    class LeakyClient:
        def system_one(self, **_: Any) -> FakeResponse:
            raise RuntimeError(secret)

    backend = TypeSafeJevAdapter(LeakyClient(), model="jev-latest")
    try:
        backend.evaluate(mixed_request())
    except Exception as exc:  # noqa: BLE001
        assert secret not in str(exc)
    else:
        pytest.fail("expected an error")
