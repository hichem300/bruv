"""TypeSafe Jev adapter.

Provider SDK types stay behind a local :class:`TypeSafeClientPort`. The real
SDK is imported only in composition code (factory), so adapter tests run with
fakes and never require the ``typesafe-sdk`` package. TypeSafe probabilities are
provider-documented calibrated probabilities, so canonical results report
``calibrated=True``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from pydantic import ValidationError

from bruv.application import ApplicationError, ProviderResponseError
from bruv.backends.interface import DecisionBackend
from bruv.domain.questions import (
    ChoiceQuestion,
    JsonValue,
    NoulQuestion,
    ScoreQuestion,
)
from bruv.domain.requests import DecisionRequest
from bruv.domain.results import (
    ChoiceAnswer,
    DecisionResult,
    NoulAnswer,
    ScoreAnswer,
    Usage,
)
from bruv.domain.validation import BackendCapabilities

_CAPABILITIES = BackendCapabilities(
    backend="typesafe",
    question_types=frozenset({"noul", "choice", "score"}),
    calibrated=True,
    allows_json_state=True,
)


@dataclass(frozen=True, slots=True)
class ProviderNoul:
    type: str = "noul"
    instructions: JsonValue | None = None
    criteria: Mapping[str, JsonValue] | None = None


@dataclass(frozen=True, slots=True)
class ProviderChoice:
    criteria: Mapping[str, JsonValue | None]
    type: str = "choice"
    instructions: JsonValue | None = None


@dataclass(frozen=True, slots=True)
class ProviderScore:
    criteria: Sequence[JsonValue]
    type: str = "score"
    instructions: JsonValue | None = None


ProviderQuestion = ProviderNoul | ProviderChoice | ProviderScore


class ProviderAnswer(Protocol):
    type: str


class ProviderUsage(Protocol):
    input_tokens: int | None
    output_tokens: int | None


class TypeSafeResponse(Protocol):
    model: str
    answers: Mapping[str, ProviderAnswer]
    usage: ProviderUsage
    request_id: str | None


@runtime_checkable
class TypeSafeClientPort(Protocol):
    """Local port matching only the SDK members bruv uses."""

    def system_one(
        self,
        *,
        state: Any,
        questions: Mapping[str, ProviderQuestion],
        model: str | None = None,
    ) -> TypeSafeResponse: ...


def _to_provider_question(question: Any) -> ProviderQuestion:
    if isinstance(question, NoulQuestion):
        return ProviderNoul(
            instructions=question.instructions,
            criteria=None
            if question.criteria is None
            else {str(k): v for k, v in question.criteria.items()},
        )
    if isinstance(question, ChoiceQuestion):
        return ProviderChoice(instructions=question.instructions, criteria=dict(question.criteria))
    if isinstance(question, ScoreQuestion):
        return ProviderScore(instructions=question.instructions, criteria=list(question.criteria))
    raise TypeError(f"unsupported question type: {type(question).__name__}")


def _answer_value(payload: Any, field: str) -> Any:
    if not hasattr(payload, field):
        raise ProviderResponseError(
            message="TypeSafe answer is missing a required field.",
            paid_request=True,
            action="Retry the request; report the response if it persists.",
        )
    return getattr(payload, field)


def _to_answer(question: Any, payload: Any) -> Any:
    answer_type = getattr(payload, "type", None)
    try:
        if isinstance(question, NoulQuestion):
            if answer_type != "noul":
                raise ValueError("expected noul answer")
            return NoulAnswer(type="noul", noul=_answer_value(payload, "noul"))
        if isinstance(question, ChoiceQuestion):
            if answer_type != "choice":
                raise ValueError("expected choice answer")
            return ChoiceAnswer(
                type="choice",
                choice=_answer_value(payload, "choice"),
                confidence=_answer_value(payload, "confidence"),
                probabilities=_answer_value(payload, "probabilities"),
            )
        if isinstance(question, ScoreQuestion):
            if answer_type != "score":
                raise ValueError("expected score answer")
            return ScoreAnswer(
                type="score",
                score=_answer_value(payload, "score"),
                confidence=_answer_value(payload, "confidence"),
                legend=_answer_value(payload, "legend"),
                probabilities=_answer_value(payload, "probabilities"),
            )
    except (ValidationError, ValueError, TypeError) as exc:
        raise ProviderResponseError(
            message="TypeSafe answer did not match canonical schema.",
            paid_request=True,
            action="Retry the request; report the response if it persists.",
        ) from exc

    raise TypeError(f"unsupported question type: {type(question).__name__}")


class TypeSafeJevAdapter(DecisionBackend):
    """Adapter for the TypeSafe SDK ``system_one`` call."""

    capabilities = _CAPABILITIES

    def __init__(self, client: TypeSafeClientPort, model: str | None) -> None:
        self._client = client
        self._model = model

    def evaluate(self, request: DecisionRequest) -> DecisionResult:
        provider_questions = {
            key: _to_provider_question(question) for key, question in request.questions.items()
        }
        try:
            response = self._client.system_one(
                state=request.state,
                questions=provider_questions,
                model=request.model or self._model,
            )
        except ProviderResponseError:
            raise
        except ApplicationError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ProviderResponseError(
                message="TypeSafe backend raised an unexpected error.",
                paid_request=True,
                action="Retry the request; report the error if it persists.",
            ) from exc
        return self._to_result(response, request)

    def _to_result(self, response: TypeSafeResponse, request: DecisionRequest) -> DecisionResult:
        model = getattr(response, "model", None)
        if not isinstance(model, str) or not model.strip():
            raise ProviderResponseError(
                message="TypeSafe response missing model identifier.",
                paid_request=True,
                action="Retry the request; report the response if it persists.",
            )

        answers_payload = getattr(response, "answers", None)
        if not isinstance(answers_payload, Mapping):
            raise ProviderResponseError(
                message="TypeSafe response missing answers mapping.",
                paid_request=True,
                action="Retry the request; report the response if it persists.",
            )

        answers = {
            question_id: _to_answer(question, answers_payload[question_id])
            for question_id, question in request.questions.items()
            if question_id in answers_payload
        }
        missing = set(request.questions) - set(answers)
        if missing:
            raise ProviderResponseError(
                message="TypeSafe response missing one or more answers.",
                paid_request=True,
                action="Retry the request; report the response if it persists.",
            )

        usage = None
        usage_payload = getattr(response, "usage", None)
        if usage_payload is not None:
            try:
                usage = Usage(
                    input_tokens=getattr(usage_payload, "input_tokens", None),
                    output_tokens=getattr(usage_payload, "output_tokens", None),
                )
            except ValidationError as exc:
                raise ProviderResponseError(
                    message="TypeSafe usage did not match canonical schema.",
                    paid_request=True,
                    action="Retry the request; report the response if it persists.",
                ) from exc

        request_id = getattr(response, "request_id", None)
        request_id_value = (
            request_id if isinstance(request_id, str) and request_id.strip() else None
        )

        return DecisionResult(
            backend="typesafe",
            model=model,
            calibrated=True,
            answers=answers,
            usage=usage,
            request_id=request_id_value,
        )


__all__ = ["TypeSafeClientPort", "TypeSafeJevAdapter"]
