"""OpenRouter backend adapter for the calibrated typed Decisions API.

OpenRouter routes unclassified model aliases to whatever model it currently
prefers, which would make bruv decisions non-reproducible and could silently
send data somewhere the operator did not choose. Every OpenRouter model
reference in bruv must therefore name one concrete model; the shared
:func:`require_concrete_model` helper enforces that at both config load and
per-request resolution time.

This adapter targets OpenRouter's calibrated typed Decisions API at
``POST /api/alpha/decisions`` (required for ``~typesafe/jev-latest``). Provider
request/response shapes stay inside this module; the adapter maps the canonical
:class:`DecisionRequest` to the upstream request schema and the response back
to a canonical :class:`DecisionResult`. Decisions API probabilities are
provider-documented calibrated probabilities, so canonical results report
``calibrated=True``.
"""

from __future__ import annotations

from typing import Any

import httpx
from pydantic import TypeAdapter, ValidationError

from bruv.application import (
    BackendUnavailableError,
    ConfigurationError,
    ProviderResponseError,
)
from bruv.backends.interface import DecisionBackend
from bruv.domain.questions import ChoiceQuestion, NoulQuestion, ScoreQuestion
from bruv.domain.requests import DecisionRequest
from bruv.domain.results import (
    ChoiceAnswer,
    DecisionResult,
    NoulAnswer,
    ScoreAnswer,
    Usage,
)
from bruv.domain.validation import BackendCapabilities

RESERVED_ROUTER_ALIAS = "openrouter/auto"

_DEFAULT_ENDPOINT = "https://openrouter.ai/api/alpha/decisions"

_CAPABILITIES = BackendCapabilities(
    backend="openrouter",
    question_types=frozenset({"noul", "choice", "score"}),
    calibrated=True,
    allows_json_state=True,
)

_USAGE_ADAPTER: TypeAdapter[Usage] = TypeAdapter(Usage)


def require_concrete_model(model: str, source: str) -> str:
    """Return ``model`` when it names one concrete OpenRouter model.

    ``source`` names where the model came from (for example ``"config"`` or
    ``"request"``) so the error can point at the right knob. Raises an unpaid
    :class:`ConfigurationError` for blank values, surrounding whitespace, or
    the reserved router alias.
    """
    stripped = model.strip()
    if not stripped:
        raise ConfigurationError(
            message=f"OpenRouter model from {source} must not be blank.",
            paid_request=False,
            action=f"Set a concrete OpenRouter model in {source}, "
            f"in 'provider/model' format, such as '~typesafe/jev-latest' "
            f"or 'anthropic/claude-3.5-haiku'.",
        )
    if stripped != model:
        raise ConfigurationError(
            message=f"OpenRouter model from {source} must not have surrounding whitespace.",
            paid_request=False,
            action=f"Remove leading or trailing spaces from the model in {source}.",
        )
    if model == RESERVED_ROUTER_ALIAS:
        raise ConfigurationError(
            message=f"OpenRouter model from {source} is the reserved router "
            f"alias '{RESERVED_ROUTER_ALIAS}'.",
            paid_request=False,
            action=f"Set a concrete OpenRouter model in {source} instead of "
            f"'{RESERVED_ROUTER_ALIAS}'.",
        )
    if "/" not in model:
        raise ConfigurationError(
            message=f"OpenRouter model from {source} must name one concrete "
            f"model as 'provider/model', got {model!r}.",
            paid_request=False,
            action=f"Enter a concrete OpenRouter model slug in 'provider/model' "
            f"format, such as '~typesafe/jev-latest' or "
            f"'anthropic/claude-3.5-haiku'.",
        )
    return model


def _question_payload(question: Any) -> dict[str, Any]:
    """Serialize a canonical question from ``.type``, ``.instructions`` and
    ``.criteria``: score criteria as a list, choice criteria as a dict, and
    noul criteria as an optional dict."""
    if isinstance(question, NoulQuestion):
        payload: dict[str, Any] = {
            "type": "noul",
            "instructions": question.instructions,
        }
        if question.criteria is not None:
            payload["criteria"] = {key: value for key, value in question.criteria.items()}
        else:
            payload["criteria"] = None
        return payload

    if isinstance(question, ChoiceQuestion):
        return {
            "type": "choice",
            "instructions": question.instructions,
            "criteria": {key: value for key, value in question.criteria.items()},
        }

    if isinstance(question, ScoreQuestion):
        return {
            "type": "score",
            "instructions": question.instructions,
            "criteria": [value for value in question.criteria],
        }

    raise TypeError(f"unsupported question type: {type(question).__name__}")


def _to_payload(request: DecisionRequest, model: str) -> dict[str, Any]:
    return {
        "model": model,
        "state": request.state,
        "questions": {
            question_id: _question_payload(question)
            for question_id, question in request.questions.items()
        },
    }


def _answer_payload(question_id: str, answers: dict[str, Any]) -> Any:
    if question_id not in answers:
        raise ProviderResponseError(
            message=f"backend response missing answer for question '{question_id}'",
            paid_request=True,
            action="Retry the request; report the question ID if it persists.",
        )
    return answers[question_id]


def _to_answer(question: Any, payload: Any) -> Any:
    if not isinstance(payload, dict):
        raise ProviderResponseError(
            message="backend answer must be a JSON object",
            paid_request=True,
            action="Retry the request; report the response if it persists.",
        )
    answer_type = payload.get("type")
    try:
        if isinstance(question, NoulQuestion):
            if answer_type != "noul":
                raise ValueError("expected noul answer")
            return NoulAnswer(type="noul", noul=payload["noul"])
        if isinstance(question, ChoiceQuestion):
            if answer_type != "choice":
                raise ValueError("expected choice answer")
            return ChoiceAnswer(
                type="choice",
                choice=payload["choice"],
                confidence=payload["confidence"],
                probabilities=payload["probabilities"],
            )
        if isinstance(question, ScoreQuestion):
            if answer_type != "score":
                raise ValueError("expected score answer")
            return ScoreAnswer(
                type="score",
                score=payload["score"],
                confidence=payload["confidence"],
                legend=payload["legend"],
                probabilities=payload["probabilities"],
            )
    except (KeyError, ValidationError, ValueError, TypeError) as exc:
        raise ProviderResponseError(
            message="backend answer did not match canonical schema",
            paid_request=True,
            action="Retry the request; report the response if it persists.",
        ) from exc

    raise TypeError(f"unsupported question type: {type(question).__name__}")


class OpenRouterAdapter(DecisionBackend):
    """Adapter for OpenRouter's calibrated typed Decisions API."""

    capabilities = _CAPABILITIES

    def __init__(
        self,
        *,
        client: httpx.Client,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 30.0,
        endpoint: str = _DEFAULT_ENDPOINT,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        self._client = client
        self._api_key = api_key
        self._model = model
        self._timeout = float(timeout)
        self._endpoint = endpoint

    def evaluate(self, request: DecisionRequest) -> DecisionResult:
        model = request.model or self._model
        if not isinstance(model, str):
            raise ConfigurationError(
                message="OpenRouter requires a concrete model for every request.",
                paid_request=False,
                action="Set a concrete OpenRouter model in config or on the request.",
            )
        model = require_concrete_model(model, "request")

        body = _to_payload(request, model)
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        try:
            response = self._client.post(
                self._endpoint,
                headers=headers,
                json=body,
                timeout=self._timeout,
            )
        except httpx.TimeoutException as exc:
            raise BackendUnavailableError(
                message="OpenRouter request timed out before any response.",
                paid_request=False,
                action="Check network connectivity and retry.",
            ) from exc
        except httpx.NetworkError as exc:
            raise BackendUnavailableError(
                message="OpenRouter endpoint could not be reached.",
                paid_request=False,
                action="Check network connectivity and retry.",
            ) from exc

        self._raise_for_status(response)

        try:
            response_body = response.json()
        except ValueError as exc:
            raise ProviderResponseError(
                message="OpenRouter response was not valid JSON.",
                paid_request=True,
                action="Retry the request; report the response if it persists.",
            ) from exc

        return self._to_result(response_body, request)

    def _raise_for_status(self, response: httpx.Response) -> None:
        status = response.status_code
        if status == 401 or status == 403:
            raise ProviderResponseError(
                message="OpenRouter rejected the API key.",
                paid_request=False,
                action="Fix the configured OpenRouter API key and retry.",
            )
        if status == 402:
            raise ProviderResponseError(
                message="OpenRouter reported insufficient credits.",
                paid_request=False,
                action="Top up OpenRouter credits and retry.",
            )
        if status == 422:
            raise ProviderResponseError(
                message="OpenRouter rejected the request as invalid.",
                paid_request=False,
                action="Fix the request and retry.",
            )
        if status == 429:
            raise BackendUnavailableError(
                message="OpenRouter is rate limiting requests.",
                paid_request=False,
                action="Wait briefly and retry.",
            )
        if status >= 500:
            raise ProviderResponseError(
                message="OpenRouter returned a server error.",
                paid_request=True,
                action="Retry the request; report the error if it persists.",
            )
        if status >= 400:
            raise ProviderResponseError(
                message=f"OpenRouter returned status {status}.",
                paid_request=False,
                action="Fix the request and retry.",
            )

    def _to_result(self, body: Any, request: DecisionRequest) -> DecisionResult:
        if not isinstance(body, dict):
            raise ProviderResponseError(
                message="OpenRouter response must be a JSON object.",
                paid_request=True,
                action="Retry the request; report the response if it persists.",
            )

        model = body.get("model")
        if not isinstance(model, str) or not model.strip():
            raise ProviderResponseError(
                message="OpenRouter response missing model identifier.",
                paid_request=True,
                action="Retry the request; report the response if it persists.",
            )

        answers_payload = body.get("answers")
        if not isinstance(answers_payload, dict):
            raise ProviderResponseError(
                message="OpenRouter response missing answers object.",
                paid_request=True,
                action="Retry the request; report the response if it persists.",
            )

        answers = {
            question_id: _to_answer(question, _answer_payload(question_id, answers_payload))
            for question_id, question in request.questions.items()
        }

        usage = None
        if "usage" in body:
            try:
                usage = _USAGE_ADAPTER.validate_python(body["usage"])
            except ValidationError as exc:
                raise ProviderResponseError(
                    message="OpenRouter usage did not match canonical schema.",
                    paid_request=True,
                    action="Retry the request; report the response if it persists.",
                ) from exc

        request_id = body.get("request_id")
        if request_id is not None and not isinstance(request_id, str):
            raise ProviderResponseError(
                message="OpenRouter request_id must be a string.",
                paid_request=True,
                action="Retry the request; report the response if it persists.",
            )

        return DecisionResult(
            backend="openrouter",
            model=model,
            calibrated=True,
            answers=answers,
            usage=usage,
            request_id=request_id,
        )


__all__ = ["RESERVED_ROUTER_ALIAS", "OpenRouterAdapter", "require_concrete_model"]