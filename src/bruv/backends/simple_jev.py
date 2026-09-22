"""Simple Jev HTTP adapter.

Provider request/response shapes stay inside this module. The adapter maps the
canonical :class:`DecisionRequest` to the upstream ``POST /v1/classifier``
schema and the response back to a canonical :class:`DecisionResult`. Simple Jev
is explicitly uncalibrated, so every canonical result reports
``calibrated=False``.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import httpx
from pydantic import TypeAdapter, ValidationError

from bruv.application import BackendUnavailableError, ProviderResponseError
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

_CAPABILITIES = BackendCapabilities(
    backend="simple-jev",
    question_types=frozenset({"noul", "choice", "score"}),
    calibrated=False,
    allows_json_state=True,
)

_USAGE_ADAPTER: TypeAdapter[Usage] = TypeAdapter(Usage)


def _question_payload(question: Any) -> dict[str, Any]:
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
        "messages": None,
        "questions": {
            question_id: _question_payload(question)
            for question_id, question in request.questions.items()
        },
        "options": {"raw_logits": False},
        "tools": None,
        "mm_processor_kwargs": None,
        "media_io_kwargs": None,
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


class SimpleJevAdapter(DecisionBackend):
    """Adapter for the upstream Simple Jev ``/v1/classifier`` endpoint."""

    capabilities = _CAPABILITIES

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        client: httpx.Client,
        timeout_seconds: float = 30.0,
        warning_sink: Callable[[str], None] | None = None,
        warning_root: str | os.PathLike[str] | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive and finite")
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._client = client
        self._timeout_seconds = float(timeout_seconds)
        self._warning_sink = warning_sink
        self._warning_root = warning_root

    def _maybe_emit_warning(self, request: DecisionRequest) -> None:
        """Once-only advisory warning; never changes evaluation results."""
        if self._warning_sink is None:
            return
        if not any(isinstance(q, NoulQuestion) for q in request.questions.values()):
            return
        root = self._warning_root
        if root is None:
            from bruv.onboarding.simple_jev_runtime import SimpleJevPaths

            root = SimpleJevPaths.default().root
        from bruv.onboarding.simple_jev_warning import emit_first_use_warning

        emit_first_use_warning(
            model=self._model,
            mode="noul",
            root=root,
            sink=self._warning_sink,
        )

    def evaluate(self, request: DecisionRequest) -> DecisionResult:
        self._maybe_emit_warning(request)
        payload = _to_payload(request, request.model or self._model)
        try:
            response = self._client.post(
                f"{self._base_url}/v1/classifier",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self._timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            raise BackendUnavailableError(
                message="Simple Jev request timed out before any response.",
                paid_request=False,
                action="Check the Simple Jev endpoint and retry.",
            ) from exc
        except httpx.NetworkError as exc:
            raise BackendUnavailableError(
                message="Simple Jev endpoint could not be reached.",
                paid_request=False,
                action="Check that Simple Jev is running and reachable.",
            ) from exc

        self._raise_for_status(response)

        try:
            body = response.json()
        except ValueError as exc:
            raise ProviderResponseError(
                message="Simple Jev response was not valid JSON.",
                paid_request=True,
                action="Retry the request; report the response if it persists.",
            ) from exc

        return self._to_result(body, request)

    def _raise_for_status(self, response: httpx.Response) -> None:
        status = response.status_code
        if status == 422:
            raise ProviderResponseError(
                message="Simple Jev rejected the request as invalid.",
                paid_request=False,
                action="Fix the request and retry.",
            )
        if status == 429:
            raise BackendUnavailableError(
                message="Simple Jev scoring queue is full.",
                paid_request=False,
                action="Wait briefly and retry.",
            )
        if status >= 500:
            raise ProviderResponseError(
                message="Simple Jev returned a server error.",
                paid_request=True,
                action="Retry the request; report the error if it persists.",
            )
        if status >= 400:
            raise ProviderResponseError(
                message=f"Simple Jev returned status {status}.",
                paid_request=False,
                action="Fix the request and retry.",
            )

    def _to_result(self, body: Any, request: DecisionRequest) -> DecisionResult:
        if not isinstance(body, dict):
            raise ProviderResponseError(
                message="Simple Jev response must be a JSON object.",
                paid_request=True,
                action="Retry the request; report the response if it persists.",
            )
        answers_payload = body.get("answers")
        if not isinstance(answers_payload, dict):
            raise ProviderResponseError(
                message="Simple Jev response missing answers object.",
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
                    message="Simple Jev usage did not match canonical schema.",
                    paid_request=True,
                    action="Retry the request; report the response if it persists.",
                ) from exc

        model = body.get("model")
        if not isinstance(model, str) or not model.strip():
            raise ProviderResponseError(
                message="Simple Jev response missing model identifier.",
                paid_request=True,
                action="Retry the request; report the response if it persists.",
            )

        return DecisionResult(
            backend="simple-jev",
            model=model,
            calibrated=False,
            answers=answers,
            usage=usage,
        )


__all__ = ["SimpleJevAdapter"]
