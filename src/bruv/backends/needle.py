"""Needle 3 local decision backend.

The optional ``needle`` package stays behind :class:`CactusNeedleRuntime` and is
imported only when that runtime is constructed. Tests and other integrations can
inject :class:`NeedleRuntime` without installing or loading model weights.
"""

from __future__ import annotations

import importlib
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from threading import RLock
from typing import Any, Protocol, runtime_checkable

from pydantic import ValidationError

from bruv.application import (
    ApplicationError,
    BackendUnavailableError,
    ConfigurationError,
    ProviderResponseError,
)
from bruv.backends.interface import DecisionBackend
from bruv.domain.questions import ChoiceQuestion, JsonValue, NoulQuestion, ScoreQuestion
from bruv.domain.requests import DecisionRequest
from bruv.domain.results import Answer, ChoiceAnswer, DecisionResult, NoulAnswer, ScoreAnswer
from bruv.domain.validation import BackendCapabilities

_DEFAULT_MODEL = "Cactus-Compute/needle3"
_TOOL_NAME = "select_answer"
_INSTALL_ACTION = "Install Needle support with: pip install 'bruv[needle] @ git+https://github.com/hichem300/bruv.git'"
_PROVIDER_ACTION = "Clarify the state and retry, or use another backend."
_NATIVE_RUNTIME_LOCK = RLock()

_CAPABILITIES = BackendCapabilities(
    backend="needle",
    question_types=frozenset({"noul", "choice", "score"}),
    calibrated=True,
    allows_json_state=True,
)


@dataclass(frozen=True, slots=True)
class NeedleSelection:
    """One constrained Needle selection.

    ``value=None`` represents an explicit empty-call refusal. Needle may omit
    confidence for that path, so ``confidence`` is optional at this narrow raw
    runtime boundary. A suppressed response also carries no usable value.
    """

    value: str | bool | None
    confidence: float | None
    suppressed: bool = False


@runtime_checkable
class NeedleRuntime(Protocol):
    """Small injectable port for one constrained local classification."""

    def classify(
        self,
        *,
        text: str,
        schema: dict[str, object],
        description: str,
    ) -> NeedleSelection: ...


def _provider_response_error(message: str) -> ProviderResponseError:
    return ProviderResponseError(
        message=message,
        paid_request=False,
        action=_PROVIDER_ACTION,
    )


def _unsupported_model_error() -> ConfigurationError:
    return ConfigurationError(
        message="Needle supports only the base Needle 3 model.",
        paid_request=False,
        action=f"Use model '{_DEFAULT_MODEL}' or omit model selection.",
    )


class CactusNeedleRuntime(NeedleRuntime):
    """Runtime backed by cactus-needle's raw ``Needle.complete`` API.

    cactus-needle binds schemas and conversation state to an agent. Each call
    therefore gets a short-lived agent, serialized process-wide around native
    runtime access, and cleaned before its final reference is released.
    """

    def __init__(self) -> None:
        os.environ.setdefault("NEEDLE_TELEMETRY", "0")
        try:
            self._needle = importlib.import_module("needle")
        except ModuleNotFoundError as exc:
            if exc.name == "needle":
                raise ConfigurationError(
                    message="Needle support is not installed.",
                    paid_request=False,
                    action=_INSTALL_ACTION,
                ) from None
            raise BackendUnavailableError(
                message="Needle runtime dependencies could not be loaded.",
                paid_request=False,
                action="Check Needle platform support and reinstall Needle support.",
            ) from None
        except Exception:
            raise BackendUnavailableError(
                message="Needle runtime could not be loaded on this platform.",
                paid_request=False,
                action="Check Needle platform support and reinstall Needle support.",
            ) from None

    def classify(
        self,
        *,
        text: str,
        schema: dict[str, object],
        description: str,
    ) -> NeedleSelection:
        tool: dict[str, object] | None = {
            "name": _TOOL_NAME,
            "description": description,
            "parameters": schema,
        }
        agent: Any = None

        with _NATIVE_RUNTIME_LOCK:
            try:
                try:
                    agent = self._needle.Needle(tools=[tool], auto_date=False)
                except Exception:
                    raise BackendUnavailableError(
                        message="Needle runtime could not initialize or load its local model.",
                        paid_request=False,
                        action="Check platform support, local cache access, and retry.",
                    ) from None

                try:
                    agent.reset()
                    response = agent.complete(text)
                except Exception:
                    raise BackendUnavailableError(
                        message="Needle local inference could not complete.",
                        paid_request=False,
                        action="Check platform support, local cache access, and retry.",
                    ) from None

                return self._parse_response(response)
            finally:
                if agent is not None:
                    try:
                        agent.reset()
                    except Exception:  # noqa: S110 - cleanup must not mask primary failures
                        pass
                    try:
                        close = getattr(agent, "close", None)
                    except Exception:
                        close = None
                    if callable(close):
                        try:
                            close()
                        except Exception:  # noqa: S110 - cleanup must not mask primary failures
                            pass
                agent = None
                tool = None

    @staticmethod
    def _parse_response(response: Any) -> NeedleSelection:
        if not isinstance(response, Mapping):
            raise _provider_response_error("Needle returned a malformed response.")

        success = response.get("success")
        if success is False:
            raise BackendUnavailableError(
                message="Needle local inference reported a runtime failure.",
                paid_request=False,
                action="Check platform support, local cache access, and retry.",
            )
        if success is not True:
            raise _provider_response_error("Needle response omitted its success status.")

        response_type = response.get("type")
        if response_type not in {"call", "respond"}:
            raise _provider_response_error("Needle response had an invalid turn type.")

        calls = response.get("function_calls")
        suppressed_calls = response.get("suppressed_calls")
        if not isinstance(calls, list) or not isinstance(suppressed_calls, list):
            raise _provider_response_error("Needle response omitted its call lists.")
        if calls and suppressed_calls:
            raise _provider_response_error("Needle response contained contradictory call lists.")
        if (calls or suppressed_calls) and response_type != "call":
            raise _provider_response_error("Needle response turn type contradicted its calls.")

        confidence = response.get("confidence")
        if suppressed_calls:
            return NeedleSelection(value=None, confidence=confidence, suppressed=True)
        if not calls:
            return NeedleSelection(value=None, confidence=confidence)
        if len(calls) != 1:
            raise _provider_response_error("Needle returned an unexpected number of selections.")

        call = calls[0]
        if not isinstance(call, Mapping) or call.get("name") != _TOOL_NAME:
            raise _provider_response_error("Needle returned an unexpected selection call.")
        arguments = call.get("arguments")
        if not isinstance(arguments, Mapping) or set(arguments) != {"value"}:
            raise _provider_response_error("Needle selection arguments were malformed.")

        return NeedleSelection(value=arguments["value"], confidence=confidence)


def _json_text(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _question_contract(
    question: NoulQuestion | ChoiceQuestion | ScoreQuestion,
) -> tuple[dict[str, object], str, dict[str, JsonValue] | None]:
    if isinstance(question, NoulQuestion):
        criteria_text = (
            "No separate true/false descriptions were supplied."
            if question.criteria is None
            else f"True/false criteria (JSON): {_json_text(dict(question.criteria))}"
        )
        return (
            {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "boolean",
                        "description": "Boolean decision selected from the supplied state.",
                    }
                },
                "required": ["value"],
                "additionalProperties": False,
            },
            f"{question.instructions}\nReturn one boolean decision. {criteria_text}",
            None,
        )

    if isinstance(question, ChoiceQuestion):
        criteria = dict(question.criteria)
        return (
            {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "string",
                        "enum": list(criteria),
                        "description": "Exact criterion key selected from the supplied state.",
                    }
                },
                "required": ["value"],
                "additionalProperties": False,
            },
            (
                f"{question.instructions}\nReturn exactly one criterion key. "
                f"Criteria by exact key (JSON): {_json_text(criteria)}"
            ),
            None,
        )

    if isinstance(question, ScoreQuestion):
        legend: dict[str, JsonValue] = {
            str(index): level for index, level in enumerate(question.criteria)
        }
        return (
            {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "string",
                        "enum": list(legend),
                        "description": "Exact level index selected from the supplied state.",
                    }
                },
                "required": ["value"],
                "additionalProperties": False,
            },
            (
                f"{question.instructions}\nReturn exactly one level index. "
                f"Indexed levels (JSON): {_json_text(legend)}"
            ),
            legend,
        )

    raise TypeError(f"unsupported question type: {type(question).__name__}")


def _validated_confidence(selection: NeedleSelection) -> float:
    confidence = selection.confidence
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
        or not 0.0 <= confidence <= 1.0
    ):
        raise _provider_response_error("Needle returned invalid selection confidence.")
    return float(confidence)


def _to_answer(
    question: NoulQuestion | ChoiceQuestion | ScoreQuestion,
    selection: NeedleSelection,
    legend: dict[str, JsonValue] | None,
) -> NoulAnswer | ChoiceAnswer | ScoreAnswer:
    if type(selection.suppressed) is not bool:
        raise _provider_response_error("Needle returned invalid suppression status.")
    if selection.suppressed:
        raise _provider_response_error("Needle withheld the answer.")
    if selection.value is None:
        raise _provider_response_error("Needle declined to answer the question.")

    confidence = _validated_confidence(selection)
    try:
        if isinstance(question, NoulQuestion):
            if type(selection.value) is not bool:
                raise ValueError("expected boolean selection")
            return NoulAnswer(value=selection.value, confidence=confidence)

        if isinstance(question, ChoiceQuestion):
            if not isinstance(selection.value, str) or selection.value not in question.criteria:
                raise ValueError("unknown choice selection")
            return ChoiceAnswer(choice=selection.value, confidence=confidence)

        if isinstance(question, ScoreQuestion):
            if (
                legend is None
                or not isinstance(selection.value, str)
                or selection.value not in legend
            ):
                raise ValueError("invalid score selection")
            return ScoreAnswer(
                score=float(int(selection.value)),
                confidence=confidence,
                legend=legend,
            )
    except (ValidationError, ValueError, TypeError):
        raise _provider_response_error(
            "Needle selection did not match the question schema."
        ) from None

    raise TypeError(f"unsupported question type: {type(question).__name__}")


class NeedleAdapter(DecisionBackend):
    """Map canonical bruv decisions to one local Needle call per question."""

    capabilities = _CAPABILITIES

    def __init__(
        self,
        runtime: NeedleRuntime,
        model: str = _DEFAULT_MODEL,
    ) -> None:
        if model != _DEFAULT_MODEL:
            raise _unsupported_model_error()
        self._runtime = runtime

    def evaluate(self, request: DecisionRequest) -> DecisionResult:
        if request.model is not None and request.model != _DEFAULT_MODEL:
            raise _unsupported_model_error()

        text = (
            request.state
            if isinstance(request.state, str)
            else json.dumps(
                request.state,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
        )
        answers: dict[str, Answer] = {}

        for question_id, question in request.questions.items():
            schema, description, legend = _question_contract(question)
            try:
                selection = self._runtime.classify(
                    text=text,
                    schema=schema,
                    description=description,
                )
            except ApplicationError:
                raise
            except Exception:
                raise BackendUnavailableError(
                    message="Needle runtime failed before returning a selection.",
                    paid_request=False,
                    action="Check platform support, local cache access, and retry.",
                ) from None
            if not isinstance(selection, NeedleSelection):
                raise _provider_response_error("Needle runtime returned an invalid selection.")
            answers[question_id] = _to_answer(question, selection, legend)

        return DecisionResult(
            backend="needle",
            model=_DEFAULT_MODEL,
            calibrated=True,
            answers=answers,
            usage=None,
            provider_metadata={
                "confidence_scope": "whole_response",
                "probabilities_available": False,
            },
        )


__all__ = [
    "CactusNeedleRuntime",
    "NeedleAdapter",
    "NeedleRuntime",
    "NeedleSelection",
]
