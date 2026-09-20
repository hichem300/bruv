"""DecisionFacade: no-call validation, paid_request preservation, error mapping."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from bruv.application import (
    AuthenticationError,
    DecisionFacade,
    ProviderResponseError,
    RequestValidationError,
)
from bruv.backends.interface import DecisionBackend
from bruv.domain.questions import ChoiceQuestion
from bruv.domain.requests import DecisionRequest
from bruv.domain.results import ChoiceAnswer, DecisionResult
from bruv.domain.validation import BackendCapabilities

SIMPLE_CAPABILITIES = BackendCapabilities(
    backend="simple-jev",
    question_types=frozenset({"noul", "choice", "score"}),
    calibrated=False,
    allows_json_state=True,
)


class RecordingBackend:
    """Records calls and never performs real evaluation."""

    capabilities = SIMPLE_CAPABILITIES

    def __init__(self, *, raise_with: BaseException | None = None) -> None:
        self.calls = 0
        self.last_request: DecisionRequest | None = None
        self._raise_with = raise_with

    def evaluate(self, request: DecisionRequest) -> DecisionResult:
        self.calls += 1
        self.last_request = request
        if self._raise_with is not None:
            raise self._raise_with
        return DecisionResult(
            backend="simple-jev",
            model="Qwen/Qwen3.5-0.8B",
            calibrated=False,
            answers={
                "route": ChoiceAnswer(
                    type="choice",
                    choice="billing",
                    confidence=0.9,
                    probabilities={"billing": 0.9, "other": 0.1},
                )
            },
        )


def _valid_request() -> DecisionRequest:
    return DecisionRequest(
        state="help",
        questions={
            "route": ChoiceQuestion(
                instructions="Route?", criteria={"sales": None, "billing": None}
            )
        },
    )


def _invalid_request() -> DecisionRequest:
    """Choice with one option violates canonical 2-50 bound."""
    with pytest.raises(ValidationError):
        DecisionRequest(
            state="help",
            questions={"route": ChoiceQuestion(instructions="Route?", criteria={"sales": None})},
        )
    # Build an unsupported-type invalid request instead: score with too many
    # levels is valid canonically but exceeds the Simple Jev scalar-entry rule
    # only for scalar entries. Use a request valid canonically but invalid for
    # Simple Jev by giving a top-level numeric state.
    return DecisionRequest(
        state=5,
        questions={
            "route": ChoiceQuestion(
                instructions="Route?", criteria={"sales": None, "billing": None}
            )
        },
    )


def test_validate_returns_issues_without_calling_backend() -> None:
    backend = RecordingBackend()
    facade = DecisionFacade(backend)
    valid, issues = facade.validate(_valid_request())
    assert valid is True
    assert issues == ()
    assert backend.calls == 0


def test_invalid_request_never_calls_backend() -> None:
    backend = RecordingBackend()
    facade = DecisionFacade(backend)
    request = _invalid_request()
    valid, issues = facade.validate(request)
    assert valid is False
    assert backend.calls == 0
    assert any(issue.code == "simple_jev_scalar_state_not_supported" for issue in issues)


def test_evaluate_invalid_request_raises_before_backend_call() -> None:
    backend = RecordingBackend()
    facade = DecisionFacade(backend)
    with pytest.raises(RequestValidationError) as exc_info:
        facade.evaluate(_invalid_request())
    assert backend.calls == 0
    assert exc_info.value.paid_request is False
    assert exc_info.value.issues  # type: ignore[attr-defined]
    assert exc_info.value.action


def test_evaluate_valid_request_calls_backend_once() -> None:
    backend = RecordingBackend()
    facade = DecisionFacade(backend)
    result = facade.evaluate(_valid_request())
    assert backend.calls == 1
    assert result.backend == "simple-jev"
    assert result.calibrated is False


def test_authentication_error_preserves_paid_request_false() -> None:
    backend = RecordingBackend(
        raise_with=AuthenticationError(message="no key", paid_request=False, action="set key")
    )
    facade = DecisionFacade(backend)
    with pytest.raises(AuthenticationError) as exc_info:
        facade.evaluate(_valid_request())
    assert exc_info.value.paid_request is False
    assert backend.calls == 1


def test_provider_response_error_marks_paid_request_true() -> None:
    backend = RecordingBackend(
        raise_with=ProviderResponseError(message="malformed", paid_request=True, action="retry")
    )
    facade = DecisionFacade(backend)
    with pytest.raises(ProviderResponseError) as exc_info:
        facade.evaluate(_valid_request())
    assert exc_info.value.paid_request is True


def test_unexpected_exception_is_not_misreported() -> None:
    backend = RecordingBackend(raise_with=RuntimeError("boom"))
    facade = DecisionFacade(backend)
    # Unexpected errors propagate unchanged; they are not translated into
    # authentication or validation errors.
    with pytest.raises(RuntimeError, match="boom"):
        facade.evaluate(_valid_request())


def test_request_validation_error_from_issues_has_stable_fields() -> None:
    issues = RequestValidationError.from_issues(
        (
            __import__(
                "bruv.domain.errors",
                fromlist=["ValidationIssue"],
            ).ValidationIssue(code="x", path=("questions", "route"), message="bad"),
        )
    )
    assert issues.code == "validation_error"
    assert issues.paid_request is False


def test_facade_satisfies_runtime_backend_protocol() -> None:
    backend = RecordingBackend()
    assert isinstance(backend, DecisionBackend)
