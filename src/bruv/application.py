"""Stable application errors and decision facade.

Errors expose a machine-readable ``code``, human ``message``, ``paid_request``
state, copy-pasteable ``action``, and optional safe ``details``. Provider
secrets, request bodies, and response bodies never enter messages or details.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bruv.backends.interface import DecisionBackend
from bruv.domain.errors import ValidationIssue
from bruv.domain.requests import DecisionRequest
from bruv.domain.results import DecisionResult
from bruv.domain.validation import BackendCapabilities, validate_request


@dataclass(frozen=True, slots=True)
class ApplicationError(Exception):
    """Base for stable, serializable application errors."""

    code: str = ""
    message: str = ""
    paid_request: bool = False
    action: str = ""
    details: dict[str, Any] | None = None

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "paid_request": self.paid_request,
            "action": self.action,
        }
        if self.details is not None:
            payload["details"] = self.details
        return payload


@dataclass(frozen=True, slots=True)
class UsageError(ApplicationError):
    code: str = "usage_error"


@dataclass(frozen=True, slots=True)
class RequestValidationError(ApplicationError):
    code: str = "validation_error"
    paid_request: bool = False
    issues: tuple[ValidationIssue, ...] = field(default_factory=tuple)

    @classmethod
    def from_issues(
        cls, issues: tuple[ValidationIssue, ...], *, action: str = ""
    ) -> RequestValidationError:
        message = (
            "; ".join(issue.message for issue in issues) if issues else "request failed validation"
        )
        return cls(
            message=message,
            action=action or "Fix the reported request fields and retry.",
            issues=issues,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload["issues"] = [
            {
                "code": issue.code,
                "path": list(issue.path),
                "message": issue.message,
            }
            for issue in self.issues
        ]
        return payload


@dataclass(frozen=True, slots=True)
class AuthenticationError(ApplicationError):
    code: str = "authentication_error"


@dataclass(frozen=True, slots=True)
class BackendUnavailableError(ApplicationError):
    code: str = "backend_unavailable"


@dataclass(frozen=True, slots=True)
class ProviderResponseError(ApplicationError):
    code: str = "provider_response_error"


@dataclass(frozen=True, slots=True)
class ConfigurationError(ApplicationError):
    code: str = "configuration_error"


class DecisionFacade:
    """Validate canonical requests, then delegate to an injected backend.

    Validation performs no I/O and never calls the backend. Evaluation fails
    fast on invalid requests so no paid call is made.
    """

    def __init__(self, backend: DecisionBackend) -> None:
        self._backend = backend
        self._capabilities: BackendCapabilities = backend.capabilities

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._capabilities

    def validate(self, request: DecisionRequest) -> tuple[bool, tuple[ValidationIssue, ...]]:
        result = validate_request(request, self._capabilities)
        return result.valid, result.issues

    def evaluate(self, request: DecisionRequest) -> DecisionResult:
        valid, issues = self.validate(request)
        if not valid:
            raise RequestValidationError.from_issues(issues)
        return self._backend.evaluate(request)


__all__ = [
    "ApplicationError",
    "AuthenticationError",
    "BackendUnavailableError",
    "ConfigurationError",
    "DecisionFacade",
    "ProviderResponseError",
    "RequestValidationError",
    "UsageError",
]
