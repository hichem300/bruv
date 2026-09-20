"""Stable process exit codes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from bruv.application import (
    ApplicationError,
    AuthenticationError,
    BackendUnavailableError,
    ConfigurationError,
    ProviderResponseError,
    RequestValidationError,
    UsageError,
)

SUCCESS = 0
USAGE_OR_VALIDATION = 2
AUTHENTICATION = 3
BACKEND_UNAVAILABLE = 4
PROVIDER_RESPONSE = 5
INTERNAL_ERROR = 70
GATE_FAILED = 10
ABSTAINED = 11

ERROR_EXIT_CODES: Mapping[type[ApplicationError], int] = {
    UsageError: USAGE_OR_VALIDATION,
    RequestValidationError: USAGE_OR_VALIDATION,
    AuthenticationError: AUTHENTICATION,
    BackendUnavailableError: BACKEND_UNAVAILABLE,
    ProviderResponseError: PROVIDER_RESPONSE,
    ConfigurationError: USAGE_OR_VALIDATION,
}


@dataclass(frozen=True, slots=True)
class CommandOutcome:
    """Final outcome of a CLI command run."""

    error: ApplicationError | None = None
    abstained: bool = False
    gate_passed: bool | None = None


def exit_code_for(outcome: CommandOutcome) -> int:
    """Map a command outcome to a stable process exit code."""
    if outcome.error is not None:
        for error_type, code in ERROR_EXIT_CODES.items():
            if isinstance(outcome.error, error_type):
                return code
        return INTERNAL_ERROR
    if outcome.abstained:
        return ABSTAINED
    if outcome.gate_passed is False:
        return GATE_FAILED
    return SUCCESS


__all__ = [
    "ABSTAINED",
    "AUTHENTICATION",
    "BACKEND_UNAVAILABLE",
    "CommandOutcome",
    "ERROR_EXIT_CODES",
    "GATE_FAILED",
    "INTERNAL_ERROR",
    "PROVIDER_RESPONSE",
    "SUCCESS",
    "USAGE_OR_VALIDATION",
    "exit_code_for",
]
