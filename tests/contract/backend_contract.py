"""Shared backend contract assertions reused across adapters."""

from __future__ import annotations

from bruv.backends.interface import DecisionBackend
from bruv.domain.requests import DecisionRequest
from bruv.domain.results import DecisionResult

__all__ = ["assert_backend_contract"]


def assert_backend_contract(backend: DecisionBackend, request: DecisionRequest) -> DecisionResult:
    """Assert behaviors every backend must satisfy for a valid request."""
    result = backend.evaluate(request)
    assert isinstance(result, DecisionResult)
    assert set(result.answers) == set(request.questions)
    assert result.backend in {"typesafe", "simple-jev"}
    assert isinstance(result.calibrated, bool)
    return result
