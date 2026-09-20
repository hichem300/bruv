"""Backend port shared by adapters and the application facade."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from bruv.domain.requests import DecisionRequest
from bruv.domain.results import DecisionResult
from bruv.domain.validation import BackendCapabilities


@runtime_checkable
class DecisionBackend(Protocol):
    """One-method port every provider adapter implements.

    Adapters own provider request/response shapes. Nothing outside an adapter
    imports provider SDK or HTTP types.
    """

    capabilities: BackendCapabilities

    def evaluate(self, request: DecisionRequest) -> DecisionResult: ...


__all__ = ["DecisionBackend"]
