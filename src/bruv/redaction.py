"""Secret-safe dry-run and message redaction."""

from __future__ import annotations

from typing import Any

from bruv.domain.requests import DecisionRequest
from bruv.domain.validation import BackendCapabilities

# Request fields that must never appear in error messages or diagnostics.
_SENSITIVE_HINTS = ("key", "token", "secret", "password", "authorization")


def redacted_dry_run(request: DecisionRequest, capabilities: BackendCapabilities) -> dict[str, Any]:
    """Return a safe dry-run preview that invokes no adapter transport."""
    return {
        "dry_run": True,
        "backend": capabilities.backend,
        "calibrated": capabilities.calibrated,
        "request": request.model_dump(mode="json"),
    }


def redact_text(value: str) -> str:
    """Replace anything that looks like a secret-bearing fragment."""
    if any(hint in value.lower() for hint in _SENSITIVE_HINTS):
        return "[redacted]"
    return value


__all__ = ["redact_text", "redacted_dry_run"]
