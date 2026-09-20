"""Threshold gates and abstain policy over canonical results."""

from __future__ import annotations

from dataclasses import dataclass

from bruv.domain.results import DecisionResult
from bruv.output.fields import FieldError, get_field


@dataclass(frozen=True, slots=True)
class GateOptions:
    """Options for a single evaluation gate."""

    field: str = "answers.*.score"
    fail_under: float | None = None
    abstain_band: tuple[float, float] | None = None


@dataclass(frozen=True, slots=True)
class GateOutcome:
    """Outcome of applying a gate to a successful result."""

    abstained: bool = False
    gate_passed: bool | None = None
    value: float | None = None


def _resolve_numeric(result: DecisionResult, field: str) -> float:
    path = field
    if path.endswith(".*.score"):
        # Pick the first answer's score when a wildcard is requested.
        for key in result.answers:
            answer = result.answers[key]
            if getattr(answer, "type", None) == "score":
                return float(get_field(result, f"answers.{key}.score"))
        raise FieldError("no score answer available for wildcard field")
    value = get_field(result, path)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise FieldError(f"field '{field}' is not numeric")
    return float(value)


def apply_gate(result: DecisionResult, options: GateOptions) -> GateOutcome:
    """Apply fail-under and abstain-band policy to a successful result."""
    if options.fail_under is None and options.abstain_band is None:
        return GateOutcome()

    value = _resolve_numeric(result, options.field)

    if options.abstain_band is not None:
        low, high = options.abstain_band
        if low <= value <= high:
            return GateOutcome(abstained=True, value=value)

    if options.fail_under is not None and value < options.fail_under:
        return GateOutcome(abstained=False, gate_passed=False, value=value)

    return GateOutcome(abstained=False, gate_passed=True, value=value)


__all__ = ["GateOptions", "GateOutcome", "apply_gate"]
