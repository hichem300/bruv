"""Threshold gates and abstain policy over canonical results."""

from __future__ import annotations

from dataclasses import dataclass

from bruv.domain.results import AbstainAnswer, DecisionResult, ScoreAnswer
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


class _AbstainResolution(Exception):
    """Internal signal: resolution found only abstain answers where a value was wanted."""


def _abstain_answer_at(result: DecisionResult, field: str) -> bool:
    """Return True when an exact ``answers.<question_id>...`` path targets an abstain answer."""
    parts = field.split(".")
    if len(parts) < 3 or parts[0] != "answers" or parts[1] == "*":
        return False
    answer = result.answers.get(parts[1])
    return isinstance(answer, AbstainAnswer)


def _resolve_numeric(result: DecisionResult, field: str) -> float:
    path = field
    if path.endswith(".*.score"):
        # Pick the first substantive score answer when a wildcard is requested.
        substantive: ScoreAnswer | None = None
        saw_abstain = False
        for answer in result.answers.values():
            if isinstance(answer, AbstainAnswer):
                saw_abstain = True
            elif isinstance(answer, ScoreAnswer):
                substantive = answer
                break
        if substantive is not None:
            return float(substantive.score)
        if saw_abstain:
            raise _AbstainResolution()
        raise FieldError("no score answer available for wildcard field")
    value = get_field(result, path)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise FieldError(f"field '{field}' is not numeric")
    return float(value)


def apply_gate(result: DecisionResult, options: GateOptions) -> GateOutcome:
    """Apply fail-under and abstain-band policy to a successful result."""
    if options.fail_under is None and options.abstain_band is None:
        return GateOutcome()

    if _abstain_answer_at(result, options.field):
        return GateOutcome(abstained=True)

    try:
        value = _resolve_numeric(result, options.field)
    except _AbstainResolution:
        return GateOutcome(abstained=True)

    if options.abstain_band is not None:
        low, high = options.abstain_band
        if low <= value <= high:
            return GateOutcome(abstained=True, value=value)

    if options.fail_under is not None and value < options.fail_under:
        return GateOutcome(abstained=False, gate_passed=False, value=value)

    return GateOutcome(abstained=False, gate_passed=True, value=value)


__all__ = ["GateOptions", "GateOutcome", "apply_gate"]
