"""Human-readable terminal output.

Color is opt-in via Rich and disabled by ``--no-color``. Diagnostics and
progress go to stderr only. Meaning never depends on color.
"""

from __future__ import annotations

from typing import Any

from bruv.application import ApplicationError
from bruv.domain.results import AbstainAnswer, ChoiceAnswer, DecisionResult, NoulAnswer, ScoreAnswer

_CALIBRATION_LABEL = {
    True: "calibrated",
    False: "uncalibrated",
}


def render_human_result(result: DecisionResult) -> str:
    """Render a canonical result for humans, always disclosing calibration."""
    lines: list[str] = [
        f"backend: {result.backend}",
        f"model: {result.model}",
        f"calibration: {_CALIBRATION_LABEL[result.calibrated]}",
    ]
    for question_id, answer in result.answers.items():
        if isinstance(answer, NoulAnswer):
            if answer.noul is not None:
                lines.append(f"{question_id}: {answer.noul:.3f}")
            else:
                lines.append(
                    f"{question_id}: {str(answer.value).lower()} "
                    f"(confidence: {answer.confidence:.3f})"
                )
        elif isinstance(answer, ChoiceAnswer):
            suffix = (
                ""
                if answer.probabilities is not None
                else f" (confidence: {answer.confidence:.3f})"
            )
            lines.append(f"{question_id}: {answer.choice}{suffix}")
        elif isinstance(answer, ScoreAnswer):
            suffix = (
                ""
                if answer.probabilities is not None
                else f" (confidence: {answer.confidence:.3f})"
            )
            lines.append(f"{question_id}: {answer.score:.3f}{suffix}")
        elif isinstance(answer, AbstainAnswer):
            if answer.confidence is not None:
                lines.append(
                    f"{question_id}: abstained ({answer.reason}, p={answer.confidence:.3f})"
                )
            else:
                lines.append(f"{question_id}: abstained ({answer.reason})")
    return "\n".join(lines) + "\n"


def render_human_error(error: ApplicationError) -> str:
    """Render a stable human-readable error line."""
    return f"error: {error.code}: {error.message}\n"


def render_diagnostic(message: str) -> str:
    """Render a diagnostic line intended for stderr."""
    return f"{message}\n"


def render_field(value: Any) -> str:
    """Render a single extracted field value for humans."""
    if isinstance(value, float):
        return f"{value:.3f}"
    if value is None:
        return "null"
    return str(value)


__all__ = [
    "render_diagnostic",
    "render_field",
    "render_human_error",
    "render_human_result",
]
