"""Pure backend-capability validation for canonical decision requests."""

from __future__ import annotations

from dataclasses import dataclass

from bruv.domain.errors import ValidationIssue
from bruv.domain.questions import ScoreQuestion
from bruv.domain.requests import DecisionRequest


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    backend: str
    question_types: frozenset[str]
    calibrated: bool
    allows_json_state: bool

    def __post_init__(self) -> None:
        if not isinstance(self.backend, str) or not self.backend.strip():
            raise ValueError("backend must be a nonblank string")


@dataclass(frozen=True, slots=True)
class ValidationResult:
    valid: bool
    issues: tuple[ValidationIssue, ...]


def _issue_sort_key(issue: ValidationIssue) -> tuple[tuple[str, ...], str, str]:
    path = tuple(f"{type(part).__name__}:{part}" for part in issue.path)
    return path, issue.code, issue.message


def _is_top_level_number_or_bool(value: object) -> bool:
    return isinstance(value, (bool, int, float))


def _is_provider_entry_scalar(value: object) -> bool:
    return isinstance(value, (bool, int, float))


def validate_request(
    request: DecisionRequest,
    capabilities: BackendCapabilities,
) -> ValidationResult:
    """Validate backend-specific constraints without performing I/O."""
    issues: list[ValidationIssue] = []

    if not capabilities.allows_json_state and not isinstance(request.state, str):
        issues.append(
            ValidationIssue(
                code="json_state_not_supported",
                path=("state",),
                message=f"Backend '{capabilities.backend}' accepts text state only.",
            )
        )

    if capabilities.backend == "simple-jev" and _is_top_level_number_or_bool(request.state):
        issues.append(
            ValidationIssue(
                code="simple_jev_scalar_state_not_supported",
                path=("state",),
                message="Simple Jev state cannot be a top-level boolean or number.",
            )
        )

    if capabilities.backend == "typesafe" and not isinstance(request.state, (str, list, dict)):
        issues.append(
            ValidationIssue(
                code="typesafe_state_type_not_supported",
                path=("state",),
                message="TypeSafe state must be text, an object, or an array.",
            )
        )

    for question_id in sorted(request.questions):
        question = request.questions[question_id]
        question_path = ("questions", question_id)

        if question.type not in capabilities.question_types:
            issues.append(
                ValidationIssue(
                    code="question_type_not_supported",
                    path=(*question_path, "type"),
                    message=(
                        f"Backend '{capabilities.backend}' does not support "
                        f"question type '{question.type}'."
                    ),
                )
            )

        if (
            capabilities.backend == "typesafe"
            and isinstance(question, ScoreQuestion)
            and len(question.criteria) > 10
        ):
            issues.append(
                ValidationIssue(
                    code="typesafe_score_level_limit",
                    path=(*question_path, "criteria"),
                    message="TypeSafe score criteria must contain between 2 and 10 levels.",
                )
            )

        if capabilities.backend == "simple-jev":
            criteria = question.criteria
            if criteria is None:
                continue
            entries = criteria.items() if isinstance(criteria, dict) else enumerate(criteria)
            for entry_key, entry in entries:
                if _is_provider_entry_scalar(entry):
                    issues.append(
                        ValidationIssue(
                            code="simple_jev_scalar_entry_not_supported",
                            path=(*question_path, "criteria", entry_key),
                            message=(
                                "Simple Jev criteria entries cannot be bare booleans or numbers."
                            ),
                        )
                    )

    ordered_issues = tuple(sorted(issues, key=_issue_sort_key))
    return ValidationResult(valid=not ordered_issues, issues=ordered_issues)
