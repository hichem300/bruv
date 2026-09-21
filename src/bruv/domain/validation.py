"""Pure backend-capability validation for canonical decision requests."""

from __future__ import annotations

import json
from dataclasses import dataclass

from bruv.domain.errors import ValidationIssue
from bruv.domain.questions import ChoiceQuestion, Question, ScoreQuestion
from bruv.domain.requests import DecisionRequest


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    backend: str
    question_types: frozenset[str]
    calibrated: bool
    allows_json_state: bool
    explicit_abstention: bool = False
    supported_total_candidates: frozenset[int] | None = None
    reserved_input_markers: tuple[str, ...] = ()
    reserved_answer_ids: frozenset[str] = frozenset()

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


def _serialize_json_state(state: object) -> str:
    """Serialize non-string state deterministically; string state passes through."""
    if isinstance(state, str):
        return state
    return json.dumps(state, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _question_text_parts(question: Question, state: object) -> list[str]:
    """Collect prompt-derived text for a question: instructions, state, and descriptions."""
    parts = [question.instructions, _serialize_json_state(state)]
    if isinstance(question, ChoiceQuestion):
        parts.extend(str(value) for value in question.criteria.values())
    elif isinstance(question, ScoreQuestion):
        parts.extend(
            json.dumps(level, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            for level in question.criteria
        )
    return parts


def _validate_reserved_input_markers(
    question_id: str,
    question: Question,
    state: object,
    markers: tuple[str, ...],
    issues: list[ValidationIssue],
) -> None:
    combined = "\n".join(_question_text_parts(question, state))
    for marker in markers:
        if marker and marker in combined:
            issues.append(
                ValidationIssue(
                    code="reserved_input_marker",
                    path=("questions", question_id),
                    message=(
                        f"Reserved input marker {marker!r} must not appear in "
                        "question-derived prompt text."
                    ),
                )
            )


def _validate_candidate_totals(
    question_id: str,
    question: Question,
    capabilities: BackendCapabilities,
    issues: list[ValidationIssue],
) -> bool:
    """Return True when candidate totals are supported; False otherwise."""
    if capabilities.supported_total_candidates is None:
        return True
    if question.type == "noul":
        total = 3
    elif question.criteria is None:
        return True
    else:
        total = len(question.criteria) + (1 if capabilities.explicit_abstention else 0)
    if total not in capabilities.supported_total_candidates:
        issues.append(
            ValidationIssue(
                code="total_candidates_not_supported",
                path=("questions", question_id, "criteria"),
                message=(
                    f"Backend '{capabilities.backend}' does not support "
                    f"{total} candidate answers for this question."
                ),
            )
        )
        return False
    return True


def _validate_reserved_answer_ids(
    question_id: str,
    question: Question,
    reserved_answer_ids: frozenset[str],
    issues: list[ValidationIssue],
) -> None:
    """Choice option IDs are the only caller-declared IDs that can collide."""
    if not isinstance(question, ChoiceQuestion):
        return
    for option_id in sorted(question.criteria):
        if option_id in reserved_answer_ids:
            issues.append(
                ValidationIssue(
                    code="reserved_answer_id_collision",
                    path=("questions", question_id, "criteria", option_id),
                    message=f"Choice option id {option_id!r} collides with a reserved answer id.",
                )
            )


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

        _validate_candidate_totals(question_id, question, capabilities, issues)
        _validate_reserved_input_markers(
            question_id,
            question,
            request.state,
            capabilities.reserved_input_markers,
            issues,
        )
        _validate_reserved_answer_ids(
            question_id,
            question,
            capabilities.reserved_answer_ids,
            issues,
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
