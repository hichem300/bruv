"""Safe field extraction over canonical results.

Supports dot paths such as ``answers.route.choice``. Unknown or private fields
are rejected so agents cannot scrape undocumented internals.
"""

from __future__ import annotations

from typing import Any

from bruv.domain.results import DecisionResult

_TOP_LEVEL_FIELDS = {
    "backend",
    "model",
    "calibrated",
    "answers",
    "usage",
    "latency_ms",
    "request_id",
}

_ANSWER_FIELDS = {
    "type",
    "choice",
    "noul",
    "score",
    "confidence",
    "probabilities",
    "legend",
    "reason",
    "source_question_type",
}

_USAGE_FIELDS = {"input_tokens", "output_tokens"}


class FieldError(ValueError):
    """Raised when a field path is unknown, private, or malformed."""


def get_field(result: DecisionResult, path: str) -> Any:
    """Return the value at a dot path, rejecting private/unknown fields."""
    parts = path.split(".") if path else []
    if not parts:
        raise FieldError("field path must not be empty")

    current: Any = result
    for index, part in enumerate(parts):
        current = _step(current, part, parts[: index + 1])

    return _jsonable(current)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


def _step(current: Any, part: str, walked: list[str]) -> Any:
    path_so_far = ".".join(walked)
    if isinstance(current, DecisionResult):
        if part not in _TOP_LEVEL_FIELDS:
            raise FieldError(f"unknown field '{part}' in '{path_so_far}'")
        return getattr(current, part)

    if walked[0] == "answers":
        return _step_into_answer(current, part, walked)

    if walked[0] == "usage":
        if not isinstance(current, object) or not hasattr(current, part):
            raise FieldError(f"unknown usage field '{part}' in '{path_so_far}'")
        if part not in _USAGE_FIELDS:
            raise FieldError(f"unknown usage field '{part}' in '{path_so_far}'")
        return getattr(current, part)

    raise FieldError(f"cannot index into '{path_so_far}'")


def _step_into_answer(current: Any, part: str, walked: list[str]) -> Any:
    path_so_far = ".".join(walked)
    if len(walked) == 2:
        if not isinstance(current, dict):
            raise FieldError(f"no answer named '{part}' in '{path_so_far}'")
        if part not in current:
            raise FieldError(f"no answer named '{part}' in '{path_so_far}'")
        return current[part]

    if len(walked) == 3:
        if part not in _ANSWER_FIELDS:
            raise FieldError(f"unknown answer field '{part}' in '{path_so_far}'")
        if not hasattr(current, part):
            raise FieldError(f"answer has no field '{part}' in '{path_so_far}'")
        return getattr(current, part)

    if len(walked) == 4 and walked[2] in {"probabilities", "legend"}:
        if not isinstance(current, dict):
            raise FieldError(f"'{walked[2]}' is not a mapping at '{path_so_far}'")
        if part not in current:
            raise FieldError(f"no key '{part}' in '{path_so_far}'")
        return current[part]

    raise FieldError(f"answer field path too deep at '{path_so_far}'")


__all__ = ["FieldError", "get_field"]
