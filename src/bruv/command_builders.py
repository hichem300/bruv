"""Pure builders that turn CLI/YAML/JSON/stdin input into canonical requests.

These builders perform no I/O against any provider. They parse bounded input,
construct canonical :class:`DecisionRequest` models, and reject incomplete
non-interactive input instead of prompting.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import yaml  # type: ignore[import-untyped]

from bruv.domain.questions import (
    JsonValue,
    Question,
    ensure_json_compatible,
)
from bruv.domain.requests import DecisionRequest

MAX_STATE_BYTES = 1_000_000  # 1 MiB ceiling for state payloads and stdin.


def build_single_question_request(
    *,
    question_id: str,
    question: Question,
    state: JsonValue,
    model: str | None,
) -> DecisionRequest:
    """Build a one-question canonical request from parsed inputs."""
    return DecisionRequest(state=state, model=model, questions={question_id: question})


def build_multi_question_request(
    *,
    questions: dict[str, Question],
    state: JsonValue,
    model: str | None,
) -> DecisionRequest:
    """Build a multi-question canonical request from parsed inputs."""
    return DecisionRequest(state=state, model=model, questions=questions)


def load_state_file(path: Path) -> JsonValue:
    """Load state from a JSON or YAML file within the byte ceiling."""
    raw = path.read_bytes()
    if len(raw) > MAX_STATE_BYTES:
        raise ValueError(f"state file exceeds {MAX_STATE_BYTES} byte ceiling")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        return cast(JsonValue, ensure_json_compatible(yaml.safe_load(raw.decode("utf-8"))))
    return cast(JsonValue, ensure_json_compatible(json.loads(raw.decode("utf-8"))))


def read_state_stdin(stream: Any) -> JsonValue:
    """Read bounded state from a text or binary stdin-like stream."""
    data = stream.read() if hasattr(stream, "read") else stream
    if isinstance(data, bytes):
        text = data.decode("utf-8")
    else:
        text = str(data)
    if len(text.encode("utf-8")) > MAX_STATE_BYTES:
        raise ValueError(f"stdin exceeds {MAX_STATE_BYTES} byte ceiling")
    stripped = text.strip()
    if not stripped:
        raise ValueError("stdin state must not be empty")
    try:
        return cast(JsonValue, ensure_json_compatible(json.loads(stripped)))
    except json.JSONDecodeError:
        return cast(JsonValue, ensure_json_compatible(yaml.safe_load(stripped)))


def load_eval_request(path: Path, state_override: JsonValue | None) -> DecisionRequest:
    """Load a JSON or YAML eval file and return a canonical request.

    The file must contain at least ``questions`` and ``state`` unless
    ``state_override`` is supplied. Unknown fields are rejected by the model.
    """
    raw = path.read_bytes()
    if len(raw) > MAX_STATE_BYTES:
        raise ValueError(f"eval file exceeds {MAX_STATE_BYTES} byte ceiling")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(raw.decode("utf-8"))
    else:
        data = json.loads(raw.decode("utf-8"))

    if not isinstance(data, dict):
        raise ValueError("eval file must contain a mapping")

    payload: dict[str, Any] = dict(data)
    if state_override is not None:
        payload["state"] = state_override
    if "state" not in payload:
        raise ValueError("eval file is missing required 'state' (or pass --state)")
    if "questions" not in payload:
        raise ValueError("eval file is missing required 'questions'")

    return DecisionRequest.model_validate(payload)


def parse_options(option_strings: list[str]) -> dict[str, JsonValue | None]:
    """Parse repeated ``--option key[=description]`` values into a criteria map."""
    criteria: dict[str, JsonValue | None] = {}
    for item in option_strings:
        if "=" in item:
            key, description = item.split("=", 1)
            criteria[key.strip()] = description
        else:
            criteria[item.strip()] = None
    return criteria


def parse_levels(level_strings: list[str]) -> list[JsonValue]:
    """Parse repeated ordered ``--level`` values into score criteria."""
    return [level for level in level_strings]


__all__ = [
    "MAX_STATE_BYTES",
    "build_multi_question_request",
    "build_single_question_request",
    "load_eval_request",
    "load_state_file",
    "parse_levels",
    "parse_options",
    "read_state_stdin",
]
