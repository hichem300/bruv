"""Canonical decision request."""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import Field, field_validator

from bruv.domain.questions import (
    CanonicalModel,
    JsonValue,
    NonBlankString,
    Question,
    ensure_json_compatible,
)


class DecisionRequest(CanonicalModel):
    state: JsonValue
    model: NonBlankString | None = None
    questions: Annotated[
        dict[NonBlankString, Question],
        Field(min_length=1, max_length=256),
    ]

    @field_validator("state", mode="before")
    @classmethod
    def validate_state_json(cls, value: Any) -> Any:
        return ensure_json_compatible(value)
