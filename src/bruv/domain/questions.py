"""Provider-independent question models and JSON value types."""

from __future__ import annotations

import math
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator
from pydantic import JsonValue as PydanticJsonValue

JsonValue: TypeAlias = PydanticJsonValue


def _nonblank(value: str) -> str:
    if not value.strip():
        raise ValueError("must not be blank")
    return value


NonBlankString: TypeAlias = Annotated[str, AfterValidator(_nonblank)]


def ensure_json_compatible(value: Any, *, _seen: set[int] | None = None) -> Any:
    """Reject values JSON cannot represent, including cycles and non-finite floats."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
        return value

    seen = set() if _seen is None else _seen
    if isinstance(value, list):
        identity = id(value)
        if identity in seen:
            raise ValueError("JSON values must not contain cycles")
        seen.add(identity)
        try:
            for item in value:
                ensure_json_compatible(item, _seen=seen)
        finally:
            seen.remove(identity)
        return value

    if isinstance(value, dict):
        identity = id(value)
        if identity in seen:
            raise ValueError("JSON values must not contain cycles")
        seen.add(identity)
        try:
            for key, item in value.items():
                if not isinstance(key, str):
                    raise ValueError("JSON object keys must be strings")
                ensure_json_compatible(item, _seen=seen)
        finally:
            seen.remove(identity)
        return value

    raise ValueError(f"{type(value).__name__} is not JSON compatible")


class CanonicalModel(BaseModel):
    """Strict base for canonical wire-safe models."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class NoulQuestion(CanonicalModel):
    type: Literal["noul"] = "noul"
    instructions: NonBlankString
    criteria: dict[Literal["true", "false"], JsonValue] | None = None

    @field_validator("criteria", mode="before")
    @classmethod
    def validate_criteria_json(cls, value: Any) -> Any:
        return ensure_json_compatible(value)


class ChoiceQuestion(CanonicalModel):
    type: Literal["choice"] = "choice"
    instructions: NonBlankString
    criteria: Annotated[dict[NonBlankString, JsonValue], Field(min_length=2, max_length=50)]

    @field_validator("criteria", mode="before")
    @classmethod
    def validate_criteria_json(cls, value: Any) -> Any:
        return ensure_json_compatible(value)


class ScoreQuestion(CanonicalModel):
    type: Literal["score"] = "score"
    instructions: NonBlankString
    criteria: Annotated[list[JsonValue], Field(min_length=2, max_length=50)]

    @field_validator("criteria", mode="before")
    @classmethod
    def validate_criteria_json(cls, value: Any) -> Any:
        return ensure_json_compatible(value)


Question: TypeAlias = Annotated[
    NoulQuestion | ChoiceQuestion | ScoreQuestion,
    Field(discriminator="type"),
]
