"""Provider-independent question models and JSON value types."""

from __future__ import annotations

import math
from typing import Annotated, Any, Literal, Self, TypeAlias, TypeVar

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic import JsonValue as PydanticJsonValue

JsonValue: TypeAlias = PydanticJsonValue

MAX_JSON_DEPTH = 64
MAX_JSON_NODES = 10_000

_K = TypeVar("_K")
_V = TypeVar("_V")
_T = TypeVar("_T")


class FrozenDict(dict[_K, _V]):
    """Dict-compatible snapshot that rejects in-place mutation."""

    @staticmethod
    def _immutable(*_args: object, **_kwargs: object) -> None:
        raise TypeError("canonical collections are immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    __ior__ = _immutable  # type: ignore[assignment]
    clear = _immutable
    pop = _immutable  # type: ignore[assignment]
    popitem = _immutable  # type: ignore[assignment]
    setdefault = _immutable  # type: ignore[assignment]
    update = _immutable


class FrozenList(list[_T]):
    """List-compatible snapshot that rejects in-place mutation."""

    @staticmethod
    def _immutable(*_args: object, **_kwargs: object) -> None:
        raise TypeError("canonical collections are immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    __iadd__ = _immutable  # type: ignore[assignment]
    __imul__ = _immutable  # type: ignore[assignment]
    append = _immutable
    clear = _immutable
    extend = _immutable
    insert = _immutable
    pop = _immutable  # type: ignore[assignment]
    remove = _immutable
    reverse = _immutable
    sort = _immutable


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return FrozenDict({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return FrozenList(_deep_freeze(item) for item in value)
    return value


def _nonblank(value: str) -> str:
    if not value.strip():
        raise ValueError("must not be blank")
    return value


NonBlankString: TypeAlias = Annotated[str, AfterValidator(_nonblank)]


def ensure_json_compatible(value: Any) -> Any:
    """Reject unsafe or non-JSON values with deterministic resource limits."""
    seen: set[int] = set()
    nodes = 0

    def visit(item: Any, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > MAX_JSON_NODES:
            raise ValueError(f"JSON values must contain at most {MAX_JSON_NODES} nodes")
        if depth > MAX_JSON_DEPTH:
            raise ValueError(f"JSON values must be at most {MAX_JSON_DEPTH} levels deep")

        if item is None or isinstance(item, (str, bool, int)):
            return
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError("JSON numbers must be finite")
            return

        if isinstance(item, (list, dict)):
            identity = id(item)
            if identity in seen:
                raise ValueError("JSON values must not contain cycles")
            seen.add(identity)
            try:
                if isinstance(item, list):
                    for child in item:
                        visit(child, depth + 1)
                else:
                    for key, child in item.items():
                        if not isinstance(key, str):
                            raise ValueError("JSON object keys must be strings")
                        visit(child, depth + 1)
            finally:
                seen.remove(identity)
            return

        raise ValueError(f"{type(item).__name__} is not JSON compatible")

    visit(value, 0)
    return value


class CanonicalModel(BaseModel):
    """Strict, deeply immutable base for canonical wire-safe models."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False, frozen=True)

    @model_validator(mode="after")
    def freeze_collections(self) -> Self:
        for field_name in type(self).model_fields:
            value = getattr(self, field_name)
            object.__setattr__(self, field_name, _deep_freeze(value))
        return self


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
