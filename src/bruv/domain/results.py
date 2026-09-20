"""Canonical decision answers and success result."""

from __future__ import annotations

import math
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import Field, field_validator, model_validator

from bruv.domain.questions import (
    CanonicalModel,
    JsonValue,
    NonBlankString,
    ensure_json_compatible,
)

Probability: TypeAlias = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
FiniteFloat: TypeAlias = Annotated[float, Field(allow_inf_nan=False)]
NonNegativeFiniteFloat: TypeAlias = Annotated[
    float,
    Field(ge=0.0, allow_inf_nan=False),
]
NonNegativeInt: TypeAlias = Annotated[int, Field(ge=0)]

# Accommodates normal six-decimal provider rounding without accepting material drift.
PROBABILITY_SUM_ABS_TOLERANCE = 1e-5


def _validate_probability_distribution(probabilities: dict[str, float]) -> None:
    if not probabilities:
        raise ValueError("probability distribution must not be empty")
    if not math.isclose(
        math.fsum(probabilities.values()),
        1.0,
        rel_tol=0.0,
        abs_tol=PROBABILITY_SUM_ABS_TOLERANCE,
    ):
        raise ValueError(
            f"probability distribution must sum to 1 within {PROBABILITY_SUM_ABS_TOLERANCE:g}"
        )


class NoulAnswer(CanonicalModel):
    type: Literal["noul"] = "noul"
    noul: Probability


class ChoiceAnswer(CanonicalModel):
    type: Literal["choice"] = "choice"
    choice: NonBlankString
    confidence: Probability
    probabilities: Annotated[
        dict[NonBlankString, Probability],
        Field(min_length=1),
    ]

    @model_validator(mode="after")
    def validate_probability_distribution(self) -> ChoiceAnswer:
        if self.choice not in self.probabilities:
            raise ValueError("choice must have a matching probability")
        _validate_probability_distribution(self.probabilities)
        return self


class ScoreAnswer(CanonicalModel):
    type: Literal["score"] = "score"
    score: FiniteFloat
    confidence: Probability
    legend: Annotated[dict[NonBlankString, JsonValue], Field(min_length=1)]
    probabilities: Annotated[
        dict[NonBlankString, Probability],
        Field(min_length=1),
    ]

    @field_validator("legend", mode="before")
    @classmethod
    def validate_legend_json(cls, value: Any) -> Any:
        return ensure_json_compatible(value)

    @model_validator(mode="after")
    def validate_probability_distribution(self) -> ScoreAnswer:
        if self.legend.keys() != self.probabilities.keys():
            raise ValueError("legend and probabilities must have identical keys")
        _validate_probability_distribution(self.probabilities)
        return self


Answer: TypeAlias = Annotated[
    NoulAnswer | ChoiceAnswer | ScoreAnswer,
    Field(discriminator="type"),
]


class Usage(CanonicalModel):
    input_tokens: NonNegativeInt | None = None
    output_tokens: NonNegativeInt | None = None


class DecisionResult(CanonicalModel):
    backend: Literal["typesafe", "simple-jev"]
    model: NonBlankString
    calibrated: bool
    answers: dict[NonBlankString, Answer]
    usage: Usage | None = None
    latency_ms: NonNegativeFiniteFloat | None = None
    request_id: NonBlankString | None = None
    provider_metadata: dict[str, JsonValue] | None = None

    @field_validator("provider_metadata", mode="before")
    @classmethod
    def validate_provider_metadata_json(cls, value: Any) -> Any:
        return ensure_json_compatible(value)
