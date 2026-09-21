"""Canonical decision answers and success result."""

from __future__ import annotations

import math
from typing import Annotated, Any, Literal, TypeAlias, cast

from pydantic import (
    Field,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

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
    noul: Probability | None = None
    value: bool | None = None
    confidence: Probability | None = None

    @model_validator(mode="after")
    def validate_answer_mode(self) -> NoulAnswer:
        probability_mode = self.noul is not None and self.value is None and self.confidence is None
        selection_mode = (
            self.noul is None and self.value is not None and self.confidence is not None
        )
        if not (probability_mode or selection_mode):
            raise ValueError("noul answer must contain either noul or both value and confidence")
        return self

    @model_serializer(mode="wrap")
    def serialize_answer(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        serialized = cast(dict[str, Any], handler(self))
        if self.noul is not None:
            serialized.pop("value", None)
            serialized.pop("confidence", None)
        else:
            serialized.pop("noul", None)
        return serialized


class ChoiceAnswer(CanonicalModel):
    type: Literal["choice"] = "choice"
    choice: NonBlankString
    confidence: Probability
    probabilities: (
        Annotated[
            dict[NonBlankString, Probability],
            Field(min_length=1),
        ]
        | None
    ) = None

    @model_validator(mode="after")
    def validate_probability_distribution(self) -> ChoiceAnswer:
        if self.probabilities is not None:
            if self.choice not in self.probabilities:
                raise ValueError("choice must have a matching probability")
            _validate_probability_distribution(self.probabilities)
        return self

    @model_serializer(mode="wrap")
    def serialize_answer(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        serialized = cast(dict[str, Any], handler(self))
        if self.probabilities is None:
            serialized.pop("probabilities", None)
        return serialized


class ScoreAnswer(CanonicalModel):
    type: Literal["score"] = "score"
    score: FiniteFloat
    confidence: Probability
    legend: Annotated[dict[NonBlankString, JsonValue], Field(min_length=1)]
    probabilities: (
        Annotated[
            dict[NonBlankString, Probability],
            Field(min_length=1),
        ]
        | None
    ) = None

    @field_validator("legend", mode="before")
    @classmethod
    def validate_legend_json(cls, value: Any) -> Any:
        return ensure_json_compatible(value)

    @model_validator(mode="after")
    def validate_probability_distribution(self) -> ScoreAnswer:
        if self.probabilities is not None:
            if self.legend.keys() != self.probabilities.keys():
                raise ValueError("legend and probabilities must have identical keys")
            _validate_probability_distribution(self.probabilities)
        return self

    @model_serializer(mode="wrap")
    def serialize_answer(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        serialized = cast(dict[str, Any], handler(self))
        if self.probabilities is None:
            serialized.pop("probabilities", None)
        return serialized


ABSTAIN_ANSWER_ID = "__abstain__"


class AbstainAnswer(CanonicalModel):
    type: Literal["abstain"] = "abstain"
    reason: Literal["insufficient_evidence", "provider_refusal", "content_filter"]
    source_question_type: Literal["noul", "choice", "score"]
    confidence: Probability | None = None
    probabilities: (
        Annotated[
            dict[NonBlankString, Probability],
            Field(min_length=1),
        ]
        | None
    ) = None
    legend: Annotated[dict[NonBlankString, JsonValue], Field(min_length=1)] | None = None

    @field_validator("legend", mode="before")
    @classmethod
    def validate_legend_json(cls, value: Any) -> Any:
        return ensure_json_compatible(value)

    @model_validator(mode="after")
    def validate_answer_mode(self) -> AbstainAnswer:
        if self.confidence is not None and self.probabilities is not None:
            if ABSTAIN_ANSWER_ID not in self.probabilities:
                raise ValueError(
                    "probability distribution must include canonical abstain key "
                    f"{ABSTAIN_ANSWER_ID!r}"
                )
            if self.probabilities[ABSTAIN_ANSWER_ID] != self.confidence:
                raise ValueError("abstain probability must exactly equal confidence")
            _validate_probability_distribution(self.probabilities)
            if self.legend is not None and self.legend.keys() != self.probabilities.keys():
                raise ValueError("legend and probabilities must have identical keys")
        elif self.confidence is None and self.probabilities is None and self.legend is None:
            pass
        else:
            raise ValueError(
                "abstain answer must contain either confidence and probabilities or none of "
                "confidence, probabilities, and legend"
            )
        return self

    @model_serializer(mode="wrap")
    def serialize_answer(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        serialized = cast(dict[str, Any], handler(self))
        if self.confidence is None:
            serialized.pop("confidence", None)
        if self.probabilities is None:
            serialized.pop("probabilities", None)
        if self.legend is None:
            serialized.pop("legend", None)
        return serialized


Answer: TypeAlias = Annotated[
    NoulAnswer | ChoiceAnswer | ScoreAnswer | AbstainAnswer,
    Field(discriminator="type"),
]


class Usage(CanonicalModel):
    input_tokens: NonNegativeInt | None = None
    output_tokens: NonNegativeInt | None = None


class DecisionResult(CanonicalModel):
    backend: NonBlankString
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
