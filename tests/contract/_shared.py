"""Shared request/response builders for backend contract tests."""

from __future__ import annotations

from typing import Any

from bruv.domain.questions import ChoiceQuestion, NoulQuestion, ScoreQuestion
from bruv.domain.requests import DecisionRequest

__all__ = ["mixed_request", "mixed_response"]


def mixed_request() -> DecisionRequest:
    return DecisionRequest(
        state="please route this ticket",
        questions={
            "route": ChoiceQuestion(
                instructions="Which team handles this?",
                criteria={"sales": None, "billing": None},
            ),
            "urgency": ScoreQuestion(
                instructions="How urgent is this?",
                criteria=["low", "medium", "high"],
            ),
            "asks_for_refund": NoulQuestion(instructions="Does the customer ask for a refund?"),
        },
    )


def mixed_response() -> dict[str, Any]:
    return {
        "model": "Qwen/Qwen3.5-0.8B",
        "answers": {
            "route": {
                "type": "choice",
                "choice": "billing",
                "confidence": 0.9,
                "probabilities": {"billing": 0.9, "other": 0.1},
            },
            "urgency": {
                "type": "score",
                "score": 1.8,
                "confidence": 0.8,
                "legend": {"0": "low", "1": "medium", "2": "high"},
                "probabilities": {"0": 0.0, "1": 0.2, "2": 0.8},
            },
            "asks_for_refund": {"type": "noul", "noul": 0.75},
        },
        "usage": {"input_tokens": 40, "output_tokens": 0},
    }
