"""RLCD Plan Task 10: abstain answer contract tests (post-behavior verification).

Covers the canonical AbstainAnswer model, sentinel normalization through the
RLCD mapping helpers, human/JSON rendering, field extraction, and gate/exit
policy. No network, no optional extras.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from bruv.backends.rlcd_calibration import (
    CANONICAL_ABSTAIN_ID,
    UPSTREAM_ABSTAIN_SENTINEL,
)
from bruv.backends.rlcd_modernbert import (
    ABSTENTION_DESCRIPTION,
    RlcdModernBertAdapter,
    _candidate_list,
    _map_answer,
)
from bruv.domain.questions import ChoiceQuestion, NoulQuestion, ScoreQuestion
from bruv.domain.results import (
    ABSTAIN_ANSWER_ID,
    PROBABILITY_SUM_ABS_TOLERANCE,
    AbstainAnswer,
    ChoiceAnswer,
    DecisionResult,
    NoulAnswer,
    ScoreAnswer,
)
from bruv.exit_codes import ABSTAINED, CommandOutcome, exit_code_for
from bruv.gates import GateOptions, apply_gate
from bruv.output.fields import FieldError, get_field
from bruv.output.json_output import render_success
from bruv.output.terminal import render_human_result

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _abstain_result(**overrides: object) -> DecisionResult:
    answer_payload: dict[str, object] = {
        "type": "abstain",
        "reason": "insufficient_evidence",
        "source_question_type": "score",
        "confidence": 0.6,
        "probabilities": {"0": 0.1, "1": 0.3, ABSTAIN_ANSWER_ID: 0.6},
        "legend": {"0": "low", "1": "high", ABSTAIN_ANSWER_ID: ABSTENTION_DESCRIPTION},
    }
    answer_payload.update(overrides)
    return DecisionResult(
        backend="rlcd-modernbert",
        model="RLCD/ModernBERT",
        calibrated=True,
        answers={"urgency": answer_payload},  # type: ignore[arg-type]
    )


def _label_only_abstain_result(source_question_type: str = "choice") -> DecisionResult:
    return DecisionResult(
        backend="needle",
        model="Cactus-Compute/needle3",
        calibrated=True,
        answers={
            "route": {
                "type": "abstain",
                "reason": "provider_refusal",
                "source_question_type": source_question_type,
            }
        },  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Probability-backed abstain answers
# ---------------------------------------------------------------------------


def test_probability_backed_abstain_parses() -> None:
    result = _abstain_result()
    answer = result.answers["urgency"]
    assert isinstance(answer, AbstainAnswer)
    assert answer.confidence == 0.6
    assert answer.probabilities is not None
    assert answer.probabilities[ABSTAIN_ANSWER_ID] == answer.confidence


@pytest.mark.parametrize("value", [-0.01, 1.01, float("nan"), float("inf"), float("-inf")])
def test_abstain_probabilities_must_be_finite_and_bounded(value: float) -> None:
    with pytest.raises(ValidationError):
        _abstain_result(
            probabilities={"0": value, "1": 0.6, ABSTAIN_ANSWER_ID: 0.4},
            confidence=0.4,
        )


def test_abstain_confidence_must_be_finite_probability() -> None:
    with pytest.raises(ValidationError):
        _abstain_result(confidence=1.5, probabilities={"0": 1.0, ABSTAIN_ANSWER_ID: 1.5})


def test_abstain_probabilities_must_sum_to_one_within_tolerance() -> None:
    # 5e-6 below 1.0: inside the 1e-5 absolute tolerance, must be accepted.
    accepted = _abstain_result(
        confidence=0.6,
        probabilities={"0": 0.2, "1": 0.199995, ABSTAIN_ANSWER_ID: 0.6},
        legend={"0": "low", "1": "medium", ABSTAIN_ANSWER_ID: ABSTENTION_DESCRIPTION},
    )
    answer = accepted.answers["urgency"]
    assert isinstance(answer, AbstainAnswer)
    assert answer.probabilities == {"0": 0.2, "1": 0.199995, ABSTAIN_ANSWER_ID: 0.6}

    # 2e-5 below 1.0: outside the tolerance, must be rejected.
    with pytest.raises(ValidationError, match=f"within {PROBABILITY_SUM_ABS_TOLERANCE:g}"):
        _abstain_result(
            confidence=0.6,
            probabilities={"0": 0.2, "1": 0.19998, ABSTAIN_ANSWER_ID: 0.6},
            legend={"0": "low", "1": "medium", ABSTAIN_ANSWER_ID: ABSTENTION_DESCRIPTION},
        )

    # Far outside: 0.5 drift, still rejected.
    with pytest.raises(ValidationError, match=f"within {PROBABILITY_SUM_ABS_TOLERANCE:g}"):
        _abstain_result(
            confidence=0.7,
            probabilities={"0": 0.4, "1": 0.4, ABSTAIN_ANSWER_ID: 0.7},
        )


def test_abstain_probability_backed_requires_probabilities_key_and_exact_confidence() -> None:
    with pytest.raises(ValidationError, match="canonical abstain key"):
        _abstain_result(
            confidence=0.6,
            probabilities={"0": 0.4, "1": 0.6},
            legend={"0": "low", "1": "high"},
        )
    with pytest.raises(ValidationError, match="must exactly equal confidence"):
        _abstain_result(
            confidence=0.5,
            probabilities={"0": 0.4, "1": 0.3, ABSTAIN_ANSWER_ID: 0.3},
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"confidence": 0.6},
        {"probabilities": {"0": 0.4, "1": 0.6, ABSTAIN_ANSWER_ID: 0.6}},
        {"legend": {"0": "low", "1": "high", ABSTAIN_ANSWER_ID: ABSTENTION_DESCRIPTION}},
        {
            "probabilities": {"0": 0.4, "1": 0.6, ABSTAIN_ANSWER_ID: 0.6},
            "legend": {"0": "low", "1": "high", ABSTAIN_ANSWER_ID: ABSTENTION_DESCRIPTION},
        },
        {
            "confidence": 0.6,
            "legend": {"0": "low", "1": "high", ABSTAIN_ANSWER_ID: ABSTENTION_DESCRIPTION},
        },
    ],
)
def test_partial_or_mixed_probability_backed_modes_are_invalid(
    overrides: dict[str, object],
) -> None:
    payload: dict[str, object] = {
        "reason": "insufficient_evidence",
        "source_question_type": "score",
    }
    payload.update(overrides)
    with pytest.raises(ValidationError, match="either confidence and probabilities or none"):
        AbstainAnswer(**payload)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "reason",
    ["insufficient_evidence", "provider_refusal", "content_filter"],
)
def test_valid_abstain_reasons_and_source_types(reason: str) -> None:
    answer = AbstainAnswer(
        reason=reason,  # type: ignore[arg-type]
        source_question_type="noul",
    )
    assert answer.reason == reason
    for source in ("noul", "choice", "score"):
        answer = AbstainAnswer(reason=reason, source_question_type=source)  # type: ignore[arg-type]
        assert answer.source_question_type == source


def test_invalid_abstain_reason_or_source_type_rejected() -> None:
    with pytest.raises(ValidationError):
        AbstainAnswer(reason="model_says_no", source_question_type="choice")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        AbstainAnswer(reason="provider_refusal", source_question_type="ranking")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Label-only abstain answers
# ---------------------------------------------------------------------------


def test_label_only_abstain_omits_inactive_fields_when_serialized() -> None:
    result = _label_only_abstain_result()
    answer = result.answers["route"]
    assert isinstance(answer, AbstainAnswer)
    assert answer.confidence is None
    assert answer.probabilities is None
    assert answer.legend is None

    expected = {
        "type": "abstain",
        "reason": "provider_refusal",
        "source_question_type": "choice",
    }
    assert answer.model_dump() == expected
    assert json.loads(answer.model_dump_json()) == expected
    assert result.model_dump()["answers"]["route"] == expected


def test_label_only_abstain_partial_presence_invalid() -> None:
    with pytest.raises(ValidationError, match="either confidence and probabilities or none"):
        AbstainAnswer(
            reason="insufficient_evidence",
            source_question_type="noul",
            confidence=0.5,
        )
    with pytest.raises(ValidationError, match="either confidence and probabilities or none"):
        AbstainAnswer(
            reason="insufficient_evidence",
            source_question_type="noul",
            legend={"a": 1},
        )
    with pytest.raises(ValidationError, match="either confidence and probabilities or none"):
        AbstainAnswer(
            reason="insufficient_evidence",
            source_question_type="noul",
            confidence=0.5,
            legend={"a": 1},
        )


# ---------------------------------------------------------------------------
# Legends
# ---------------------------------------------------------------------------


def test_probability_backed_legend_must_cover_full_probability_keys_including_abstain() -> None:
    result = _abstain_result()
    answer = result.answers["urgency"]
    assert isinstance(answer, AbstainAnswer)
    assert answer.legend is not None
    assert answer.probabilities is not None
    assert answer.legend.keys() == answer.probabilities.keys()
    assert ABSTAIN_ANSWER_ID in answer.legend
    assert answer.legend[ABSTAIN_ANSWER_ID] == ABSTENTION_DESCRIPTION


def test_probability_backed_legend_mismatch_invalid() -> None:
    with pytest.raises(ValidationError, match="identical keys"):
        _abstain_result(
            legend={"0": "low", "1": "high"},  # missing the abstain entry
        )
    with pytest.raises(ValidationError, match="identical keys"):
        _abstain_result(
            legend={"0": "low", "1": "high", ABSTAIN_ANSWER_ID: ABSTENTION_DESCRIPTION, "2": "x"},
        )


def test_label_only_legend_absent_for_probability_backed_adversarial_shape() -> None:
    # Legend without probability backing is rejected by the mode validator.
    with pytest.raises(ValidationError, match="either confidence and probabilities or none"):
        AbstainAnswer(
            reason="insufficient_evidence",
            source_question_type="score",
            legend={"0": "low", "1": "high", ABSTAIN_ANSWER_ID: ABSTENTION_DESCRIPTION},
        )


# ---------------------------------------------------------------------------
# Adapter-produced sentinel normalization (network-free helpers)
# ---------------------------------------------------------------------------


def _score_question() -> ScoreQuestion:
    return ScoreQuestion(instructions="How urgent?", criteria=["low", "medium", "high"])


def _choice_question() -> ChoiceQuestion:
    return ChoiceQuestion(instructions="Route?", criteria={"sales": 1, "billing": 2})


def _abstain_map(question: object, probs: list[float]) -> AbstainAnswer:
    candidates = _candidate_list(question)  # type: ignore[arg-type]
    winner = len(candidates) - 1
    answer, _, _ = _map_answer(question, candidates, probs, winner)  # type: ignore[arg-type]
    assert isinstance(answer, AbstainAnswer)
    return answer


def test_adapter_abstain_probabilities_use_canonical_keys_only() -> None:
    probs = [0.2, 0.1, 0.1, 0.6]
    answer = _abstain_map(_score_question(), probs)
    assert set(answer.probabilities or {}) == {"0", "1", "2", ABSTAIN_ANSWER_ID}
    assert UPSTREAM_ABSTAIN_SENTINEL not in (answer.probabilities or {})


def test_adapter_choice_abstain_legend_covers_full_distribution_with_reserved_mapping() -> None:
    probs = [0.3, 0.2, 0.5]
    answer = _abstain_map(_choice_question(), probs)
    assert answer.probabilities is not None
    assert answer.legend is not None
    assert answer.legend.keys() == answer.probabilities.keys()
    assert answer.legend == {
        "sales": 1,
        "billing": 2,
        ABSTAIN_ANSWER_ID: ABSTENTION_DESCRIPTION,
    }
    assert answer.probabilities[ABSTAIN_ANSWER_ID] == answer.confidence


def test_adapter_noul_abstain_has_no_legend() -> None:
    question = NoulQuestion(instructions="Refund requested?")
    answer = _abstain_map(question, [0.2, 0.3, 0.5])
    assert answer.legend is None
    assert answer.source_question_type == "noul"
    assert set(answer.probabilities or {}) == {"true", "false", ABSTAIN_ANSWER_ID}


def test_upstream_sentinel_never_serialized() -> None:
    probs = [0.2, 0.1, 0.1, 0.6]
    answer = _abstain_map(_score_question(), probs)
    serialized = json.dumps(answer.model_dump(mode="json"))
    assert UPSTREAM_ABSTAIN_SENTINEL not in serialized
    assert CANONICAL_ABSTAIN_ID == ABSTAIN_ANSWER_ID
    assert ABSTAIN_ANSWER_ID in serialized
    # The adapter class itself must not embed the sentinel anywhere canonical.
    assert not hasattr(RlcdModernBertAdapter, UPSTREAM_ABSTAIN_SENTINEL)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_render_human_result_abstain_with_probability() -> None:
    result = _abstain_result()
    text = render_human_result(result)
    assert "urgency: abstained (insufficient_evidence, p=0.600)" in text
    assert "calibration: calibrated" in text


def test_render_human_result_abstain_without_probability() -> None:
    result = _label_only_abstain_result()
    text = render_human_result(result)
    assert "route: abstained (provider_refusal)" in text
    assert "p=" not in text


def test_render_success_json_includes_abstain_and_omits_inactive_fields() -> None:
    rendered = render_success(_abstain_result())
    payload = json.loads(rendered)
    answer = payload["answers"]["urgency"]
    assert answer["type"] == "abstain"
    assert answer["confidence"] == 0.6
    assert answer["probabilities"][ABSTAIN_ANSWER_ID] == 0.6
    assert answer["reason"] == "insufficient_evidence"
    assert answer["source_question_type"] == "score"

    label_only = json.loads(render_success(_label_only_abstain_result()))
    route = label_only["answers"]["route"]
    assert route["type"] == "abstain"
    for key in ("confidence", "probabilities", "legend"):
        assert key not in route


# ---------------------------------------------------------------------------
# Field extraction
# ---------------------------------------------------------------------------


def test_get_field_resolves_abstain_fields() -> None:
    result = _abstain_result()
    assert get_field(result, "answers.urgency.reason") == "insufficient_evidence"
    assert get_field(result, "answers.urgency.source_question_type") == "score"
    assert get_field(result, "answers.urgency.confidence") == 0.6
    assert get_field(result, f"answers.urgency.probabilities.{ABSTAIN_ANSWER_ID}") == 0.6


def test_get_field_resolves_probability_backed_legend_keys() -> None:
    result = _abstain_result()  # score abstain with full legend
    assert get_field(result, "answers.urgency.legend.0") == "low"
    assert get_field(result, "answers.urgency.legend.1") == "high"
    abstain_legend = get_field(result, f"answers.urgency.legend.{ABSTAIN_ANSWER_ID}")
    assert abstain_legend == ABSTENTION_DESCRIPTION


def test_get_field_unknown_legend_key_is_stable_error() -> None:
    result = _abstain_result()
    with pytest.raises(FieldError, match="no key '9'"):
        get_field(result, "answers.urgency.legend.9")


def test_get_field_resolves_label_only_abstain_reason() -> None:
    result = _label_only_abstain_result()
    assert get_field(result, "answers.route.reason") == "provider_refusal"


def test_get_field_invalid_field_raises_stable_field_error() -> None:
    result = _abstain_result()
    with pytest.raises(FieldError):
        get_field(result, "answers.urgency.nonexistent")
    with pytest.raises(FieldError):
        get_field(result, "answers.nope.type")
    with pytest.raises(FieldError):
        get_field(result, "answers.urgency._private")


def test_get_field_label_only_confidence_returns_null() -> None:
    result = _label_only_abstain_result()
    assert get_field(result, "answers.route.confidence") is None


# ---------------------------------------------------------------------------
# Gates and exit policy
# ---------------------------------------------------------------------------


def test_exact_targeted_abstained_answer_abstains_with_exit_eleven() -> None:
    result = _abstain_result()
    outcome = apply_gate(result, GateOptions(field="answers.urgency.score", fail_under=0.5))
    assert outcome.abstained is True
    assert outcome.gate_passed is None
    assert exit_code_for(CommandOutcome(abstained=outcome.abstained)) == ABSTAINED


def test_wildcard_with_only_abstain_answers_abstains() -> None:
    result = _label_only_abstain_result()
    outcome = apply_gate(result, GateOptions(field="answers.*.score", fail_under=0.5))
    assert outcome.abstained is True
    assert exit_code_for(CommandOutcome(abstained=True)) == ABSTAINED


def test_mixed_sibling_substantive_remains_gateable() -> None:
    result = _abstain_result()
    mixed = DecisionResult(
        backend="rlcd-modernbert",
        model="RLCD/ModernBERT",
        calibrated=True,
        answers={
            "urgency": result.answers["urgency"],
            "severity": ScoreAnswer(
                score=0.8,
                confidence=0.9,
                legend={"0": "low", "1": "high"},
                probabilities={"0": 0.2, "1": 0.8},
            ),
        },
    )
    outcome = apply_gate(mixed, GateOptions(field="answers.*.score", fail_under=0.5))
    assert outcome.abstained is False
    assert outcome.gate_passed is True
    assert outcome.value == pytest.approx(0.8)


def test_missing_answer_stays_field_error() -> None:
    result = _abstain_result()
    with pytest.raises(FieldError):
        apply_gate(result, GateOptions(field="answers.severity.score", fail_under=0.5))


def test_malformed_field_stays_field_error() -> None:
    result = _abstain_result()
    with pytest.raises(FieldError):
        apply_gate(result, GateOptions(field="usage.route.score", fail_under=0.5))


# ---------------------------------------------------------------------------
# Existing TypeSafe / Simple / Needle result serialization unchanged
# ---------------------------------------------------------------------------


def _simple_result() -> DecisionResult:
    return DecisionResult(
        backend="simple-jev",
        model="Qwen/Qwen3.5-0.8B",
        calibrated=False,
        answers={
            "route": ChoiceAnswer(
                choice="billing",
                confidence=0.9,
                probabilities={"billing": 0.9, "other": 0.1},
            ),
            "urgency": {
                "type": "score",
                "score": 1.8,
                "confidence": 0.8,
                "legend": {"0": "low", "1": "medium", "2": "high"},
                "probabilities": {"0": 0.0, "1": 0.2, "2": 0.8},
            },  # type: ignore[arg-type]
        },
    )


def _typesafe_result() -> DecisionResult:
    return DecisionResult(
        backend="typesafe",
        model="jev-1.13.0",
        calibrated=True,
        answers={
            "refund": NoulAnswer(value=True, confidence=0.9),
            "route": ChoiceAnswer(choice="sales", confidence=0.8),
        },
    )


def _needle_result() -> DecisionResult:
    return DecisionResult(
        backend="needle",
        model="Cactus-Compute/needle3",
        calibrated=True,
        answers={
            "refund": NoulAnswer(noul=0.75),
            "urgency": ScoreAnswer(
                score=2.0,
                confidence=0.7,
                legend={"0": "low", "1": "medium", "2": "high"},
            ),
        },
    )


def test_typesafe_serialization_unchanged() -> None:
    expected_answers = {
        "refund": {"type": "noul", "value": True, "confidence": 0.9},
        "route": {"type": "choice", "choice": "sales", "confidence": 0.8},
    }
    assert _typesafe_result().model_dump()["answers"] == expected_answers


def test_simple_jev_serialization_unchanged() -> None:
    expected_answers = {
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
    }
    assert _simple_result().model_dump()["answers"] == expected_answers


def test_needle_serialization_unchanged() -> None:
    expected_answers = {
        "refund": {"type": "noul", "noul": 0.75},
        "urgency": {
            "type": "score",
            "score": 2.0,
            "confidence": 0.7,
            "legend": {"0": "low", "1": "medium", "2": "high"},
        },
    }
    assert _needle_result().model_dump()["answers"] == expected_answers


def test_abstain_tolerance_constant_is_unchanged() -> None:
    assert PROBABILITY_SUM_ABS_TOLERANCE == 1e-5
