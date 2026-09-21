"""Unit tests for the RLCD ModernBERT adapter using fake tokenizer/runtime/calibrator.

No model weights, ONNX runtime, or network is involved: the tokenizer and
runtime are fakes that record calls, and the calibrator is the real in-memory
artifact dataclass with a full per_k table.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
from pydantic import ValidationError

from bruv.application import (
    ConfigurationError,
    DecisionFacade,
    ProviderResponseError,
    RequestValidationError,
)
from bruv.backends.rlcd_artifacts import REPO_ID, REVISION
from bruv.backends.rlcd_calibration import (
    CALIBRATOR_SCOPE,
    UPSTREAM_ABSTAIN_SENTINEL,
    RlcdCalibrator,
)
from bruv.backends.rlcd_modernbert import (
    ABSTENTION_DESCRIPTION,
    RlcdModernBertAdapter,
)
from bruv.domain.requests import DecisionRequest
from bruv.domain.results import ABSTAIN_ANSWER_ID
from bruv.domain.validation import validate_request

LOGIT_WIDTH = 25
LEAK_SENTINEL = "PROMPT-LEAK-SENTINEL-9d41c2f7b8e6"


def _temperatures() -> dict[int, float]:
    return {k: 0.5 + 0.25 * k for k in range(2, LOGIT_WIDTH + 1)}


def _calibrator() -> RlcdCalibrator:
    return RlcdCalibrator(
        model_id=REPO_ID,
        temperature=99.0,
        log_temperature=88.0,
        scope=CALIBRATOR_SCOPE,
        per_k=_temperatures(),
        artifact_hash="calibrator-hash",
    )


class FakeTokenizer:
    """Records encode_batch calls and returns fixed 4-token encodings."""

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def encode_batch(self, texts: list[str]) -> tuple[object, object]:
        self.batches.append(list(texts))
        size = len(texts)
        return [[1, 2, 3, 4]] * size, [[1, 1, 1, 1]] * size


class FakeRuntime:
    """Records run calls and returns pre-built logits rows."""

    def __init__(self, rows: list[list[float]]) -> None:
        self.rows = rows
        self.calls = 0
        self.last_kwargs: dict[str, object] | None = None

    def run(self, input_ids: object, attention_mask: object) -> object:
        self.calls += 1
        self.last_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        return self.rows


def _logits_row(winner: int, width: int = LOGIT_WIDTH) -> list[float]:
    row = [0.25 * index for index in range(width)]
    row[winner] = 10.0
    return row


def _adapter(runtime: FakeRuntime, tokenizer: FakeTokenizer) -> RlcdModernBertAdapter:
    return RlcdModernBertAdapter(
        runtime=runtime,
        tokenizer=tokenizer,
        calibrator=_calibrator(),
    )


def _softmax(logits: list[float], temperature: float) -> list[float]:
    scaled = [value / temperature for value in logits]
    peak = max(scaled)
    exps = [math.exp(value - peak) for value in scaled]
    total = math.fsum(exps)
    return [exp / total for exp in exps]


def _noul_request(state: Any = "pipe state") -> DecisionRequest:
    return DecisionRequest(
        state=state,
        questions={"q1": {"type": "noul", "instructions": "the pipe is broken"}},
    )


def _choice_request(criteria: dict[str, Any], state: Any = "ctx") -> DecisionRequest:
    return DecisionRequest(
        state=state,
        questions={"q1": {"type": "choice", "instructions": "pick one", "criteria": criteria}},
    )


def _score_request(criteria: list[Any], state: Any = "ctx") -> DecisionRequest:
    return DecisionRequest(
        state=state,
        questions={"q1": {"type": "score", "instructions": "rate it", "criteria": criteria}},
    )


def _evaluate(adapter: RlcdModernBertAdapter, request: DecisionRequest) -> Any:
    return DecisionFacade(backend=adapter).evaluate(request)


# ---------------------------------------------------------------------------
# Literal prompt formatting and marker assembly
# ---------------------------------------------------------------------------


def test_noul_prompt_is_exact_literal_assembly() -> None:
    runtime = FakeRuntime([_logits_row(0)])
    tokenizer = FakeTokenizer()
    _evaluate(_adapter(runtime, tokenizer), _noul_request(state="pipe state"))

    expected = (
        "<<LABEL>>true: the pipe is broken"
        "<<LABEL>>false: not the pipe is broken"
        f"<<LABEL>>{ABSTENTION_DESCRIPTION}"
        "<<SEP>>Context:\npipe state\n\nEvaluate proposition: the pipe is broken"
    )
    assert tokenizer.batches == [[expected]]


def test_choice_prompt_uses_it_is_labels_and_question_first_text() -> None:
    runtime = FakeRuntime([_logits_row(0)])
    tokenizer = FakeTokenizer()
    _evaluate(
        _adapter(runtime, tokenizer),
        _choice_request({"yes": "ok", "no": 3}, state={"b": 1, "a": [1, 2]}),
    )

    expected = (
        "<<LABEL>>It is ok"
        "<<LABEL>>It is 3"
        f"<<LABEL>>{ABSTENTION_DESCRIPTION}"
        '<<SEP>>Question: pick one\n\nContext:\n{"a":[1,2],"b":1}'
    )
    assert tokenizer.batches == [[expected]]


def test_score_prompt_labels_embed_level_json_and_value() -> None:
    runtime = FakeRuntime([_logits_row(0)])
    tokenizer = FakeTokenizer()
    _evaluate(
        _adapter(runtime, tokenizer),
        _score_request(["good", {"k": True}], state="ctx"),
    )

    expected = (
        '<<LABEL>>"good" (Value: 0)'
        '<<LABEL>>{"k":true} (Value: 1)'
        f"<<LABEL>>{ABSTENTION_DESCRIPTION}"
        "<<SEP>>Question: rate it\n\nContext:\nctx"
    )
    assert tokenizer.batches == [[expected]]


def test_multi_question_request_is_one_batch_and_exactly_one_run() -> None:
    rows = [_logits_row(0), _logits_row(1), _logits_row(2)]
    runtime = FakeRuntime(rows)
    tokenizer = FakeTokenizer()
    request = DecisionRequest(
        state="ctx",
        questions={
            "n": {"type": "noul", "instructions": "n?"},
            "c": {"type": "choice", "instructions": "c?", "criteria": {"a": 1, "b": 2}},
            "s": {"type": "score", "instructions": "s?", "criteria": ["low", "high"]},
        },
    )
    result = _evaluate(_adapter(runtime, tokenizer), request)

    assert len(tokenizer.batches) == 1
    assert len(tokenizer.batches[0]) == 3
    assert runtime.calls == 1
    assert runtime.last_kwargs is not None
    input_ids = runtime.last_kwargs["input_ids"]
    assert len(input_ids) == 3  # type: ignore[arg-type]
    assert set(result.answers) == {"n", "c", "s"}


# ---------------------------------------------------------------------------
# Pre-inference validation: reserved markers and candidate totals
# ---------------------------------------------------------------------------


def test_reserved_marker_in_state_fails_before_runtime_call() -> None:
    runtime = FakeRuntime([])
    tokenizer = FakeTokenizer()
    adapter = _adapter(runtime, tokenizer)

    with pytest.raises(RequestValidationError) as exc_info:
        _evaluate(adapter, _noul_request(state="has <<LABEL>> inside"))

    codes = {issue.code for issue in exc_info.value.issues}
    assert "reserved_input_marker" in codes
    assert runtime.calls == 0
    assert tokenizer.batches == []


def test_reserved_marker_in_criteria_fails_before_runtime_call() -> None:
    runtime = FakeRuntime([])
    tokenizer = FakeTokenizer()
    adapter = _adapter(runtime, tokenizer)

    with pytest.raises(RequestValidationError):
        _evaluate(adapter, _choice_request({"a": "fine", "b": "<<SEP>> leak"}))

    assert runtime.calls == 0
    assert tokenizer.batches == []
    with pytest.raises(RequestValidationError):
        _evaluate(adapter, _score_request(["ok", "<<LABEL>> bad"]))

    assert runtime.calls == 0


def test_capabilities_pin_counts_types_and_markers() -> None:
    capabilities = RlcdModernBertAdapter.capabilities
    assert capabilities.backend == "rlcd-modernbert"
    assert capabilities.question_types == frozenset({"noul", "choice", "score"})
    assert capabilities.calibrated is True
    assert capabilities.allows_json_state is True
    assert capabilities.explicit_abstention is True
    assert capabilities.reserved_input_markers == ("<<LABEL>>", "<<SEP>>")
    assert capabilities.reserved_answer_ids == frozenset({ABSTAIN_ANSWER_ID})
    assert capabilities.supported_total_candidates == frozenset({2, 3, 4, 5, 6, 7, 9, 11, 17, 25})


@pytest.mark.parametrize("substantive", [2, 3, 4, 5, 6, 8, 10, 16, 24])
def test_supported_substantive_counts_validate_clean(substantive: int) -> None:
    request = _choice_request({str(i): i for i in range(substantive)})
    valid, issues = DecisionFacade(backend=_adapter(FakeRuntime([]), FakeTokenizer())).validate(
        request
    )
    assert valid, [issue.message for issue in issues]


def test_unsupported_total_is_rejected_before_inference() -> None:
    # 25 substantive + abstain = 26 total, which is outside the supported set.
    runtime = FakeRuntime([])
    tokenizer = FakeTokenizer()
    adapter = _adapter(runtime, tokenizer)
    request = _choice_request({str(i): i for i in range(25)})

    valid, issues = DecisionFacade(backend=adapter).validate(request)
    assert not valid
    assert issues[0].code == "total_candidates_not_supported"

    with pytest.raises(RequestValidationError):
        _evaluate(adapter, request)
    assert runtime.calls == 0
    assert tokenizer.batches == []


def test_score_total_above_supported_set_is_rejected() -> None:
    request = _score_request([f"level-{i}" for i in range(25)])
    valid, issues = DecisionFacade(backend=_adapter(FakeRuntime([]), FakeTokenizer())).validate(
        request
    )
    assert not valid
    assert issues[0].code == "total_candidates_not_supported"


def test_one_substantive_candidate_is_unreachable_via_pydantic_minimums() -> None:
    with pytest.raises(ValidationError):
        _choice_request({"only": 1})
    with pytest.raises(ValidationError):
        _score_request(["only"])
    # The adapter still declares the 2-candidate total as supported.
    assert 2 in RlcdModernBertAdapter.capabilities.supported_total_candidates


def test_seven_substantive_choices_total_is_not_supported() -> None:
    request = _choice_request({str(i): i for i in range(7)})
    valid, issues = DecisionFacade(backend=_adapter(FakeRuntime([]), FakeTokenizer())).validate(
        request
    )
    assert not valid
    assert issues[0].code == "total_candidates_not_supported"


# ---------------------------------------------------------------------------
# Substantive mapping for every question type
# ---------------------------------------------------------------------------


def test_noul_substantive_mapping_reports_true_probability() -> None:
    runtime = FakeRuntime([_logits_row(0)])
    result = _evaluate(_adapter(runtime, FakeTokenizer()), _noul_request())

    answer = result.answers["q1"]
    assert answer.type == "noul"
    temperatures = _temperatures()
    probs = _softmax(_logits_row(0)[:3], temperatures[3])
    conditional = [p / math.fsum(probs[:2]) for p in probs[:2]]
    assert answer.noul == pytest.approx(conditional[0], rel=1e-12)


def test_choice_substantive_mapping_excludes_abstain_and_is_consistent() -> None:
    criteria = {"a": 1, "b": 2, "c": 3}
    runtime = FakeRuntime([_logits_row(1)])
    result = _evaluate(_adapter(runtime, FakeTokenizer()), _choice_request(criteria))

    answer = result.answers["q1"]
    assert answer.type == "choice"
    assert answer.choice == "b"
    probs = _softmax(_logits_row(1)[:4], _temperatures()[4])
    conditional = [p / math.fsum(probs[:3]) for p in probs[:3]]
    expected = {"a": conditional[0], "b": conditional[1], "c": conditional[2]}
    assert set(answer.probabilities) == {"a", "b", "c"}
    assert ABSTAIN_ANSWER_ID not in answer.probabilities
    for key, value in expected.items():
        assert answer.probabilities[key] == pytest.approx(value, rel=1e-12)
    assert answer.confidence == pytest.approx(expected["b"], rel=1e-12)
    assert math.isclose(math.fsum(answer.probabilities.values()), 1.0, abs_tol=1e-12)


def test_score_substantive_mapping_legend_and_expected_value() -> None:
    criteria = ["low", "mid", "high"]
    runtime = FakeRuntime([_logits_row(2)])
    result = _evaluate(_adapter(runtime, FakeTokenizer()), _score_request(criteria))

    answer = result.answers["q1"]
    assert answer.type == "score"
    assert answer.score == 2.0
    assert answer.legend == {"0": "low", "1": "mid", "2": "high"}
    probs = _softmax(_logits_row(2)[:4], _temperatures()[4])
    conditional = [p / math.fsum(probs[:3]) for p in probs[:3]]
    expected_dist = {str(i): conditional[i] for i in range(3)}
    assert set(answer.probabilities) == set(expected_dist)
    for key, value in expected_dist.items():
        assert answer.probabilities[key] == pytest.approx(value, rel=1e-12)
    expected_value = math.fsum(i * conditional[i] for i in range(3))
    metadata = result.provider_metadata["q1"]["rlcd"]
    assert metadata["expected_value_over_substantive_mass"] == pytest.approx(
        expected_value, rel=1e-12
    )


# ---------------------------------------------------------------------------
# Abstain mapping, full distribution, and sentinel hygiene
# ---------------------------------------------------------------------------


def _abstain_request() -> DecisionRequest:
    return _choice_request({"a": 1, "b": 2})


def test_abstain_mapping_returns_full_distribution_and_legend() -> None:
    runtime = FakeRuntime([_logits_row(2)])  # winner index 2 == abstain for K=3
    result = _evaluate(_adapter(runtime, FakeTokenizer()), _abstain_request())

    answer = result.answers["q1"]
    assert answer.type == "abstain"
    assert answer.reason == "insufficient_evidence"
    assert answer.source_question_type == "choice"
    probs = _softmax(_logits_row(2)[:3], _temperatures()[3])
    expected_full = {"a": probs[0], "b": probs[1], ABSTAIN_ANSWER_ID: probs[2]}
    assert set(answer.probabilities) == {"a", "b", ABSTAIN_ANSWER_ID}
    for key, value in expected_full.items():
        assert answer.probabilities[key] == pytest.approx(value, rel=1e-12)
    assert answer.probabilities[ABSTAIN_ANSWER_ID] == answer.confidence
    assert math.isclose(math.fsum(answer.probabilities.values()), 1.0, abs_tol=1e-12)
    assert answer.legend == {"a": 1, "b": 2, ABSTAIN_ANSWER_ID: ABSTENTION_DESCRIPTION}


def test_abstain_on_noul_has_no_legend() -> None:
    runtime = FakeRuntime([_logits_row(2)])
    result = _evaluate(_adapter(runtime, FakeTokenizer()), _noul_request())

    answer = result.answers["q1"]
    assert answer.type == "abstain"
    assert answer.legend is None
    assert answer.confidence is not None
    assert math.isclose(math.fsum(answer.probabilities.values()), 1.0, abs_tol=1e-12)


def test_upstream_sentinel_never_escapes_serialized_results() -> None:
    substantive = _evaluate(
        _adapter(FakeRuntime([_logits_row(1)]), FakeTokenizer()), _abstain_request()
    )
    abstain = _evaluate(
        _adapter(FakeRuntime([_logits_row(2)]), FakeTokenizer()), _abstain_request()
    )

    for result in (substantive, abstain):
        payload = result.model_dump_json()
        assert UPSTREAM_ABSTAIN_SENTINEL not in payload
        assert ABSTAIN_ANSWER_ID in payload


# ---------------------------------------------------------------------------
# Provider metadata
# ---------------------------------------------------------------------------


def test_metadata_revision_scope_per_k_and_probability_base() -> None:
    runtime = FakeRuntime([_logits_row(1)])
    result = _evaluate(_adapter(runtime, FakeTokenizer()), _abstain_request())

    metadata = result.provider_metadata["q1"]["rlcd"]
    assert metadata["revision"] == REVISION
    assert metadata["calibration_scope"] == CALIBRATOR_SCOPE
    assert metadata["temperature_path"] == "per_k:3"
    assert metadata["probability_base"] == "conditional_on_sufficient_evidence"
    probs = _softmax(_logits_row(1)[:3], _temperatures()[3])
    assert metadata["full_distribution"]["a"] == pytest.approx(probs[0], rel=1e-12)
    assert metadata["full_distribution"][ABSTAIN_ANSWER_ID] == pytest.approx(probs[2], rel=1e-12)


def test_expected_value_absent_for_non_score_questions() -> None:
    runtime = FakeRuntime([_logits_row(1)])
    result = _evaluate(_adapter(runtime, FakeTokenizer()), _abstain_request())

    metadata = result.provider_metadata["q1"]["rlcd"]
    assert "expected_value_over_substantive_mass" not in metadata


def test_result_headers_are_pinned() -> None:
    runtime = FakeRuntime([_logits_row(0)])
    result = _evaluate(_adapter(runtime, FakeTokenizer()), _abstain_request())

    assert result.backend == "rlcd-modernbert"
    assert result.model == REPO_ID
    assert result.calibrated is True
    assert result.usage is None


# ---------------------------------------------------------------------------
# Malformed logits and pin enforcement (all failures unpaid)
# ---------------------------------------------------------------------------


def _assert_unpaid_response_error(exc: pytest.ExceptionInfo[ProviderResponseError]) -> None:
    error = exc.value
    assert isinstance(error, ProviderResponseError)
    assert error.paid_request is False


def test_malformed_logit_rows_fail_unpaid() -> None:
    request = _choice_request({"a": 1, "b": 2})
    single_question_cases: list[object] = [
        "not-a-batch",  # not a 2D batch
        [_logits_row(0)[:24]],  # wrong width
        [["x"] * LOGIT_WIDTH],  # non-numeric
        [[True] + [0.0] * (LOGIT_WIDTH - 1)],  # boolean
        [[float("inf")] + [0.0] * (LOGIT_WIDTH - 1)],  # non-finite
    ]

    for logits in single_question_cases:
        runtime = FakeRuntime(logits)  # type: ignore[arg-type]
        with pytest.raises(ProviderResponseError) as exc_info:
            _evaluate(_adapter(runtime, FakeTokenizer()), request)
        _assert_unpaid_response_error(exc_info)
        assert runtime.calls == 1

    # Row-count mismatch with two questions.
    two_questions = DecisionRequest(
        state="ctx",
        questions={
            "q1": {"type": "noul", "instructions": "n?"},
            "q2": {"type": "noul", "instructions": "n2?"},
        },
    )
    for logits in ([_logits_row(0)], [_logits_row(0), _logits_row(1), _logits_row(2)]):
        runtime = FakeRuntime(logits)
        with pytest.raises(ProviderResponseError) as exc_info:
            _evaluate(_adapter(runtime, FakeTokenizer()), two_questions)
        _assert_unpaid_response_error(exc_info)


def test_boolean_and_nonfinite_logits_name_the_failure_without_leaking_values() -> None:
    request = _choice_request({"a": 1, "b": 2})

    runtime = FakeRuntime([[True] + [0.0] * (LOGIT_WIDTH - 1)])
    with pytest.raises(ProviderResponseError) as bool_error:
        _evaluate(_adapter(runtime, FakeTokenizer()), request)
    assert "non-numeric" in bool_error.value.message

    runtime = FakeRuntime([[float("nan")] + [0.0] * (LOGIT_WIDTH - 1)])
    with pytest.raises(ProviderResponseError) as nan_error:
        _evaluate(_adapter(runtime, FakeTokenizer()), request)
    assert "non-finite" in nan_error.value.message


def test_uniform_logits_still_normalize_to_a_choice() -> None:
    # A degenerate uniform row must not crash or abstain: softmax normalizes
    # it to a flat conditional distribution over the substantive candidates.
    runtime = FakeRuntime([[1.0] * LOGIT_WIDTH])
    result = _evaluate(_adapter(runtime, FakeTokenizer()), _choice_request({"a": 1, "b": 2}))

    answer = result.answers["q1"]
    assert answer.type == "choice"
    assert answer.choice == "a"
    assert answer.confidence == pytest.approx(0.5, rel=1e-12)


def test_unpinned_model_override_rejected_before_any_call() -> None:
    runtime = FakeRuntime([])
    tokenizer = FakeTokenizer()
    with pytest.raises(ConfigurationError) as exc_info:
        RlcdModernBertAdapter(
            runtime=runtime,
            tokenizer=tokenizer,
            calibrator=_calibrator(),
            model="other/model",
        )
    assert exc_info.value.paid_request is False
    assert runtime.calls == 0
    assert tokenizer.batches == []


def test_unpinned_revision_override_rejected_before_any_call() -> None:
    runtime = FakeRuntime([])
    tokenizer = FakeTokenizer()
    with pytest.raises(ConfigurationError):
        RlcdModernBertAdapter(
            runtime=runtime,
            tokenizer=tokenizer,
            calibrator=_calibrator(),
            revision="deadbeef",
        )
    assert runtime.calls == 0
    assert tokenizer.batches == []


def test_request_level_model_override_rejected_before_any_call() -> None:
    runtime = FakeRuntime([])
    tokenizer = FakeTokenizer()
    adapter = _adapter(runtime, tokenizer)
    request = _abstain_request().model_copy(update={"model": "other/model"})

    with pytest.raises(ConfigurationError) as exc_info:
        _evaluate(adapter, request)
    assert exc_info.value.paid_request is False
    assert runtime.calls == 0
    assert tokenizer.batches == []


def test_pinned_request_model_is_accepted() -> None:
    runtime = FakeRuntime([_logits_row(0)])
    tokenizer = FakeTokenizer()
    adapter = _adapter(runtime, tokenizer)
    request = _abstain_request().model_copy(update={"model": REPO_ID})

    result = _evaluate(adapter, request)
    assert result.model == REPO_ID
    assert runtime.calls == 1


def test_validate_request_rejects_reserved_answer_id_collision() -> None:
    request = _choice_request({ABSTAIN_ANSWER_ID: 1, "b": 2})
    valid, issues = DecisionFacade(backend=_adapter(FakeRuntime([]), FakeTokenizer())).validate(
        request
    )
    assert not valid
    assert issues[0].code == "reserved_answer_id_collision"


def test_validate_request_rejects_marker_in_instructions() -> None:
    request = DecisionRequest(
        state="ctx",
        questions={"q1": {"type": "noul", "instructions": "<<SEP>> trap"}},
    )
    valid, issues = DecisionFacade(backend=_adapter(FakeRuntime([]), FakeTokenizer())).validate(
        request
    )
    assert not valid
    assert any(issue.code == "reserved_input_marker" for issue in issues)


def test_error_messages_never_leak_prompt_or_logit_content() -> None:
    # A distinctive long token in the malformed logits row and a distinctive
    # criterion id must both stay out of the sanitized error message.
    runtime = FakeRuntime([[LEAK_SENTINEL] * LOGIT_WIDTH])
    with pytest.raises(ProviderResponseError) as exc_info:
        _evaluate(
            _adapter(runtime, FakeTokenizer()),
            _choice_request({LEAK_SENTINEL: 1, "b": 2}),
        )
    message = exc_info.value.message
    assert LEAK_SENTINEL not in message
    assert "non-numeric" in message


def test_validate_request_helper_is_the_capability_aware_entrypoint() -> None:
    request = _choice_request({str(i): i for i in range(24)})
    result = validate_request(request, RlcdModernBertAdapter.capabilities)
    assert result.valid is True
