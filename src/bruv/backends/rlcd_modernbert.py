"""RLCD ModernBERT local decision backend.

The optional ``rlcd-modernbert`` dependencies (``numpy``, ``onnxruntime``,
``tokenizers``) stay behind :class:`OnnxRlcdRuntime` and
:class:`TokenizersRlcdTokenizer`, and are imported only when those ports are
constructed. Importing this module never requires any optional dependency, so
core ``bruv`` stays installable without the extra.

Exact upstream behavior: every question is formatted with the pinned
``<<LABEL>>``/``<<SEP>>`` markers, tokenized in one batch, and scored in
exactly one ONNX run. Logits are calibrated per choice count with the pinned
calibrator, and the canonical ``__abstain__`` id replaces the upstream
sentinel everywhere. No fabricated or uncalibrated fallback is ever emitted.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, SupportsFloat, cast, runtime_checkable

from pydantic import ValidationError

from bruv.application import (
    ApplicationError,
    BackendUnavailableError,
    ConfigurationError,
    ProviderResponseError,
)
from bruv.backends.interface import DecisionBackend
from bruv.backends.rlcd_artifacts import (
    INSTALL_ACTION,
    REPO_ID,
    REVISION,
    ensure_artifacts,
)
from bruv.backends.rlcd_calibration import (
    RlcdCalibrator,
    calibrated_distribution,
    load_calibrator,
)
from bruv.domain.questions import (
    ChoiceQuestion,
    JsonValue,
    NoulQuestion,
    ScoreQuestion,
)
from bruv.domain.requests import DecisionRequest
from bruv.domain.results import (
    ABSTAIN_ANSWER_ID,
    AbstainAnswer,
    Answer,
    ChoiceAnswer,
    DecisionResult,
    NoulAnswer,
    ScoreAnswer,
)
from bruv.domain.validation import BackendCapabilities

ABSTENTION_DESCRIPTION = "insufficient evidence"
LABEL_MARKER_ID = 50368
SEP_MARKER_ID = 50369
MAX_SEQUENCE_LENGTH = 512

_LABEL_MARKER = "<<LABEL>>"
_SEP_MARKER = "<<SEP>>"
_PAD_TOKEN: str = "[PAD]"  # noqa: S105 - pinned vocabulary token, not a secret
_PAD_TOKEN_ID = 50283
_EXPECTED_LOGIT_WIDTH = 25

_ARTIFACT_ACTION = "Re-download artifacts or reinstall 'bruv[rlcd-modernbert]'."
_RUNTIME_ACTION = "Check the local RLCD runtime installation and retry."
_LOGITS_ACTION = "Re-run the evaluation or check the RLCD model output."

_SUPPORTED_TOTAL_CANDIDATES = frozenset({2, 3, 4, 5, 6, 7, 9, 11, 17, 25})


def _dependency_error() -> ConfigurationError:
    return ConfigurationError(
        message="RLCD support requires the optional rlcd-modernbert extra.",
        paid_request=False,
        action=INSTALL_ACTION,
    )


def _artifact_error(message: str) -> ProviderResponseError:
    return ProviderResponseError(
        message=message,
        paid_request=False,
        action=_ARTIFACT_ACTION,
    )


def _runtime_error(message: str) -> BackendUnavailableError:
    return BackendUnavailableError(
        message=message,
        paid_request=False,
        action=_RUNTIME_ACTION,
    )


def _artifact_unavailable(message: str) -> BackendUnavailableError:
    return BackendUnavailableError(
        message=message,
        paid_request=False,
        action=_ARTIFACT_ACTION,
    )


def _logits_error(message: str) -> ProviderResponseError:
    return ProviderResponseError(
        message=message,
        paid_request=False,
        action=_LOGITS_ACTION,
    )


def _unsupported_model_error() -> ConfigurationError:
    return ConfigurationError(
        message="RLCD supports only the pinned ModernBERT model revision.",
        paid_request=False,
        action=(f"Use model {REPO_ID!r} at revision {REVISION!r} or omit model selection."),
    )


@runtime_checkable
class RlcdRuntime(Protocol):
    """Injectable port for one batched RLCD forward pass."""

    def run(self, input_ids: object, attention_mask: object) -> object: ...


class OnnxRlcdRuntime:
    """ONNX Runtime CPU runtime for the pinned RLCD ModernBERT model.

    ``numpy`` and ``onnxruntime`` are imported lazily at construction so this
    module stays importable without the optional extra. Inputs are int64; the
    first model output (the logits tensor) is returned unchanged.
    """

    def __init__(self, model_path: Path) -> None:
        try:
            import numpy as np  # type: ignore[import-not-found]
            import onnxruntime  # type: ignore[import-not-found]
        except ImportError:
            raise _dependency_error() from None
        try:
            options = onnxruntime.SessionOptions()
            options.intra_op_num_threads = 4
            options.inter_op_num_threads = 1
            options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            session = onnxruntime.InferenceSession(
                str(model_path),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
        except Exception:
            raise _runtime_error("RLCD ONNX session could not be created.") from None
        self._np: Any = np
        self._session: Any = session

    def run(self, input_ids: object, attention_mask: object) -> object:
        np = self._np
        try:
            ids = np.asarray(input_ids, dtype=np.int64)
            mask = np.asarray(attention_mask, dtype=np.int64)
            outputs = self._session.run(
                ["logits"],
                {"input_ids": ids, "attention_mask": mask},
            )
        except Exception:
            raise _runtime_error("RLCD ONNX inference failed.") from None
        if not outputs:
            raise _runtime_error("RLCD ONNX model produced no logits output.")
        return outputs[0]


@runtime_checkable
class RlcdTokenizerPort(Protocol):
    """Injectable port for batched RLCD prompt tokenization."""

    def encode_batch(self, texts: list[str]) -> tuple[object, object]: ...


class TokenizersRlcdTokenizer:
    """HF ``tokenizers`` port matching the pinned RLCD preprocessing exactly.

    ``tokenizers`` and ``numpy`` are imported lazily. The tokenizer_config
    ``model_input_names`` must start with ``input_ids``/``attention_mask``, the
    vocabulary must carry the pinned marker and pad token ids, truncation caps
    at 512, and padding is dynamic (longest in batch) with the pinned pad id.
    """

    def __init__(self, tokenizer_path: Path, tokenizer_config_path: Path) -> None:
        try:
            import numpy as np
            from tokenizers import Tokenizer  # type: ignore[import-not-found]
        except ImportError:
            raise _dependency_error() from None
        try:
            with open(tokenizer_config_path, encoding="utf-8") as handle:
                config = json.load(handle)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            raise _artifact_error("RLCD tokenizer_config artifact is missing or corrupt.") from None
        input_names = config.get("model_input_names") if isinstance(config, dict) else None
        if input_names is None:
            input_names = ["input_ids", "attention_mask"]
        elif not isinstance(input_names, list) or input_names[:2] != [
            "input_ids",
            "attention_mask",
        ]:
            raise _artifact_unavailable("RLCD tokenizer_config model_input_names are unexpected.")
        try:
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
        except Exception:
            raise _artifact_error("RLCD tokenizer artifact could not be loaded.") from None
        if (
            tokenizer.token_to_id(_LABEL_MARKER) != LABEL_MARKER_ID
            or tokenizer.token_to_id(_SEP_MARKER) != SEP_MARKER_ID
            or tokenizer.token_to_id(_PAD_TOKEN) != _PAD_TOKEN_ID
        ):
            raise _artifact_error("RLCD tokenizer vocabulary is missing required special tokens.")
        tokenizer.enable_truncation(max_length=MAX_SEQUENCE_LENGTH)
        tokenizer.enable_padding(pad_id=_PAD_TOKEN_ID, pad_token=_PAD_TOKEN)
        self._tokenizer: Any = tokenizer
        self._np: Any = np

    def encode_batch(self, texts: list[str]) -> tuple[object, object]:
        np = self._np
        encodings = self._tokenizer.encode_batch(texts)
        input_ids = np.array([encoding.ids for encoding in encodings], dtype=np.int64)
        attention_mask = np.array(
            [encoding.attention_mask for encoding in encodings], dtype=np.int64
        )
        return input_ids, attention_mask


def _level_text(value: JsonValue) -> str:
    """Serialize a JSON value deterministically in compact sorted form."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _state_text(state: JsonValue) -> str:
    """String state passes through; everything else becomes deterministic JSON."""
    if isinstance(state, str):
        return state
    return _level_text(state)


@dataclass(frozen=True, slots=True)
class _Candidate:
    """One scored answer candidate: its canonical id and prompt label."""

    id: str
    label: str


_ABSTAIN_CANDIDATE = _Candidate(ABSTAIN_ANSWER_ID, ABSTENTION_DESCRIPTION)


def _candidate_list(
    question: NoulQuestion | ChoiceQuestion | ScoreQuestion,
) -> list[_Candidate]:
    """Build candidates in fixed order: substantive first, canonical abstain last."""
    if isinstance(question, NoulQuestion):
        substantive = [
            _Candidate("true", f"true: {question.instructions}"),
            _Candidate("false", f"false: not {question.instructions}"),
        ]
    elif isinstance(question, ChoiceQuestion):
        substantive = [
            _Candidate(str(key), f"It is {value}") for key, value in question.criteria.items()
        ]
    elif isinstance(question, ScoreQuestion):
        substantive = [
            _Candidate(str(index), f"{_level_text(level)} (Value: {index})")
            for index, level in enumerate(question.criteria)
        ]
    else:
        raise TypeError(f"unsupported question type: {type(question).__name__}")
    return [*substantive, _ABSTAIN_CANDIDATE]


def _question_text(
    question: NoulQuestion | ChoiceQuestion | ScoreQuestion,
    state: JsonValue,
) -> str:
    state_text = _state_text(state)
    if isinstance(question, NoulQuestion):
        return f"Context:\n{state_text}\n\nEvaluate proposition: {question.instructions}"
    return f"Question: {question.instructions}\n\nContext:\n{state_text}"


def _prompt_for(candidates: list[_Candidate], text: str) -> str:
    parts = [f"{_LABEL_MARKER}{candidate.label}" for candidate in candidates]
    parts.append(f"{_SEP_MARKER}{text}")
    return "".join(parts)


def _legend_values(
    question: NoulQuestion | ChoiceQuestion | ScoreQuestion,
) -> dict[str, JsonValue]:
    """Map substantive candidate ids to their original caller-supplied values."""
    if isinstance(question, ChoiceQuestion):
        return {str(key): value for key, value in question.criteria.items()}
    if isinstance(question, ScoreQuestion):
        return {str(index): level for index, level in enumerate(question.criteria)}
    return {}


def _logits_rows(logits: object, batch_size: int) -> list[list[float]]:
    """Validate the raw model output shape and finiteness; return float rows.

    The output must be a 2D batch with exactly one row per question and a
    width of exactly 25. Errors never include logits or prompt content.
    """
    if isinstance(logits, (str, bytes, bytearray)):
        raise _logits_error("RLCD model output was not a 2D batch.")
    try:
        rows = list(cast("Iterable[object]", logits))
    except TypeError:
        raise _logits_error("RLCD model output was not a 2D batch.") from None
    if len(rows) != batch_size:
        raise _logits_error("RLCD model output row count did not match the question count.")

    parsed: list[list[float]] = []
    for row in rows:
        if isinstance(row, (str, bytes, bytearray)):
            raise _logits_error("RLCD model output was not a 2D batch.")
        try:
            values = list(cast("Iterable[object]", row))
        except TypeError:
            raise _logits_error("RLCD model output was not a 2D batch.") from None
        if len(values) != _EXPECTED_LOGIT_WIDTH:
            raise _logits_error("RLCD model output width was unexpected.")
        parsed_row: list[float] = []
        for value in values:
            if isinstance(value, bool):
                raise _logits_error("RLCD logits row contained non-numeric values.")
            try:
                number = float(cast("SupportsFloat", value))
            except (TypeError, ValueError, OverflowError):
                raise _logits_error("RLCD logits row contained non-numeric values.") from None
            if not math.isfinite(number):
                raise _logits_error("RLCD logits row contained non-finite values.")
            parsed_row.append(number)
        parsed.append(parsed_row)
    return parsed


def _conditional_mass(probs: list[float]) -> list[float]:
    """Drop abstain, renormalize the substantive mass; fail closed on zero."""
    total = math.fsum(probs)
    if not math.isfinite(total) or total <= 0.0:
        raise _logits_error("RLCD produced no usable substantive probability mass.")
    normalized = [prob / total for prob in probs]
    if not all(math.isfinite(prob) for prob in normalized):
        raise _logits_error("RLCD produced no usable substantive probability mass.")
    return normalized


def _map_answer(
    question: NoulQuestion | ChoiceQuestion | ScoreQuestion,
    candidates: list[_Candidate],
    probs: list[float],
    winner: int,
) -> tuple[Answer, dict[str, float], float | None]:
    """Map one calibrated row to a canonical answer.

    Probability keys are always canonical; the upstream abstain sentinel never
    escapes. Returns the answer, the full distribution including
    ``__abstain__``, and the score expected value when the question is a score.
    """
    abstain_index = len(candidates) - 1
    full_distribution = {
        candidate.id: float(prob) for candidate, prob in zip(candidates, probs, strict=True)
    }
    expected_value: float | None = None

    if winner == abstain_index:
        legend: dict[str, JsonValue] | None = None
        if not isinstance(question, NoulQuestion):
            legend = {
                **_legend_values(question),
                ABSTAIN_ANSWER_ID: ABSTENTION_DESCRIPTION,
            }
        answer: Answer = AbstainAnswer(
            reason="insufficient_evidence",
            source_question_type=question.type,
            confidence=probs[abstain_index],
            probabilities=full_distribution,
            legend=legend,
        )
        return answer, full_distribution, expected_value

    substantive_probs = _conditional_mass(probs[:abstain_index])
    substantive_ids = [candidate.id for candidate in candidates[:abstain_index]]

    if isinstance(question, NoulQuestion):
        answer = NoulAnswer(noul=substantive_probs[0])
    elif isinstance(question, ChoiceQuestion):
        selected = candidates[winner].id
        answer = ChoiceAnswer(
            choice=selected,
            confidence=substantive_probs[winner],
            probabilities=dict(zip(substantive_ids, substantive_probs, strict=True)),
        )
    elif isinstance(question, ScoreQuestion):
        answer = ScoreAnswer(
            score=float(winner),
            confidence=substantive_probs[winner],
            legend=_legend_values(question),
            probabilities=dict(zip(substantive_ids, substantive_probs, strict=True)),
        )
        expected_value = math.fsum(
            float(index) * prob for index, prob in enumerate(substantive_probs)
        )
    else:
        raise TypeError(f"unsupported question type: {type(question).__name__}")
    return answer, full_distribution, expected_value


_CAPABILITIES = BackendCapabilities(
    backend="rlcd-modernbert",
    question_types=frozenset({"noul", "choice", "score"}),
    calibrated=True,
    allows_json_state=True,
    explicit_abstention=True,
    supported_total_candidates=_SUPPORTED_TOTAL_CANDIDATES,
    reserved_input_markers=(_LABEL_MARKER, _SEP_MARKER),
    reserved_answer_ids=frozenset({ABSTAIN_ANSWER_ID}),
)


class RlcdModernBertAdapter(DecisionBackend):
    """Calibrated local RLCD ModernBERT backend.

    All questions are tokenized in one batch and scored in exactly one runtime
    run. Only the pinned model revision is accepted, checked before any
    inference. All errors are sanitized: prompt, state, and logits content
    never enters an error message.
    """

    capabilities = _CAPABILITIES

    def __init__(
        self,
        runtime: RlcdRuntime,
        tokenizer: RlcdTokenizerPort,
        calibrator: RlcdCalibrator,
        *,
        model: str = REPO_ID,
        revision: str = REVISION,
    ) -> None:
        if model != REPO_ID or revision != REVISION:
            raise _unsupported_model_error()
        self._runtime = runtime
        self._tokenizer = tokenizer
        self._calibrator = calibrator

    def evaluate(self, request: DecisionRequest) -> DecisionResult:
        if request.model is not None and request.model != REPO_ID:
            raise _unsupported_model_error()

        prepared: list[
            tuple[str, NoulQuestion | ChoiceQuestion | ScoreQuestion, list[_Candidate]]
        ] = []
        prompts: list[str] = []
        for question_id, question in request.questions.items():
            candidates = _candidate_list(question)
            prompts.append(_prompt_for(candidates, _question_text(question, request.state)))
            prepared.append((question_id, question, candidates))

        try:
            encoded = self._tokenizer.encode_batch(prompts)
        except ApplicationError:
            raise
        except Exception:
            raise _runtime_error("RLCD tokenization failed.") from None
        try:
            input_ids, attention_mask = encoded
        except (TypeError, ValueError):
            raise _runtime_error("RLCD tokenizer returned an unexpected structure.") from None

        # Exactly one forward pass for the whole request.
        try:
            logits = self._runtime.run(input_ids=input_ids, attention_mask=attention_mask)
        except ApplicationError:
            raise
        except Exception:
            raise _runtime_error("RLCD model inference failed.") from None

        rows = _logits_rows(logits, len(prepared))

        answers: dict[str, Answer] = {}
        provider_metadata: dict[str, JsonValue] = {}
        try:
            for (question_id, question, candidates), row in zip(prepared, rows, strict=True):
                k = len(candidates)
                probabilities, winner = calibrated_distribution(row, self._calibrator, k)
                answer, full_distribution, expected_value = _map_answer(
                    question, candidates, probabilities, winner
                )
                answers[question_id] = answer
                rlcd_payload: dict[str, JsonValue] = {
                    "revision": REVISION,
                    "calibration_scope": self._calibrator.scope,
                    "temperature_path": f"per_k:{k}",
                    "probability_base": "conditional_on_sufficient_evidence",
                    "full_distribution": cast("JsonValue", full_distribution),
                }
                if expected_value is not None:
                    rlcd_payload["expected_value_over_substantive_mass"] = expected_value
                provider_metadata[question_id] = {"rlcd": rlcd_payload}
        except ValidationError as exc:
            raise ProviderResponseError(
                message="RLCD output did not map to a canonical answer.",
                paid_request=False,
                action=_LOGITS_ACTION,
            ) from exc

        return DecisionResult(
            backend="rlcd-modernbert",
            model=REPO_ID,
            calibrated=True,
            answers=answers,
            usage=None,
            provider_metadata=provider_metadata,
        )


def build_adapter() -> RlcdModernBertAdapter:
    """Construct the real RLCD adapter from pinned, verified artifacts."""
    artifacts = ensure_artifacts()
    calibrator = load_calibrator(artifacts["calibrator.json"])
    runtime = OnnxRlcdRuntime(artifacts["model.onnx"])
    tokenizer = TokenizersRlcdTokenizer(
        artifacts["tokenizer.json"],
        artifacts["tokenizer_config.json"],
    )
    return RlcdModernBertAdapter(runtime=runtime, tokenizer=tokenizer, calibrator=calibrator)


__all__ = [
    "ABSTENTION_DESCRIPTION",
    "LABEL_MARKER_ID",
    "MAX_SEQUENCE_LENGTH",
    "SEP_MARKER_ID",
    "OnnxRlcdRuntime",
    "RlcdModernBertAdapter",
    "RlcdRuntime",
    "RlcdTokenizerPort",
    "TokenizersRlcdTokenizer",
    "build_adapter",
]
