# RLCD ModernBERT Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `rlcd-modernbert` as a third local `bruv` backend for `noul`, `choice`, and `score` decisions, running the published RLCD ModernBERT 151M FP32 ONNX model directly. No upstream SDK, no Git/pip source dependencies, no `trust_remote_code`, no PyTorch, no Transformers, no GLiClass package. Exact upstream prompt formatting and calibration; never emit a fabricated or uncalibrated fallback result.

**Architecture:** One lazy in-process adapter behind the existing `DecisionBackend` port, registered in the existing registry as one `BackendDefinition`. Shared foundations owned by this plan: canonical `AbstainAnswer` (probability-backed mode, designed to permit later label-only mode), registry-compatible `DecisionResult.backend` string, generic `BackendCapabilities` metadata (`explicit_abstention`, `supported_total_candidates`, `reserved_input_markers`, `reserved_answer_ids`), and metadata-driven validation/gates/render/schema. Runtime pieces: pinned-revision downloader with SHA-256 enforcement over the default `huggingface_hub` cache, `tokenizers`-based tokenizer, direct ONNX Runtime session, and per-K stratified calibrator (`per_k` keys only, no global fallback). OpenRouter is out of scope; shared abstractions only must stay extensible for it.

**Tech Stack:** Python 3.11+, Pydantic 2, Typer, optional extras `onnxruntime`, `tokenizers`, `numpy`, `huggingface_hub`, pytest, Ruff, mypy, Hatchling.

**Execution constraints:** Work in the existing checkout at `/root/business/PROJECTS/bruv`. Do not create a worktree. Do not run subagents or workflows. Do not write `.memory/`. Implement behavior first, then add/update focused tests; no red-green/TDD sequencing. Sequential tasks: each task leaves the tree green and ends in a focused commit.

---

## Pinned upstream facts (source of truth)

Model repo `heman10x/rlcd-modernbert-151m`, immutable revision `8af2496eb63c7fa66d7d234e1f62629380030eb4`. Never track `main`.

Required runtime file set with enforced hashes:

| File | Size (bytes) | SHA-256 |
| --- | --- | --- |
| `model.onnx` | 606,323,181 | `4ae01f822538b000fa0e55859d4b3e6b40871d860149397e8784428b2a42ee5e` |
| `tokenizer.json` | 3,583,596 | `8bb449eb0c037aae44115b65905bb339b8f3f74eb37067c19127feb3c0755723` |
| `tokenizer_config.json` | 380 | `fb54f027372062b2ca52282efb04d178a8b57167a00cd8f4e816515823a2c016` |
| `calibrator.json` | 1,259 | `af2a876993148efa0726b6ccf710fe2303897d20c0ce8c7c9036eb50f64d23de` |

Design-time-only facts (never fetched at runtime): `config.json` gives `class_token_index` 50368, `text_token_index` 50369, `max_num_classes` 25, dtype float32. `bundle_manifest.json` gives opset 17, `max_capacity_logits` 25.

Calibrator artifact (`rlcd-calibrator-v1`): `model_id "openjev-modernbert-151.4m"`, `temperature 2.8039`, `log_temperature 1.0309995577626943`, `scope "open_domain_calibrated_v1"`, `per_k` keyed by `str(K)` for K in {2, 3, 4, 5, 6, 7, 9, 11, 17, 25}. `log_temperature` is schema-validated but never used as a temperature.

Tokenizer contract: `<<LABEL>>` → 50368, `<<SEP>>` → 50369 (added special tokens); TemplateProcessing wraps `[CLS]` (50281) + text + `[SEP]` (50282); truncation max length 512 (overrides advertised 8192); padding with `[PAD]` (50283) to longest in batch; outputs `input_ids` and `attention_mask` int64.

Prompt contract:

- Template `choice`/`score` text: `"Question: {question}\n\nContext:\n{context}"`.
- Template `noul` text: `"Context:\n{context}\n\nEvaluate proposition: {proposition}"`.
- Labels: choice `f"It is {opt.description}"`; score `f"{level.description} (Value: {level.value})"`; noul `f"true: {proposition}"` and `f"false: not {proposition}"`; abstention description always `insufficient evidence`.
- bruv `ScoreQuestion.criteria` is a plain `list[JsonValue]`; it has no `description`/`value` objects. The adapter owns the score derivation: deterministic string-index IDs (`"0"`, `"1"`, ...), each level's deterministic JSON serialization (`json.dumps(level, sort_keys=True, separators=(",", ":"), ensure_ascii=False)`) as its description text, label `f"{description} (Value: {index})"`, and the index as the numeric score value.
- Assembled prompt: `<<LABEL>>desc1<<LABEL>>desc2...<<LABEL>>insufficient evidence<<SEP>>text` (abstention label first or last is irrelevant; order of labels matches candidate order, abstention slot is part of the label set).

Calibration formula (K = total candidates including abstention slot):

1. `T = per_k[str(K)]` — explicit key only; missing key is a validation error before inference (no `exp(log_temperature)` fallback).
2. `cal_logits = logits[:K] / T`.
3. `probs = softmax(cal_logits)` then renormalize to sum exactly 1.0 (floating point).
4. Winner = argmax of `cal_logits`; upstream sentinel `__insufficient_evidence__` normalizes to canonical `__abstain__`.

Supported total candidates K: {2, 3, 4, 5, 6, 7, 9, 11, 17, 25}. Substantive choice/score counts: {2, 3, 4, 5, 6, 8, 10, 16, 24} (bruv minimum 2 substantive candidates rules out K = 2). `noul` is always K = 3. Substantive count above 24 is a validation error (no silent truncation).

Canonical result mapping: substantive answers carry renormalized conditional-on-sufficient-evidence probabilities; abstained answers carry the full distribution; the full map always travels in `AbstainAnswer.probabilities` and provider metadata. `calibrated: true` on every successful answer. Upstream `concentration` and expected-value statistics are not mapped in v1 (expected value recorded as metadata only).

---

## File map

**Create**
- `src/bruv/backends/rlcd_calibration.py` — calibrator schema, per-K loading, calibrated softmax, sentinel constants.
- `src/bruv/backends/rlcd_artifacts.py` — pinned file table, download/cache/hash verification, offline handling.
- `src/bruv/backends/rlcd_modernbert.py` — `RlcdRuntime` port, ONNX runtime, tokenizer port, prompt formatting, adapter, canonical mapping.
- `tests/unit/backends/test_rlcd_modernbert.py` — fake-runtime adapter unit tests.
- `tests/unit/backends/test_rlcd_calibration.py` — calibration math and rejection tests.
- `tests/unit/backends/test_rlcd_artifacts.py` — hash/offline/download failure tests (no network).
- `tests/contract/test_rlcd_contract.py` — backend contract and capability metadata tests.
- `tests/contract/test_abstain_contract.py` — canonical `AbstainAnswer` contract tests.
- `tests/integration/test_rlcd_cli.py` — CLI wiring with injected fake runtime.
- `tests/integration/test_rlcd_smoke.py` — real model smoke test, marked, excluded from normal CI.

**Modify**
- `src/bruv/domain/results.py` — `AbstainAnswer`, `Answer` union, `DecisionResult.backend` widening.
- `src/bruv/domain/validation.py` — new `BackendCapabilities` fields, metadata-driven validation.
- `src/bruv/backends/registry.py` — `rlcd-modernbert` definition, `runtime_packages` metadata.
- `src/bruv/backends/factory.py` — no change expected; verify registry passthrough covers the new backend.
- `src/bruv/config.py` — `rlcd_model`, `rlcd_revision` informational values.
- `src/bruv/onboarding/doctor.py` — multi-package dependency check, RLCD cache/hash/schema checks, no downloads.
- `src/bruv/onboarding/setup.py` — RLCD setup path.
- `src/bruv/output/terminal.py` — abstain rendering.
- `src/bruv/output/fields.py` — abstain answer fields.
- `src/bruv/gates.py` — abstain resolution for targeted answers.
- `src/bruv/contracts/spec.py` — `rlcd-modernbert` advertised, capability notes.
- `src/bruv/contracts/schemas.py` — regenerated from models (no hand edits beyond regeneration flow).
- `pyproject.toml` — `rlcd-modernbert` extra.
- `tests/unit/domain/test_results.py`, `tests/unit/domain/test_validation.py`, `tests/unit/backends/test_registry.py`, `tests/unit/onboarding/test_doctor.py`, `tests/unit/test_config.py`, `tests/integration/test_cli_commands.py`, `tests/integration/test_setup_doctor.py`, `tests/contract/test_contract_snapshots.py`, `tests/contract/test_machine_contracts.py`, `tests/fixtures/cli/spec.json`, `tests/fixtures/cli/output.schema.json` — extended/regenerated.
- `README.md`, `docs/installation.md`, `docs/configuration.md`, `docs/cli-reference.md`, `docs/provider-contracts/rlcd-modernbert.md` (new), `CHANGELOG.md`.

---

## Task 1: Canonical `AbstainAnswer` and registry-compatible backend field

- [ ] In `src/bruv/domain/results.py`, add module constants and `AbstainAnswer` after `ScoreAnswer`:

```python
ABSTAIN_ANSWER_ID = "__abstain__"


class AbstainAnswer(CanonicalModel):
    type: Literal["abstain"] = "abstain"
    reason: Literal["insufficient_evidence", "provider_refusal", "content_filter"]
    source_question_type: Literal["noul", "choice", "score"]
    confidence: Probability | None = None
    probabilities: (
        Annotated[dict[NonBlankString, Probability], Field(min_length=1)] | None
    ) = None
    legend: Annotated[dict[NonBlankString, JsonValue], Field(min_length=1)] | None = None

    @model_validator(mode="after")
    def validate_answer_mode(self) -> AbstainAnswer:
        probability_backed = self.confidence is not None and self.probabilities is not None
        label_only = (
            self.confidence is None and self.probabilities is None and self.legend is None
        )
        if not (probability_backed or label_only):
            raise ValueError(
                "abstain answer is either probability-backed (confidence + probabilities) "
                "or label-only (all three omitted)"
            )
        if self.legend is not None and self.probabilities is None:
            raise ValueError("legend requires probability-backed abstain answers")
        if self.probabilities is not None:
            if ABSTAIN_ANSWER_ID not in self.probabilities:
                raise ValueError("abstain probabilities must include the reserved '__abstain__' key")
            if self.probabilities[ABSTAIN_ANSWER_ID] != self.confidence:
                raise ValueError("probabilities['__abstain__'] must equal confidence")
            _validate_probability_distribution(self.probabilities)
        return self

    @model_serializer(mode="wrap")
    def serialize_answer(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        serialized = cast(dict[str, Any], handler(self))
        for optional in ("confidence", "probabilities", "legend"):
            if getattr(self, optional) is None:
                serialized.pop(optional, None)
        return serialized
```

- [ ] Add `AbstainAnswer` to the `Answer` union: `NoulAnswer | ChoiceAnswer | ScoreAnswer | AbstainAnswer` (discriminator stays `type`).
- [ ] Widen `DecisionResult.backend` from `Literal["typesafe", "simple-jev", "needle"]` to `NonBlankString` (import from `bruv.domain.questions`). Serialized values for existing backends are unchanged. This is a documented contract change.
- [ ] Update `src/bruv/output/fields.py`: add `"reason"` and `"source_question_type"` to `_ANSWER_FIELDS` (probabilities/legend/confidence already allowed).
- [ ] Update `src/bruv/output/terminal.py` to render `AbstainAnswer`:

```python
elif isinstance(answer, AbstainAnswer):
    if answer.confidence is not None:
        lines.append(
            f"{question_id}: abstained ({answer.reason}, p={answer.confidence:.3f})"
        )
    else:
        lines.append(f"{question_id}: abstained ({answer.reason})")
```

- [ ] Run checks:

```bash
cd /root/business/PROJECTS/bruv && .venv/bin/ruff format src/bruv/domain/results.py src/bruv/output/fields.py src/bruv/output/terminal.py && .venv/bin/ruff check src/bruv && .venv/bin/mypy src/bruv && .venv/bin/pytest tests/unit/domain tests/unit/output tests/contract -q
```

Expected: format clean, lint clean, types clean, all listed tests pass. Existing TypeSafe/Simple Jev/Needle payloads unchanged.

- [ ] Commit: `feat: add canonical abstain answer contract`.

## Task 2: Generic capability metadata and metadata-driven validation

- [ ] Extend `BackendCapabilities` in `src/bruv/domain/validation.py` (defaults preserve existing backends):

```python
@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    backend: str
    question_types: frozenset[str]
    calibrated: bool
    allows_json_state: bool
    explicit_abstention: bool = False
    supported_total_candidates: frozenset[int] | None = None
    reserved_input_markers: tuple[str, ...] = ()
    reserved_answer_ids: frozenset[str] = frozenset()
```

Semantics: `explicit_abstention` — backend can emit canonical `AbstainAnswer`. `supported_total_candidates` — accepted total candidate counts including any abstention slot; `None` means unconstrained. `reserved_input_markers` — literal substrings rejected in prompt-derived input text (prompt substrings only, not answer IDs). `reserved_answer_ids` — canonical answer IDs reserved for the backend's special answers.

- [ ] Add metadata-driven checks to `validate_request`, replacing backend-name conditionals for the new rules (existing `simple-jev`/`typesafe` branches stay untouched). Helper:

```python
def _prompt_text(request: DecisionRequest, question: object) -> str:
    """All text the backend would embed in its prompt for this question."""
    import json

    state = request.state
    state_text = state if isinstance(state, str) else json.dumps(
        state, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    parts: list[str] = [str(question.instructions), state_text]
    if isinstance(question, ChoiceQuestion):
        parts.extend(str(value) for value in question.criteria.values())
    elif isinstance(question, ScoreQuestion):
        parts.extend(str(level) for level in question.criteria)
    return "\n".join(parts)
```

Checks (issue codes: `reserved_input_marker`, `total_candidates_not_supported`, `reserved_answer_id_collision`; path at the offending question field):

1. Marker rejection: when `capabilities.reserved_input_markers` is nonempty, any marker substring found in `_prompt_text(...)` is a validation issue. Applies to canonical question instructions, state-derived text, option descriptions, and level descriptions.
2. Candidate limits: when `capabilities.supported_total_candidates` is not None:
   - `noul`: total = 3.
   - `choice`: total = `len(question.criteria) + 1` when `explicit_abstention` else `len(question.criteria)`.
   - `score`: same formula over `len(question.criteria)`.
   Total not in the set → validation issue before any inference.
3. Reserved answer IDs: when `capabilities.reserved_answer_ids` is nonempty, any choice option ID (key of `question.criteria`) colliding with a reserved ID is a validation issue. `noul` uses fixed IDs `true`/`false` and score uses adapter-owned string indices, so neither can collide.

- [ ] Derive score level IDs exactly as the adapter will: deterministic string indices (`str(index)` over `question.criteria`). Score level IDs are adapter-owned and never user-supplied, so reserved-answer-ID collision checks apply to choice option IDs only. Do not reference `command_builders.parse_levels`; it has no bearing on plain `JsonValue` score criteria.
- [ ] Run checks:

```bash
cd /root/business/PROJECTS/bruv && .venv/bin/ruff format src/bruv/domain/validation.py && .venv/bin/ruff check src/bruv && .venv/bin/mypy src/bruv && .venv/bin/pytest tests/unit/domain -q
```

Expected: all pass; existing backend validation behavior byte-identical (default metadata = no new checks fire).

- [ ] Commit: `feat: metadata-driven backend capability validation`.

## Task 3: Calibrator module

- [ ] Create `src/bruv/backends/rlcd_calibration.py`:

```python
"""RLCD per-K stratified calibration. per_k keys only; no global fallback."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bruv.application import ProviderResponseError

UPSTREAM_ABSTAIN_SENTINEL = "__insufficient_evidence__"
CANONICAL_ABSTAIN_ID = "__abstain__"
CALIBRATOR_SCOPE = "open_domain_calibrated_v1"


@dataclass(frozen=True, slots=True)
class RlcdCalibrator:
    model_id: str
    temperature: float
    log_temperature: float
    scope: str
    per_k: dict[int, float]
    artifact_hash: str


def load_calibrator(path: Path) -> RlcdCalibrator:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProviderResponseError(
            message="RLCD calibrator artifact is missing or corrupt.",
            paid_request=False,
            action="Re-download artifacts or reinstall 'bruv[rlcd-modernbert]'.",
        ) from None
    if raw.get("format_version") != "rlcd-calibrator-v1":
        raise ProviderResponseError(
            message="RLCD calibrator schema is unsupported.",
            paid_request=False,
            action="Use the pinned calibrator artifact for this backend revision.",
        )
    per_k: dict[int, float] = {}
    for key, value in raw.get("per_k", {}).items():
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ProviderResponseError(
                message=f"RLCD calibrator per_k[{key!r}] is not a finite number.",
                paid_request=False,
                action="Use the pinned calibrator artifact for this backend revision.",
            )
        per_k[int(key)] = float(value)
    for field in ("temperature", "log_temperature", "scope", "model_id", "artifact_hash"):
        if field not in raw:
            raise ProviderResponseError(
                message=f"RLCD calibrator is missing required field {field!r}.",
                paid_request=False,
                action="Use the pinned calibrator artifact for this backend revision.",
            )
    return RlcdCalibrator(
        model_id=str(raw["model_id"]),
        temperature=float(raw["temperature"]),
        log_temperature=float(raw["log_temperature"]),
        scope=str(raw["scope"]),
        per_k=per_k,
        artifact_hash=str(raw["artifact_hash"]),
    )


def temperature_for(calibrator: RlcdCalibrator, k: int) -> float:
    """Explicit per_k[str(K)] only. Missing key is a caller-side validation error."""
    return calibrator.per_k[k]


def calibrated_distribution(
    logits_row: "np.typing.NDArray[np.float32]", calibrator: RlcdCalibrator, k: int
) -> tuple[list[float], int]:
    """Return (renormalized probabilities over k slots, winner index)."""
    temperature = temperature_for(calibrator, k)
    cal_logits = np.asarray(logits_row[:k], dtype=np.float64) / temperature
    winner = int(np.argmax(cal_logits))
    shifted = cal_logits - cal_logits[winner]
    exps = np.exp(shifted)
    probs = exps / exps.sum()
    probs = probs / probs.sum()  # exact renormalization in floating point
    return [float(value) for value in probs], winner
```

No smoothing, clamping, or rescaling beyond the exact renormalization. `log_temperature` is loaded and validated but never applied.

- [ ] Run checks:

```bash
cd /root/business/PROJECTS/bruv && .venv/bin/ruff format src/bruv/backends/rlcd_calibration.py && .venv/bin/ruff check src/bruv && .venv/bin/mypy src/bruv
```

Expected: clean.

- [ ] Commit: `feat: add RLCD per-K calibrator loader`.

## Task 4: Artifact downloader with hash enforcement

- [ ] Create `src/bruv/backends/rlcd_artifacts.py`:

```python
"""Pinned RLCD artifact retrieval and verification. Fails closed."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from bruv.application import BackendUnavailableError, ConfigurationError

REPO_ID = "heman10x/rlcd-modernbert-151m"
REVISION = "8af2496eb63c7fa66d7d234e1f62629380030eb4"

_INSTALL_ACTION = "Install RLCD support with: pip install 'bruv[rlcd-modernbert]'"

@dataclass(frozen=True, slots=True)
class RequiredArtifact:
    name: str
    size: int
    sha256: str
    required_always: bool = True  # model.onnx may be checked only when present (doctor)

REQUIRED_ARTIFACTS: tuple[RequiredArtifact, ...] = (
    RequiredArtifact("model.onnx", 606_323_181, "4ae01f822538b000fa0e55859d4b3e6b40871d860149397e8784428b2a42ee5e"),
    RequiredArtifact("tokenizer.json", 3_583_596, "8bb449eb0c037aae44115b65905bb339b8f3f74eb37067c19127feb3c0755723"),
    RequiredArtifact("tokenizer_config.json", 380, "fb54f027372062b2ca52282efb04d178a8b57167a00cd8f4e816515823a2c016"),
    RequiredArtifact("calibrator.json", 1_259, "af2a876993148efa0726b6ccf710fe2303897d20c0ce8c7c9036eb50f64d23de"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_artifacts(repo_id: str = REPO_ID, revision: str = REVISION) -> dict[str, Path]:
    """Download (or reuse cache) then verify every required artifact. Never executes unverified files."""
    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as exc:
        if exc.name == "huggingface_hub":
            raise ConfigurationError(
                message="RLCD support is not installed.",
                paid_request=False,
                action=_INSTALL_ACTION,
            ) from None
        raise
    offline = os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in {"1", "true", "yes"}
    resolved: dict[str, Path] = {}
    try:
        from huggingface_hub.errors import (
            EntryNotFoundError,
            HfHubHTTPError,
            LocalEntryNotFoundError,
            OfflineModeIsEnabled,
        )
        import requests
    except ModuleNotFoundError as exc:
        if exc.name in {"huggingface_hub", "requests"}:
            raise ConfigurationError(
                message="RLCD support is not installed.",
                paid_request=False,
                action=_INSTALL_ACTION,
            ) from None
        raise
    expected_transport_errors = (
        EntryNotFoundError,
        HfHubHTTPError,
        LocalEntryNotFoundError,
        OfflineModeIsEnabled,
        requests.RequestException,
        OSError,
        TimeoutError,
    )
    for artifact in REQUIRED_ARTIFACTS:
        try:
            path = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    revision=revision,
                    filename=artifact.name,
                    local_files_only=offline,
                )
            )
        except expected_transport_errors as exc:
            # Exact boundary: HfHubHTTPError/RequestException cover network+HTTP failures,
            # LocalEntryNotFoundError/OfflineModeIsEnabled cover offline-missing-cache, and
            # OSError/TimeoutError cover filesystem and socket timeouts. Anything else is a
            # bug and must surface as a traceback, not a backend-unavailable error.
            if offline:
                raise BackendUnavailableError(
                    message=f"RLCD offline mode set and verified cache is missing: {artifact.name}.",
                    paid_request=False,
                    action="Clear HF_HUB_OFFLINE to download artifacts once, or pre-populate the cache.",
                ) from None
            raise BackendUnavailableError(
                message=f"RLCD artifact retrieval failed for {artifact.name}.",
                paid_request=False,
                action="Check network access to huggingface.co and retry.",
            ) from exc
        found = _sha256(path)
        if found != artifact.sha256 or path.stat().st_size != artifact.size:
            raise BackendUnavailableError(
                message=(
                    f"RLCD artifact checksum mismatch for {artifact.name}: "
                    f"expected {artifact.sha256}, found {found}."
                ),
                paid_request=False,
                action="Delete the corrupted cache entry and retry the download.",
            )
        resolved[artifact.name] = path
    return resolved


def cached_artifact_status() -> dict[str, str]:
    """Doctor-only: report hash status of cached artifacts without downloading or importing the adapter."""
        try:
            from huggingface_hub import try_to_load_from_cache
        except ModuleNotFoundError:
        return {artifact.name: "dependency-missing" for artifact in REQUIRED_ARTIFACTS}
    status: dict[str, str] = {}
    for artifact in REQUIRED_ARTIFACTS:
        path = try_to_load_from_cache(repo_id=REPO_ID, revision=REVISION, filename=artifact.name)
        if not isinstance(path, Path) or not path.is_file():
            status[artifact.name] = "missing"
            continue
        status[artifact.name] = "verified" if _sha256(path) == artifact.sha256 else "hash-mismatch"
    return status
```

- [ ] Run checks:

```bash
cd /root/business/PROJECTS/bruv && .venv/bin/ruff format src/bruv/backends/rlcd_artifacts.py && .venv/bin/ruff check src/bruv && .venv/bin/mypy src/bruv
```

Expected: clean. No network used in this task; the exact exception tuple keeps Ruff's `B902`/bleach-free lint surface clean without any `noqa`.

- [ ] Commit: `feat: add pinned RLCD artifact verification`.

## Task 5: RLCD ModernBERT adapter

- [ ] Create `src/bruv/backends/rlcd_modernbert.py` with these exact pieces:

Runtime ports and default implementations:

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class RlcdRuntime(Protocol):
    def run(self, input_ids: object, attention_mask: object) -> object: ...


class OnnxRlcdRuntime:
    """Default runtime over the verified cached model.onnx (CPU, upstream session options)."""

    def __init__(self, model_path: Path) -> None:
        import numpy as np
        import onnxruntime as ort

        self._np = np
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )

    def run(self, input_ids: object, attention_mask: object) -> object:
        return self._session.run(
            ["logits"],
            {
                "input_ids": self._np.asarray(input_ids, dtype=self._np.int64),
                "attention_mask": self._np.asarray(attention_mask, dtype=self._np.int64),
            },
        )[0]
```

Tokenizer port and default implementation (`tokenizers` library, lazy import):

```python
@runtime_checkable
class RlcdTokenizerPort(Protocol):
    def encode_batch(self, prompts: list[str]) -> tuple[object, object]:
        """Return (input_ids, attention_mask) padded to the longest sequence."""
        ...


LABEL_MARKER_ID = 50368   # <<LABEL>>
SEP_MARKER_ID = 50369     # <<SEP>>
MAX_SEQUENCE_LENGTH = 512


class TokenizersRlcdTokenizer:
    def __init__(self, tokenizer_path: Path, tokenizer_config_path: Path) -> None:
        import json

        from tokenizers import Tokenizer

        config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
        input_names = config.get("model_input_names", ["input_ids", "attention_mask"])
        if input_names[:2] != ["input_ids", "attention_mask"]:
            raise BackendUnavailableError(
                message="RLCD tokenizer_config model_input_names are unexpected.",
                paid_request=False,
                action="Use the pinned tokenizer_config.json for this backend revision.",
            )
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._tokenizer.enable_truncation(max_length=MAX_SEQUENCE_LENGTH)  # overrides 8192
        self._tokenizer.enable_padding(pad_id=50283, pad_token="[PAD]", length=None)

    def encode_batch(self, prompts: list[str]) -> tuple[object, object]:
        import numpy as np

        encodings = self._tokenizer.encode_batch(prompts)
        input_ids = np.asarray([encoding.ids for encoding in encodings], dtype=np.int64)
        attention = np.asarray([encoding.attention_mask for encoding in encodings], dtype=np.int64)
        return input_ids, attention
```

Prompt formatting (module-level, pure functions):

```python
ABSTENTION_DESCRIPTION = "insufficient evidence"

def _level_text(level: JsonValue) -> str:
    return json.dumps(level, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def format_text(question, state_text: str) -> str:
    if isinstance(question, NoulQuestion):
        return f"Context:\n{state_text}\n\nEvaluate proposition: {question.instructions}"
    return f"Question: {question.instructions}\n\nContext:\n{state_text}"


def format_prompt(labels: list[str], text: str) -> str:
    assembled = "".join(f"<<LABEL>>{label}" for label in labels)
    return f"{assembled}<<SEP>>{text}"


def candidate_spec(question, capabilities) -> tuple[list[str], list[str]]:
    """Return (labels, ids) in candidate order, abstention slot included for abstention-capable backends."""
    if isinstance(question, NoulQuestion):
        proposition = question.instructions
        ids = ["true", "false"]
        labels = [f"true: {proposition}", f"false: not {proposition}"]
    elif isinstance(question, ChoiceQuestion):
        ids = list(question.criteria)
        labels = [f"It is {question.criteria[key]}" for key in ids]
    else:  # ScoreQuestion
        ids = [str(index) for index in range(len(question.criteria))]
        labels = [
            f"{_level_text(level)} (Value: {index})"
            for index, level in enumerate(question.criteria)
        ]
    if capabilities.explicit_abstention:
        ids.append(CANONICAL_ABSTAIN_ID)
        labels.append(ABSTENTION_DESCRIPTION)
    return labels, ids
```

Score level IDs are deterministic string indices over `question.criteria`, consistent with Task 2 validation; `parse_levels` plays no role for RLCD.

Adapter:

```python
class RlcdModernBertAdapter(DecisionBackend):
    capabilities = BackendCapabilities(
        backend="rlcd-modernbert",
        question_types=frozenset({"noul", "choice", "score"}),
        calibrated=True,
        allows_json_state=True,
        explicit_abstention=True,
        supported_total_candidates=frozenset({2, 3, 4, 5, 6, 7, 9, 11, 17, 25}),
        reserved_input_markers=("<<LABEL>>", "<<SEP>>"),
        reserved_answer_ids=frozenset({"__abstain__"}),
    )

    def __init__(
        self,
        runtime: RlcdRuntime,
        tokenizer: RlcdTokenizerPort,
        calibrator: RlcdCalibrator,
        model: str = REPO_ID,
        revision: str = REVISION,
    ) -> None: ...

    def evaluate(self, request: DecisionRequest) -> DecisionResult:
        # 1. Per question: candidate_spec, format_text/format_prompt.
        # 2. One batch: every prompt tokenized in a single encode_batch call; exactly ONE
        #    self._runtime.run call per evaluate() invocation (batched, padded, truncated).
        # 3. Validate logits: finite (NaN/Inf -> ProviderResponseError), width == 25
        #    (max_capacity_logits), else ProviderResponseError; slice [:, :K] per row.
        # 4. calibrated_distribution per row with K = len(ids); K must be an explicit per_k key
        #    (guaranteed by validate_request metadata; defense-in-depth check remains).
        # 5. Map results (below).
```

Mapping rules (exact):

- Normalize the winner ID: if upstream sentinel `__insufficient_evidence__` is the winning ID (or appears in any probability map), replace with `__abstain__` at the adapter boundary. Never emit the sentinel in canonical output or metadata.
- Abstention wins (winner is abstention slot): emit `AbstainAnswer(reason="insufficient_evidence", source_question_type=<question type>, confidence=p_abstain, probabilities={canonical id: p for every candidate}, legend=<choice options or score levels legend, absent for noul>)`. No `ChoiceAnswer`/`ScoreAnswer`/`NoulAnswer` for that question; sibling questions complete normally.
- `choice` substantive win: `ChoiceAnswer(choice=<option id>, confidence=p_win, probabilities={option id: renormalized p over substantive options only})` — abstention probability dropped from this map, remaining values renormalized to sum 1.0. Legend semantics unchanged (existing `ChoiceAnswer` has no legend field).
- `score` substantive win: `ScoreAnswer(score=<float winning index>, confidence=p_win, legend={str(index): level}, probabilities={str(index): renormalized p over substantive levels})`. The adapter-owned index is the numeric score value.
- `noul` substantive win: `NoulAnswer(noul=p_true / (p_true + p_false))` — conditional on sufficient evidence; probability mode (value/confidence omitted).
- `DecisionResult(backend="rlcd-modernbert", model="heman10x/rlcd-modernbert-151m", calibrated=True, ...)`.
- `provider_metadata` per question id: `{"rlcd": {"revision": REVISION, "calibration_scope": "open_domain_calibrated_v1", "temperature_path": f"per_k:{K}", "probability_base": "conditional_on_sufficient_evidence", "full_distribution": {canonical id: p (with __abstain__)}, "expected_value_over_substantive_mass": <upstream value, recorded only>}}`. `concentration` is not mapped.
- Choice/score probability maps use canonical candidate IDs (option IDs, level IDs) — never the reserved `__abstain__` key.

Lazy construction function used by the registry (imports `huggingface_hub` implicitly via artifacts, then `onnxruntime`/`tokenizers`/`numpy` inside runtime classes):

```python
def build_adapter() -> RlcdModernBertAdapter:
    artifacts = ensure_artifacts()
    calibrator = load_calibrator(artifacts["calibrator.json"])
    runtime = OnnxRlcdRuntime(artifacts["model.onnx"])
    tokenizer = TokenizersRlcdTokenizer(artifacts["tokenizer.json"], artifacts["tokenizer_config.json"])
    return RlcdModernBertAdapter(runtime=runtime, tokenizer=tokenizer, calibrator=calibrator)
```

Error mapping (all `paid_request: false`): missing optional dependency → `ConfigurationError` with exact action `Install RLCD support with: pip install 'bruv[rlcd-modernbert]'`; download/network failure → `BackendUnavailableError`; checksum mismatch → `BackendUnavailableError` naming file, expected, found; missing/corrupt/schema-invalid calibrator → `BackendUnavailableError` or `ProviderResponseError` per Task 3 (no uncalibrated fallback); non-finite logits, unexpected logits width, shape mismatch → `ProviderResponseError`. Errors never contain prompts or request content.

- [ ] Run checks:

```bash
cd /root/business/PROJECTS/bruv && .venv/bin/ruff format src/bruv/backends/rlcd_modernbert.py && .venv/bin/ruff check src/bruv && .venv/bin/mypy src/bruv
```

Expected: clean; no imports of `onnxruntime`/`tokenizers`/`numpy`/`huggingface_hub` at module import time (verify with `python -c "import bruv.backends.rlcd_modernbert"` under the venv without the extra).

- [ ] Commit: `feat: add RLCD ModernBERT adapter`.

## Task 6: Registry, config, and dependency extra

- [ ] Register in `src/bruv/backends/registry.py`:

```python
def _build_rlcd_modernbert(context: BackendBuildContext) -> DecisionBackend:
    from bruv.backends.rlcd_modernbert import build_adapter

    return build_adapter()

BackendDefinition(
    name="rlcd-modernbert",
    capabilities=BackendCapabilities(
        backend="rlcd-modernbert",
        question_types=frozenset({"noul", "choice", "score"}),
        calibrated=True,
        allows_json_state=True,
        explicit_abstention=True,
        supported_total_candidates=frozenset({2, 3, 4, 5, 6, 7, 9, 11, 17, 25}),
        reserved_input_markers=("<<LABEL>>", "<<SEP>>"),
        reserved_answer_ids=frozenset({"__abstain__"}),
    ),
    build=_build_rlcd_modernbert,
    setup_description="local RLCD ModernBERT, calibrated. ~606 MB first-use download from Hugging Face; inference is local and free.",
    install_hint="Install RLCD support with: pip install 'bruv[rlcd-modernbert]'",
    needs_credentials=False,
    runs_local=True,
    optional_module=None,
    runtime_packages=("onnxruntime", "tokenizers", "numpy", "huggingface_hub"),
)
```

Add `runtime_packages: tuple[str, ...] = ()` to `BackendDefinition` (multi-package extra; no `optional_module` shortcut). Keep the builder lazy: registry import, config validation, spec, doctor, dry-run must never import `rlcd_modernbert` or the four runtime packages.

- [ ] Extend `src/bruv/config.py`:

```python
rlcd_model: str = "heman10x/rlcd-modernbert-151m"
rlcd_revision: str = "8af2496eb63c7fa66d7d234e1f62629380030eb4"
```

Informational in the initial release (exactly one pinned model); a revision override pointing elsewhere fails the checksum table and therefore fails closed at adapter construction. Backend validation already derives from the registry — no new branch.

- [ ] Add to `pyproject.toml` optional-dependencies:

```toml
rlcd-modernbert = [
  "numpy>=1.26,<3",
  "onnxruntime>=1.17,<2",
  "tokenizers>=0.19,<1",
  "huggingface_hub>=0.23,<2",
]
```

- [ ] Update `src/bruv/contracts/spec.py` and `src/bruv/contracts/schemas.py` in this same task so the commit stays green: `BACKEND_VALUES` is registry-derived and now includes `rlcd-modernbert` automatically; add a capability note string in the spec payload documenting RLCD probability semantics (substantive probabilities are conditional on sufficient evidence; abstained answers carry the full calibrated distribution with the reserved `__abstain__` key). `schemas.py` needs no hand edits: the output schema regenerates from `DecisionResult` (new `abstain` answer variant, widened `backend`); bump `SCHEMA_VERSION` if the project convention requires it for contract changes.
- [ ] Regenerate machine-contract snapshots from runtime-generated data, never hand-authored, and update `tests/contract/test_machine_contracts.py` expectations in the same commit:

```bash
cd /root/business/PROJECTS/bruv && .venv/bin/pytest tests/contract/test_contract_snapshots.py tests/contract/test_machine_contracts.py -q --snapshot-update
```

(Use the repository's existing snapshot-update mechanism found in `tests/contract/test_contract_snapshots.py`; if none exists, follow its documented regeneration flow.)

- [ ] Run checks:

```bash
cd /root/business/PROJECTS/bruv && .venv/bin/ruff check src/bruv && .venv/bin/mypy src/bruv && .venv/bin/pytest tests/unit/backends/test_registry.py tests/unit/test_config.py tests/contract -q
```

Expected: pass. `bruv spec` now lists `rlcd-modernbert`; snapshots updated; TypeSafe, Simple Jev, and Needle snapshot payloads byte-identical to before; config validation accepts the backend without importing runtime packages.

- [ ] Commit: `feat: register rlcd-modernbert backend and extra`.

## Task 7: Setup and lazy doctor

- [ ] Add `_rlcd_handler` in `src/bruv/onboarding/setup.py` (no credential, mirrors Needle handler):

```python
def _rlcd_handler(*, prompt, confirm, print_line, credential_path) -> SetupResult:
    from bruv.backends.registry import get_backend_definition

    install_hint = get_backend_definition("rlcd-modernbert").install_hint
    if install_hint:
        print_line(f"{install_hint} (optional; only needed if the packages are missing).")
    print_line("RLCD ModernBERT runs fully local. First use downloads about 606 MB from Hugging Face.")
    print_line("Inference is local and free. No credential needed.")
    return SetupResult(backend="rlcd-modernbert", persisted=False, next_command="bruv doctor")

_HANDLERS["rlcd-modernbert"] = _rlcd_handler
```

- [ ] Update `src/bruv/onboarding/doctor.py`. Doctor for `rlcd-modernbert` must never download, never import or construct the adapter, never create an ONNX session, never load the model:
  - `_check_dependency`: when `runtime_packages` is nonempty, check each package with `importlib.util.find_spec` and report the first missing package with the exact install hint; when empty, keep the existing `optional_module` path.
  - `_check_platform`: `runs_local=True` covers RLCD via the existing Linux/macOS/Windows x86_64/arm64 check (onnxruntime ships CPU wheels for these). Parameterize the failure fix text per backend so RLCD never prints Needle's message, e.g. derive it from the definition name:

```python
fix = (
    f"{backend} supports Linux, macOS, and Windows on x86_64 and arm64. "
    "Use the typesafe or simple-jev backend on this platform."
)
```

  (Apply the same parameterization to Needle's existing branch so both read naturally; existing Needle output text stays equivalent.)
  - `_check_cache_writable`: `runs_local=True` already runs the existing cache-parent writability check.
  - New `_check_rlcd_cache(backend)`: only when backend is `rlcd-modernbert`; calls `bruv.backends.rlcd_artifacts.cached_artifact_status()` (which imports only `huggingface_hub`, never the adapter); reports per-file name + hash status (`verified` / `hash-mismatch` / `missing`); `model.onnx` missing is a warning-style fail item with fix "run one evaluation to download artifacts"; `hash-mismatch` on any file fails with fix "delete the corrupted cache entry and retry". Never prints cache internals beyond file names and hash status. Doctor laziness is proven by tests.
- [ ] Run checks:

```bash
cd /root/business/PROJECTS/bruv && .venv/bin/ruff check src/bruv && .venv/bin/mypy src/bruv && .venv/bin/pytest tests/unit/onboarding -q
```

Expected: pass. `bruv doctor` with backend `rlcd-modernbert` performs zero network I/O.

- [ ] Commit: `feat: wire RLCD setup and lazy doctor checks`.

## Task 8: Gates and abstain-aware output
- [ ] Update `src/bruv/gates.py`: when the field path targets an answer that is an `AbstainAnswer`, resolve the gate against `AbstainAnswer` fields (e.g. `answers.<id>.confidence` resolves to the abstention probability) and mark the outcome abstained so `exit_code_for` returns 11. Concretely:

```python
# In apply_gate, before numeric resolution:
target_parts = options.field.split(".")
if len(target_parts) >= 2 and target_parts[0] == "answers" and target_parts[1] in result.answers:
    answer = result.answers[target_parts[1]]
    if getattr(answer, "type", None) == "abstain":
        return GateOutcome(abstained=True)
# Wildcard `answers.*.score` with no score answer but an abstain answer present:
# return GateOutcome(abstained=True) instead of FieldError.
```

CLI exit path already maps `abstained=True` to exit 11; no CLI change needed.

- [ ] Run checks:

```bash
cd /root/business/PROJECTS/bruv && .venv/bin/ruff check src/bruv && .venv/bin/mypy src/bruv && .venv/bin/pytest tests/unit/test_gates.py tests/unit/output -q
```

Expected: pass; existing gate behavior for non-abstain answers unchanged.

- [ ] Commit: `feat: resolve gates against abstain answers`.

## Task 9: Adapter unit tests (fake runtime, no model download)

- [ ] Create `tests/unit/backends/test_rlcd_modernbert.py` with a fake tokenizer + fake `RlcdRuntime`. Cover:
  - Exact prompt formatting for all three question types (template strings asserted literally, including `<<LABEL>>`/`<<SEP>>` assembly and the abstention description).
  - Marker safety rejection is exercised at validation level (Task 11); here assert adapter surfaces validation errors before any `run` call when markers are present.
  - K bounds: substantive counts {2,3,4,5,6,8,10,16,24} accepted for choice/score; 25+ substantive rejected before inference; K=2 total (one substantive) is unreachable, not capability-rejected: `ChoiceQuestion`/`ScoreQuestion` enforce a minimum of 2 criteria, so the test asserts request construction raises a pydantic `ValidationError`.
  - Batched single run: exactly one `run` call per `evaluate()` regardless of question count.
  - Full result mapping: choice/score/noul substantive wins (renormalized substantive maps, no `__abstain__` key in substantive maps), abstention win (`AbstainAnswer` with full distribution, `probabilities["__abstain__"] == confidence`, legend present for choice/score, absent for noul), sentinel normalization (`__insufficient_evidence__` never escapes).
  - Metadata: `temperature_path=f"per_k:{K}"`, revision, calibration scope, probability base.
  - Error paths: non-finite logits, wrong logits width, `ProviderResponseError`; `paid_request: false` on every raised error.
  - `DecisionResult.backend` accepts the registry-advertised string.
- [ ] Create `tests/unit/backends/test_rlcd_calibration.py`: table-driven over every supported per_k key {2,3,4,5,6,7,9,11,17,25}; renormalization exactness; argmax preservation; missing per_k key raises before inference (no global fallback); corrupt/missing calibrator fields raise `ProviderResponseError`.
- [ ] Create `tests/unit/backends/test_rlcd_artifacts.py`: checksum mismatch, size mismatch, offline-without-cache fail-closed, missing-dependency `ConfigurationError` with exact install action — all with monkeypatched `hf_hub_download` and temp files, zero network.
- [ ] Run:

```bash
cd /root/business/PROJECTS/bruv && .venv/bin/pytest tests/unit/backends -q
```

Expected: all pass.

- [ ] Commit: `test: cover RLCD adapter, calibration, and artifacts`.

## Task 10: `AbstainAnswer` contract tests

- [ ] Create `tests/contract/test_abstain_contract.py`:
  - Probability-backed validation: confidence + probabilities required together, finite [0,1], sum 1.0 within `PROBABILITY_SUM_ABS_TOLERANCE`, reserved `__abstain__` key present and equal to confidence.
  - Label-only validation: presence of confidence/probabilities/legend rejected (contract designed for later hosted backends; RLCD never emits this mode).
  - Legend rules: present for choice/score source, exactly covering substantive candidates; absent for noul.
  - Sentinel normalization: adapter-produced maps contain `__abstain__`, never `__insufficient_evidence__`.
  - Render/JSON: `render_human_result` abstain line; `render_success` includes the abstain answer; `get_field` resolves `answers.<id>.reason`, `.confidence`, `.probabilities.__abstain__`.
  - Gates: targeted abstained answer → `GateOutcome(abstained=True)`; `exit_code_for` → 11; sibling substantive answers unaffected.
- [ ] Run:

```bash
cd /root/business/PROJECTS/bruv && .venv/bin/pytest tests/contract/test_abstain_contract.py tests/unit/domain/test_results.py -q
```

Expected: all pass.

- [ ] Commit: `test: cover abstain answer contract`.

## Task 11: Capability metadata and validation tests

- [ ] Create `tests/contract/test_rlcd_contract.py` and extend `tests/unit/domain/test_validation.py`:
  - Metadata-driven marker rejection: a choice option description, score level description, noul proposition, or state text containing `<<LABEL>>` or `<<SEP>>` yields `reserved_input_marker` issues for `rlcd-modernbert` only; the same request is valid for `typesafe`/`simple-jev`/`needle` (default metadata unchanged).
  - Candidate limits: choice with 25 options rejected; 24 accepted; noul always valid; counts derived from `supported_total_candidates`, not backend names.
  - Reserved answer IDs: choice option named `__abstain__` rejected only for backends declaring `reserved_answer_ids`; allowed (as today) for other backends. Score level IDs are adapter-owned string indices and cannot collide, so assert no reserved-answer-ID issue is ever produced for score questions.
  - Validation happens before inference: adapter fake runtime asserts `run` never called when validation fails.
- [ ] Run:

```bash
cd /root/business/PROJECTS/bruv && .venv/bin/pytest tests/contract/test_rlcd_contract.py tests/unit/domain/test_validation.py -q
```

Expected: all pass.

- [ ] Commit: `test: cover capability metadata validation`.

## Task 12: Doctor, setup, registry, CLI integration tests

- [ ] Extend `tests/unit/onboarding/test_doctor.py` and `tests/integration/test_setup_doctor.py`:
  - Doctor for `rlcd-modernbert` never downloads, never imports `bruv.backends.rlcd_modernbert`, never constructs the adapter, never creates an ONNX session, never loads the model (assert via monkeypatch import sentinel on `bruv.backends.rlcd_modernbert`).
  - Dependency check covers all four packages individually; missing one reports the exact install hint.
  - Cache status: verified/missing/hash-mismatch branches with no network.
- [ ] Extend `tests/unit/backends/test_registry.py`: `rlcd-modernbert` present; definition matches capability constants; builder lazy.
- [ ] Extend `tests/unit/test_config.py`: `rlcd_model`/`rlcd_revision` defaults, informational precedence, `JEV_BACKEND=rlcd-modernbert` accepted.
- [ ] Create `tests/integration/test_rlcd_cli.py`: Typer CLI with injected fake backend through the registry build context — human output, JSON output, field extraction, gates, dry-run, abstain exit 11, safe errors. Existing CLI tests extended minimally for the new backend name.
- [ ] Run:

```bash
cd /root/business/PROJECTS/bruv && .venv/bin/pytest tests/unit tests/integration tests/contract -q
```

Expected: all pass.

- [ ] Commit: `test: verify RLCD onboarding and CLI contracts`.

## Task 13: Packaging, docs, real smoke, full verification

- [ ] Recreate venv with the extra and verify packaging isolation:

```bash
cd /root/business/PROJECTS/bruv && python3 -m venv --clear .venv && . .venv/bin/activate && python -m pip install -e '.[dev]' && python -m pip install -e '.[rlcd-modernbert]'
```

Expected: the extra installs only the four runtime dependencies (verify with `pip check` and `pip show onnxruntime tokenizers numpy huggingface_hub`); core `bruv` still importable without the extra (assert in `tests/packaging/test_clean_install.py` that a no-extra install lacks `onnxruntime` and errors with the exact install action).

- [ ] Write `tests/integration/test_rlcd_smoke.py`, marked `rlcd_smoke`, excluded from normal runs (`-m "not rlcd_smoke"` default via `addopts` consideration or explicit skip logic): end-to-end `evaluate()` on a small fixture after `ensure_artifacts()` hash verification; asserts calibrated outputs, real probability sums, abstention path reachable, `calibrated=true`. Include the tokenizer contract assertions (gated on verified cached artifacts, skipping with a clear reason when absent): a sample prompt tokenizes `[CLS]`-prefixed and `[SEP]`-suffixed with `<<LABEL>>` → 50368 and `<<SEP>>` → 50369, truncation to 512, batch padding to the longest sequence. Skips cleanly when artifacts are absent — never downloads in normal CI.
- [ ] Docs: README backend section; `docs/installation.md` extra instructions; `docs/configuration.md` `rlcd_model`/`rlcd_revision`; `docs/cli-reference.md` backend flag; new `docs/provider-contracts/rlcd-modernbert.md` covering pinned revision, hash table, prompt/calibration contract, probability-base disclosure (substantive = conditional on sufficient evidence), abstain contract, error mapping, offline mode; `CHANGELOG.md` entry covering the backend, the extra, `AbstainAnswer`, capability metadata, `DecisionResult.backend` widening, and the probability-base contract change.
- [ ] Disk guard before smoke: check free space; do not download when free space is under 1 GB beyond the ~606 MB requirement; report blocker instead of filling the filesystem.
- [ ] If space permits, run the real smoke:

```bash
cd /root/business/PROJECTS/bruv && .venv/bin/pytest tests/integration/test_rlcd_smoke.py -m rlcd_smoke -q -s
```

Expected: pass with real calibrated outputs. Never claim smoke success if the download or inference was skipped.
- [ ] Full verification, serially:

```bash
cd /root/business/PROJECTS/bruv && .venv/bin/ruff format --check . && .venv/bin/ruff check . && .venv/bin/mypy src/bruv && .venv/bin/pytest -m "not rlcd_smoke" --cov=bruv -q && .venv/bin/python -m build
```

Expected: every command exits 0; zero failures; coverage at project threshold; wheel and sdist build; wheel metadata shows `rlcd-modernbert` extra is optional.
- [ ] Check `git status --short` for secrets, model binaries, cache artifacts, and malformed tokens before each commit; fix any matches.
- [ ] Commit docs/smoke/verification: `docs: document RLCD ModernBERT backend` then `chore: record RLCD backend verification` (evidence appended to `docs/launch-evidence/`).
