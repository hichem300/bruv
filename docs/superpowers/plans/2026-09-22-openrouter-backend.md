# OpenRouter Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `openrouter` as a paid hosted `bruv` backend for `noul`, `choice`, and `score` decisions over the OpenRouter Chat Completions API with strict structured outputs. One plain `httpx` POST per `evaluate()` call. No SDK, no workspace Classifiers, no default model, no `openrouter/auto`, no retries, no healing, no fabricated confidence. Abstention is explicit through the shared canonical `AbstainAnswer` in label-only mode.

**This plan owns:** the shared label-only substantive answer modes (`ChoiceAnswer.confidence`/`ScoreAnswer.confidence` optional, `NoulAnswer` value-only mode) and the hosted OpenRouter adapter. The shared canonical `AbstainAnswer`, the registry-compatible `DecisionResult.backend` field, and the generic `BackendCapabilities` fields arrive from the completed RLCD foundation (prerequisite below).

**Architecture:** One `BackendDefinition` registered in the existing registry; one adapter module behind `DecisionBackend` speaking a narrow injectable transport port over `httpx`. One batched nonstreaming POST carrying every question, `response_format` `json_schema` with `strict: true` and literal per-question properties, `provider.require_parameters`, privacy defaults `data_collection="deny"` and `zdr=true`. Strict one-shot JSON parsing; malformed output fails. Stable HTTP/body/refusal/content-filter error mapping with conservative `paid_request` send-state accounting. Label-only canonical answers with `calibrated=false`; schema-selected abstentions and provider refusal/content-filter paths map to label-only `AbstainAnswer`.

**Tech Stack:** Python 3.11+, Pydantic 2, Typer, core `httpx` (already a core dependency; no new dependency extra), pytest with a hand-rolled fake transport, pytest-httpx available in `dev`, Ruff, mypy, Hatchling.

**Execution constraints:** Work in the existing checkout. Do not create a worktree. Implement behavior first, then add and run tests; no red-green/TDD sequencing. Use subagents sequentially with `openai-codex/gpt-5.6-sol`; delegated agents must not launch subagents or workflows. Do not touch `.memory/`.

---

## Prerequisite: completed RLCD foundation

The `rlcd-modernbert` plan must be completed and committed before Task 1. Verify with:

```bash
cd /root/business/PROJECTS/bruv
grep -n "class AbstainAnswer" src/bruv/domain/results.py
grep -n "__abstain__" src/bruv/domain/results.py src/bruv/domain/validation.py
grep -n "explicit_abstention\|reserved_answer_ids" src/bruv/domain/validation.py
grep -rn "rlcd" docs/superpowers/plans/ --include="*.md" -l
.venv/bin/python -m pytest tests/contract -q
```

Expected: `AbstainAnswer` exists with both probability-backed and label-only modes validated; `Answer` union includes it; `BackendCapabilities` declares `explicit_abstention` and `reserved_answer_ids`; `validate_request` already rejects user-supplied option/level IDs colliding with `reserved_answer_ids`; `DecisionResult.backend` accepts registry-advertised nonblank names; all contract tests pass. **If any check fails, stop and complete the RLCD plan first. Do not re-implement shared contracts here.**

---

## File map

**Create**
- `src/bruv/backends/openrouter.py` — transport port, resolved-model helper, strict JSON schema builder, request body, response parsing, canonical mapping, abstention paths, error mapping, metadata sanitation.
- `tests/unit/backends/test_openrouter.py` — fake-transport adapter coverage (schema, precedence, strict parsing, abstention, errors, metadata).
- `tests/contract/test_openrouter_contract.py` — canonical backend contract and label-only JSON shapes.
- `tests/integration/test_openrouter_live_smoke.py` — opt-in paid smoke test (Task 17 only).

**Modify**
- `src/bruv/domain/results.py` — `ChoiceAnswer.confidence` and `ScoreAnswer.confidence` optional; `NoulAnswer` value-only mode; serializers omit inactive fields only.
- `src/bruv/config.py` — `openrouter_model` (no default), `openrouter_data_collection`, `openrouter_zdr` with validation.
- `src/bruv/onboarding/credentials.py` — `OPENROUTER_API_KEY` support alongside TypeSafe.
- `src/bruv/backends/registry.py` — `openrouter` `BackendDefinition`, builder, `credential_env` metadata.
- `src/bruv/onboarding/doctor.py` — credential-env-generic checks, hosted endpoint text, opt-in non-paid `GET /key` check.
- `src/bruv/onboarding/setup.py` — OpenRouter handler.
- `src/bruv/output/terminal.py` — label-only and abstain rendering.
- `src/bruv/output/fields.py` — abstain answer field paths.
- `src/bruv/gates.py` — abstain-targeted gates resolve exit 11.
- `src/bruv/contracts/spec.py` — backend capability notes.
- `src/bruv/cli.py` — `openrouter` accepted wherever backend names are declared (drive from registry where possible).
- `tests/fixtures/cli/spec.json`, `tests/fixtures/cli/output.schema.json`, `tests/contract/test_machine_contracts.py` — machine contract refresh.
- `tests/unit/test_config.py`, `tests/unit/onboarding/`, `tests/integration/test_cli_commands.py`, `tests/integration/test_setup_doctor.py`, `tests/integration/test_cli_errors.py` — behavior coverage.
- `README.md`, `docs/cli-reference.md`, `docs/configuration.md`, `CHANGELOG.md` — backend, model rule, privacy defaults, paid policy, label-only abstain mode.
- `pyproject.toml` — `live` pytest marker registration only; no dependency changes.

**Explicitly excluded:** OpenRouter workspace Classifiers, streaming, retries/idempotency/replay, response healing/fence stripping/json-repair, default models or `openrouter/auto`, model-reported confidence, per-question calls, provider ranking beyond `require_parameters`. No dependency extra: core `httpx` suffices.

---

## Task 1: Prerequisite verification

- [ ] Run the prerequisite verification block above.
- [ ] Confirm `tests/contract/test_contract_snapshots.py` passes unchanged (baseline byte-compatibility).
- [ ] Confirm `grep -n "backend: " src/bruv/domain/results.py` shows the registry-compatible nonblank backend field (no closed `Literal`).

Expected outcome: all checks pass with no code changes. If any fails, stop.

No commit (verification only).

## Task 2: Label-only substantive answer modes

- [ ] Make `ChoiceAnswer.confidence` and `ScoreAnswer.confidence` optional in `src/bruv/domain/results.py`; enforce exactly one mode; keep serializers omitting inactive fields only.

```python
class ChoiceAnswer(CanonicalModel):
    type: Literal["choice"] = "choice"
    choice: NonBlankString
    confidence: Probability | None = None
    probabilities: (
        Annotated[dict[NonBlankString, Probability], Field(min_length=1)] | None
    ) = None

    @model_validator(mode="after")
    def validate_answer_mode(self) -> ChoiceAnswer:
        probability_mode = self.confidence is not None and self.probabilities is not None
        label_only_mode = self.confidence is None and self.probabilities is None
        if not (probability_mode or label_only_mode):
            raise ValueError(
                "choice answer requires confidence and probabilities together, or neither"
            )
        if self.probabilities is not None:
            if self.choice not in self.probabilities:
                raise ValueError("choice must have a matching probability")
            _validate_probability_distribution(self.probabilities)
        return self

    @model_serializer(mode="wrap")
    def serialize_answer(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        serialized = cast(dict[str, Any], handler(self))
        if self.confidence is None:
            serialized.pop("confidence", None)
        if self.probabilities is None:
            serialized.pop("probabilities", None)
        return serialized
```

`ScoreAnswer` mirrors this exactly: `confidence: Probability | None = None`; the mode validator requires confidence and probabilities together or both absent; when `probabilities` is present, legend keys must equal probability keys and the distribution check runs; the serializer pops `confidence` when `None` in addition to the existing `probabilities` pop.

- [ ] Add the `NoulAnswer` value-only mode; preserve probability mode and value-plus-confidence selection mode.

```python
    @model_validator(mode="after")
    def validate_answer_mode(self) -> NoulAnswer:
        probability_mode = self.noul is not None and self.value is None and self.confidence is None
        selection_mode = (
            self.noul is None and self.value is not None and self.confidence is not None
        )
        value_only_mode = self.noul is None and self.value is not None and self.confidence is None
        if not (probability_mode or selection_mode or value_only_mode):
            raise ValueError(
                "noul answer must contain noul, value with confidence, or value only"
            )
        return self
```

The existing `NoulAnswer` serializer already pops inactive fields; verify it pops `confidence` when `None` in the value-only mode and extend it if not.

- [ ] Run behavior verification with existing suites (byte-compatibility proof, no new tests yet):

```bash
cd /root/business/PROJECTS/bruv
.venv/bin/python -m pytest tests/contract/test_contract_snapshots.py tests/contract/test_simple_jev_contract.py tests/contract/test_typesafe_contract.py -q
.venv/bin/python -m pytest tests/unit -q
```

Expected: all pass unchanged. TypeSafe, Simple Jev, Needle, and RLCD payloads stay byte-identical because those backends still supply their current fields; serializers omit only inactive fields. If any existing test asserts the exact old `NoulAnswer`/`ChoiceAnswer`/`ScoreAnswer` mode error text, update only the asserted message to the new wording; behavior assertions stay unchanged.

Commit: `feat: label-only canonical answer modes`

## Task 3: OpenRouter config and model precedence

- [ ] Add config values to `AppConfig` in `src/bruv/config.py`:

```python
    openrouter_model: str | None = None
    openrouter_data_collection: Literal["deny", "allow"] = "deny"
    openrouter_zdr: bool = True

    @field_validator("openrouter_model")
    @classmethod
    def _validate_openrouter_model(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value.strip():
            raise ValueError("openrouter_model must not be blank")
        if value.strip() != value:
            raise ValueError("openrouter_model must not be blank-padded")
        if value == "openrouter/auto":
            raise ValueError("openrouter_model must be a concrete model, not openrouter/auto")
        return value
```

`backend = "openrouter"` validates automatically through the existing registry-driven `_registered_backend` validator; `openrouter_model` itself has no default and no fallback.

- [ ] Add the precedence helper in `src/bruv/backends/openrouter.py` (created in this task with just this helper plus module docstring; the adapter arrives in Tasks 5–7):

```python
RESERVED_ROUTER_ALIAS = "openrouter/auto"


def require_concrete_model(model: str, *, source: str) -> None:
    """Reject blank, blank-padded, and router-passthrough model ids before any request."""
    if not model.strip():
        raise ConfigurationError(
            message=f"{source} must not be blank.",
            paid_request=False,
            action="Set a concrete model such as \"openai/gpt-4o-mini\".",
        )
    if model.strip() != model:
        raise ConfigurationError(
            message=f"{source} must not be blank-padded.",
            paid_request=False,
            action="Set a concrete model without surrounding whitespace.",
        )
    if model == RESERVED_ROUTER_ALIAS:
        raise ConfigurationError(
            message=f"{source} must be a concrete model, not {RESERVED_ROUTER_ALIAS}.",
            paid_request=False,
            action="Set a concrete model such as \"openai/gpt-4o-mini\".",
        )
```

- [ ] Verify: `cd /root/business/PROJECTS/bruv && .venv/bin/python -m pytest tests/unit/test_config.py -q` — existing tests pass; add focused `test_config.py` cases afterward (behavior-first rule): config fallback, `openrouter/auto` rejection in both places, blank-padded rejection, explicit `request.model` override precedence.

Commit: `feat: openrouter config and model precedence`

## Task 4: OpenRouter credentials

- [ ] Extend `Credentials` and parsing in `src/bruv/onboarding/credentials.py`:

```python
CREDENTIAL_ENVIRONMENTS = ("TYPESAFE_API_KEY", "OPENROUTER_API_KEY")


@dataclass(frozen=True, slots=True)
class Credentials:
    typesafe_api_key: str | None = None
    openrouter_api_key: str | None = None

    @property
    def has_typesafe(self) -> bool:
        return bool(self.typesafe_api_key)

    @property
    def has_openrouter(self) -> bool:
        return bool(self.openrouter_api_key)

    def has_for(self, env_name: str) -> bool:
        if env_name == "OPENROUTER_API_KEY":
            return self.has_openrouter
        if env_name == "TYPESAFE_API_KEY":
            return self.has_typesafe
        return False
```

- [ ] `load_credentials`: read `OPENROUTER_API_KEY` from the environment before the credential file, mirroring the existing TypeSafe flow. Generalize `_parse_credential_text` to return `dict[str, str]` keyed by env name for both `TYPESAFE_API_KEY=` and `OPENROUTER_API_KEY=` lines.
- [ ] Change `save_credentials` to `save_credentials(env_name: str, key: str, *, path: Path | None = None)`, rejecting unknown names with `ConfigurationError`, merging both keys into the existing restricted file (`_write_restricted` writes sorted `NAME="value"` lines), preserving mode `0600` and Windows ACL behavior. Keep the secret value out of every error message.
- [ ] Update the TypeSafe call site in `src/bruv/onboarding/setup.py` to `save_credentials("TYPESAFE_API_KEY", api_key, path=credential_path)`.
- [ ] Verify: `cd /root/business/PROJECTS/bruv && .venv/bin/python -m pytest tests/unit/onboarding tests/integration/test_setup_doctor.py -q` — existing tests pass; then add focused cases for the second env name, merged persistence, and unchanged TypeSafe behavior.

Commit: `feat: openrouter credential storage`

## Task 5: Register the openrouter backend

- [ ] Add `credential_env: str | None = None` to `BackendDefinition` in `src/bruv/backends/registry.py`; set `credential_env="TYPESAFE_API_KEY"` on the typesafe definition.
- [ ] Register the new definition and builder:

```python
def _build_openrouter(context: BackendBuildContext) -> DecisionBackend:
    # Import in two steps: the helper exists from Task 3, the adapter only from
    # Task 7. Model and credential checks must run before the adapter import so
    # Task 5 tests never hit an ImportError.
    from bruv.backends.openrouter import require_concrete_model

    configured_model = context.config.openrouter_model
    if configured_model is None:
        raise ConfigurationError(
            message="openrouter_model is not configured.",
            paid_request=False,
            action='Set openrouter_model in bruv.toml, e.g. openrouter_model = "openai/gpt-4o-mini".',
        )
    require_concrete_model(configured_model, source="openrouter_model")
    if not context.credentials.has_openrouter:
        raise ConfigurationError(
            message="No OpenRouter API key found.",
            paid_request=False,
            action="Set OPENROUTER_API_KEY or run `bruv setup`.",
        )
    dependency = context.selected_dependency()
    from bruv.backends.openrouter import OpenRouterAdapter, OpenRouterTransport

    transport = (
        cast(OpenRouterTransport, dependency)
        if dependency is not None
        else httpx.Client(
            base_url="https://openrouter.ai/api/v1",
            follow_redirects=False,
            verify=True,
            timeout=context.config.request_timeout_seconds,
        )
    )
    return OpenRouterAdapter(
        transport=transport,
        api_key=context.credentials.openrouter_api_key or "",
        configured_model=configured_model,
        data_collection=context.config.openrouter_data_collection,
        zdr=context.config.openrouter_zdr,
    )
```

```python
        BackendDefinition(
            name="openrouter",
            capabilities=BackendCapabilities(
                backend="openrouter",
                question_types=frozenset({"noul", "choice", "score"}),
                calibrated=False,
                allows_json_state=True,
                explicit_abstention=True,
                reserved_answer_ids=frozenset({"__abstain__"}),
            ),
            build=_build_openrouter,
            setup_description="hosted, uncalibrated, paid. Needs OPENROUTER_API_KEY.",
            needs_credentials=True,
            credential_env="OPENROUTER_API_KEY",
        ),
```

No install hint: `httpx` is already a core dependency. `runs_local` stays `False`. No backend-name branches outside this registry: config validation, CLI acceptance, setup, doctor, and specs all read registry metadata.

- [ ] Verify: `cd /root/business/PROJECTS/bruv && .venv/bin/python -m pytest tests/unit/backends tests/unit/test_application.py -q` — existing tests pass; add focused registry tests after (backend listed, missing key/model produce `ConfigurationError` with `paid_request=False`).

Commit: `feat: register openrouter backend`

## Task 6: Adapter — strict schema, request body, privacy routing

- [ ] Create `src/bruv/backends/openrouter.py`. Add the transport port and capability constant:

```python
"""OpenRouter Chat Completions adapter.

One nonstreaming authenticated POST per ``evaluate()`` call with strict
structured outputs. Plain ``httpx`` only: no SDK, no streaming, no retry, no
response healing. OpenRouter workspace Classifiers are explicitly out of scope.
"""

from __future__ import annotations

import json
import time
from typing import Any, Protocol

import httpx
from pydantic import TypeAdapter

from bruv.application import (
    BackendUnavailableError,
    ConfigurationError,
    ProviderResponseError,
    UsageError,
)
from bruv.domain.questions import ChoiceQuestion, NoulQuestion, ScoreQuestion
from bruv.domain.requests import DecisionRequest
from bruv.domain.results import (
    AbstainAnswer,
    Answer,
    ChoiceAnswer,
    DecisionResult,
    NoulAnswer,
    ScoreAnswer,
    Usage,
)
from bruv.domain.validation import BackendCapabilities

CAPABILITIES = BackendCapabilities(
    backend="openrouter",
    question_types=frozenset({"noul", "choice", "score"}),
    calibrated=False,
    allows_json_state=True,
    explicit_abstention=True,
    reserved_answer_ids=frozenset({"__abstain__"}),
)

ABSTAIN_ID = "__abstain__"
ABSTAIN_REASONS = ("insufficient_evidence", "provider_refusal", "content_filter")


class OpenRouterTransport(Protocol):
    """Narrow port around ``httpx.Client.post`` so tests inject fakes."""

    def post(
        self, url: str, *, json: dict[str, Any], headers: dict[str, str]
    ) -> httpx.Response: ...
```

- [ ] Add the strict per-request JSON schema with literal question-ID properties and abstention:

```python
def _question_schema(question: Any) -> dict[str, Any]:
    if isinstance(question, NoulQuestion):
        candidates = ["true", "false"]
    elif isinstance(question, ChoiceQuestion):
        candidates = list(question.criteria)
    elif isinstance(question, ScoreQuestion):
        candidates = [level_id for level_id, _value in _score_levels(question)]
    else:
        raise TypeError(f"unsupported question type: {type(question).__name__}")
    return {
        "type": "object",
        "properties": {
            "selection": {"type": "string", "enum": [*candidates, ABSTAIN_ID]},
            "abstain_reason": {"type": "string", "enum": list(ABSTAIN_REASONS)},
        },
        "required": ["selection"],
        "additionalProperties": False,
    }


def _response_format(request: DecisionRequest) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "bruv_decisions",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    question_id: _question_schema(question)
                    for question_id, question in request.questions.items()
                },
                "required": list(request.questions),
                "additionalProperties": False,
            },
        },
    }
```

`abstain_reason` stays optional in the JSON Schema (no `allOf`/`if/then`): the adapter parser enforces the exact pairing instead. Abstain without a valid `abstain_reason`, or an `abstain_reason` beside a substantive selection, both fail in `_to_answers` (Task 7); Task 13 asserts the request schema contains no conditional keywords and that the parser enforces the pairing.

- [ ] Add score-level derivation and the user content builder:

```python
def _score_levels(question: ScoreQuestion) -> list[tuple[str, float]]:
    """Derive (level_id, numeric_value) pairs for score criteria.

    Dict levels use their ``id`` and numeric ``value`` when present; scalar
    numeric levels become ``level_<index>``. Non-numeric values are a request
    error before any send.
    """
    levels: list[tuple[str, float]] = []
    for index, level in enumerate(question.criteria):
        if isinstance(level, dict):
            level_id = level.get("id")
            raw_value = level.get("value")
            level_id_str = str(level_id) if isinstance(level_id, str) and level_id.strip() else f"level_{index}"
        else:
            level_id_str = f"level_{index}"
            raw_value = level
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise ValueError(f"score level {index} has no numeric value")
        levels.append((level_id_str, float(raw_value)))
    return levels


def _json_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _question_brief(question: Any) -> str:
    """Render actual candidate IDs with descriptions or score values."""
    if isinstance(question, NoulQuestion):
        return "candidates: true, false"
    if isinstance(question, ChoiceQuestion):
        options = "; ".join(
            f"{option_id}: {_json_text(description)}"
            for option_id, description in question.criteria.items()
        )
        return f"candidates: {options}"
    if isinstance(question, ScoreQuestion):
        levels = "; ".join(
            f"{level_id}: {_json_text(value)}" for level_id, value in _score_levels(question)
        )
        return f"levels: {levels}"
    raise TypeError(f"unsupported question type: {type(question).__name__}")


def _user_content(request: DecisionRequest) -> str:
    parts = [f"State:\n{_json_text(request.state)}"]
    for question_id, question in request.questions.items():
        parts.append(
            f"Question {question_id} ({question.type}): {question.instructions}\n"
            f"{_question_brief(question)}"
        )
    return "\n\n".join(parts)
```

`_question_brief` reuses the same helpers as the schema builder (`question.criteria` keys for choice, `_score_levels` for score), so prompt text, enum identifiers, and the score legend stay consistent; descriptions and instructions never alter enum identifiers.

- [ ] Add the request body with privacy defaults and provider routing:

```python
def _request_body(
    request: DecisionRequest,
    model: str,
    *,
    data_collection: str,
    zdr: bool,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": _user_content(request)}],
        "response_format": _response_format(request),
        "provider": {
            "require_parameters": True,
            "data_collection": data_collection,
            "zdr": zdr,
        },
    }
```

Privacy routing: `data_collection="deny"` and `zdr=true` are defaults from config; only explicit config relaxes them; the relaxed values travel in result metadata (Task 7). Routing failure because no provider honors the settings is surfaced as a backend-unavailable error (Task 8); no privacy setting is ever dropped to make routing succeed. During implementation, verify the `zdr` placement against https://openrouter.ai/docs/guides/routing/provider-selection and adjust nesting if the API expects it outside `provider`.

- [ ] Verify: `cd /root/business/PROJECTS/bruv && .venv/bin/python -m pytest tests/unit -q` — no adapter tests yet; module imports cleanly (`.venv/bin/python -c "import bruv.backends.openrouter"`).

Commit (with Task 7): `feat: openrouter adapter with strict schema and abstention`

## Task 7: Adapter — response parsing, canonical mapping, abstention

- [ ] Add strict one-shot response parsing. Exactly one parse attempt; no fence stripping, no healing, no retry, no second relaxed attempt:

```python
class _ParsedResponse:
    __slots__ = ("content", "refusal_reason", "usage_payload", "cost", "request_id")

    def __init__(
        self,
        content: dict[str, Any] | None,
        refusal_reason: str | None,
        usage_payload: dict[str, Any] | None,
        cost: float | None,
        request_id: str | None,
    ) -> None:
        self.content = content
        self.refusal_reason = refusal_reason
        self.usage_payload = usage_payload
        self.cost = cost
        self.request_id = request_id


def _parse_response(response: httpx.Response) -> _ParsedResponse:
    try:
        body = response.json()
    except ValueError as exc:
        raise ProviderResponseError(
            message="OpenRouter response was not valid JSON.",
            paid_request=True,
            action="Retry as a new request; report the response if it persists.",
        ) from exc
    if not isinstance(body, dict):
        raise ProviderResponseError(
            message="OpenRouter response must be a JSON object.",
            paid_request=True,
            action="Retry as a new request; report the response if it persists.",
        )
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ProviderResponseError(
            message="OpenRouter response contained no choices.",
            paid_request=True,
            action="Retry as a new request; report the response if it persists.",
        )
    message = choices[0].get("message")
    finish_reason = choices[0].get("finish_reason")
    refusal_reason: str | None = None
    if finish_reason == "content_filter":
        refusal_reason = "content_filter"
    elif isinstance(message, dict) and isinstance(message.get("refusal"), str) and message["refusal"].strip():
        refusal_reason = "provider_refusal"
    content: dict[str, Any] | None = None
    if isinstance(message, dict) and message.get("content") is not None:
        raw_content = message["content"]
        if isinstance(raw_content, str):
            try:
                raw_content = json.loads(raw_content)
            except ValueError as exc:
                raise ProviderResponseError(
                    message="OpenRouter structured content was not valid JSON.",
                    paid_request=True,
                    action="Retry as a new request; report the response if it persists.",
                ) from exc
        if not isinstance(raw_content, dict):
            raise ProviderResponseError(
                message="OpenRouter structured content was not a JSON object.",
                paid_request=True,
                action="Retry as a new request; report the response if it persists.",
            )
        content = raw_content
    usage_payload = body.get("usage") if isinstance(body.get("usage"), dict) else None
    cost = usage_payload.get("cost") if usage_payload else None
    cost_value = float(cost) if isinstance(cost, (int, float)) and not isinstance(cost, bool) else None
    request_id = body.get("id")
    request_id_value = request_id if isinstance(request_id, str) and request_id.strip() else None
    return _ParsedResponse(content, refusal_reason, usage_payload, cost_value, request_id_value)
```

- [ ] Add canonical mapping. Label-only substantive answers plus schema-selected and refusal/content-filter abstentions; siblings never aborted:

```python
def _substantive_answer(question: Any, question_id: str, selection: Any) -> Answer:
    if not isinstance(selection, str):
        raise ProviderResponseError(
            message=f"Question '{question_id}' selection is outside the schema enum.",
            paid_request=True,
            action="Retry as a new request; report the response if it persists.",
        )
    if isinstance(question, NoulQuestion):
        if selection not in ("true", "false"):
            raise ProviderResponseError(
                message=f"Question '{question_id}' selection is outside the schema enum.",
                paid_request=True,
                action="Retry as a new request; report the response if it persists.",
            )
        return NoulAnswer(value=selection == "true")
    if isinstance(question, ChoiceQuestion):
        if selection not in question.criteria:
            raise ProviderResponseError(
                message=f"Question '{question_id}' selection is outside the schema enum.",
                paid_request=True,
                action="Retry as a new request; report the response if it persists.",
            )
        return ChoiceAnswer(choice=selection)
    if isinstance(question, ScoreQuestion):
        levels = _score_levels(question)
        values = dict(levels)
        if selection not in values:
            raise ProviderResponseError(
                message=f"Question '{question_id}' selection is outside the schema enum.",
                paid_request=True,
                action="Retry as a new request; report the response if it persists.",
            )
        return ScoreAnswer(
            score=values[selection],
            legend={level_id: value for level_id, value in levels},
        )
    raise TypeError(f"unsupported question type: {type(question).__name__}")


def _to_answers(request: DecisionRequest, parsed: _ParsedResponse) -> dict[str, Answer]:
    answers: dict[str, Answer] = {}
    for question_id, question in request.questions.items():
        entry = parsed.content.get(question_id) if parsed.content is not None else None
        if not isinstance(entry, dict):
            if parsed.refusal_reason is not None:
                answers[question_id] = AbstainAnswer(
                    reason=parsed.refusal_reason,
                    source_question_type=question.type,
                )
                continue
            raise ProviderResponseError(
                message=f"OpenRouter response missing structured answer for question '{question_id}'.",
                paid_request=True,
                action="Retry as a new request; report the response if it persists.",
            )
        selection = entry.get("selection")
        abstain_reason = entry.get("abstain_reason")
        if selection == ABSTAIN_ID:
            if abstain_reason not in ABSTAIN_REASONS:
                raise ProviderResponseError(
                    message=f"Abstained answer for question '{question_id}' missing a valid abstain_reason.",
                    paid_request=True,
                    action="Retry as a new request; report the response if it persists.",
                )
            answers[question_id] = AbstainAnswer(
                reason=abstain_reason,
                source_question_type=question.type,
            )
            continue
        if abstain_reason is not None:
            raise ProviderResponseError(
                message=f"Question '{question_id}' included abstain_reason with a substantive selection.",
                paid_request=True,
                action="Retry as a new request; report the response if it persists.",
            )
        answers[question_id] = _substantive_answer(question, question_id, selection)
    extra = set(parsed.content or {}) - set(request.questions)
    if extra:
        raise ProviderResponseError(
            message="OpenRouter structured content contained unknown question keys.",
            paid_request=True,
            action="Retry as a new request; report the response if it persists.",
        )
    return answers
```

Missing question keys are a provider-response error only when no refusal/content-filter marker exists; with such a marker, affected questions become label-only `AbstainAnswer` with reason `provider_refusal` or `content_filter` and siblings with structured content complete normally.

- [ ] Add usage/cost metadata sanitation and the adapter class. Token counts and cost (when present) go to provider metadata; the API key, credentials, and authorization headers never enter metadata, logs, or errors (existing redaction covers them):

```python
_USAGE_ADAPTER: TypeAdapter[Usage] = TypeAdapter(Usage)


class OpenRouterAdapter:
    """Hosted structured-output adapter over the OpenRouter Chat Completions API."""

    capabilities = CAPABILITIES

    def __init__(
        self,
        *,
        transport: OpenRouterTransport,
        api_key: str,
        configured_model: str,
        data_collection: str = "deny",
        zdr: bool = True,
    ) -> None:
        self._transport = transport
        self._api_key = api_key
        self._configured_model = configured_model
        self._data_collection = data_collection
        self._zdr = zdr

    def evaluate(self, request: DecisionRequest) -> DecisionResult:
        model = request.model if request.model is not None else self._configured_model
        require_concrete_model(model, source="model")
        try:
            body = _request_body(
                request,
                model,
                data_collection=self._data_collection,
                zdr=self._zdr,
            )
        except ValueError as exc:
            # Non-numeric score levels and similar request-shape problems fail
            # before any send with paid_request=False.
            raise UsageError(
                message=str(exc),
                paid_request=False,
                action="Fix the request and retry.",
            ) from exc
        started = time.monotonic()
        response = self._send(body)
        latency_ms = (time.monotonic() - started) * 1000.0
        parsed = _parse_response(response)
        answers = _to_answers(request, parsed)
        metadata: dict[str, Any] = {
            "privacy": {"data_collection": self._data_collection, "zdr": self._zdr},
        }
        usage = None
        if parsed.usage_payload is not None:
            try:
                usage = _USAGE_ADAPTER.validate_python(
                    {
                        "input_tokens": parsed.usage_payload.get("prompt_tokens"),
                        "output_tokens": parsed.usage_payload.get("completion_tokens"),
                    }
                )
            except ValidationError as exc:
                raise ProviderResponseError(
                    message="OpenRouter usage payload did not match the canonical schema.",
                    paid_request=True,
                    action="Retry as a new request; report the response if it persists.",
                ) from exc
            token_counts = {
                key: parsed.usage_payload[key]
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                if isinstance(parsed.usage_payload.get(key), int)
            }
            if token_counts:
                metadata["tokens"] = token_counts
        if parsed.cost is not None:
            metadata["cost"] = parsed.cost
        return DecisionResult(
            backend="openrouter",
            model=model,
            calibrated=False,
            answers=answers,
            usage=usage,
            latency_ms=latency_ms,
            request_id=parsed.request_id,
            provider_metadata=metadata,
        )

    def _send(self, body: dict[str, Any]) -> httpx.Response:
        headers = {"Authorization": f"Bearer {self._api_key}"}
        try:
            return self._transport.post("/chat/completions", json=body, headers=headers)
        except httpx.ConnectTimeout as exc:
            raise BackendUnavailableError(
                message="OpenRouter connection timed out before the request was sent.",
                paid_request=False,
                action="Check network connectivity and retry.",
            ) from exc
        except httpx.ConnectError as exc:
            raise BackendUnavailableError(
                message="OpenRouter connection could not be established before the request was sent.",
                paid_request=False,
                action="Check network connectivity and DNS resolution and retry.",
            ) from exc
        except httpx.TransportError as exc:
            # All remaining transport failures (ReadTimeout, ReadError, TLS
            # failures after connect): send state is not knowable, so
            # paid_request stays conservatively true.
            raise ProviderResponseError(
                message="OpenRouter transport failed; the request may already have been sent.",
                paid_request=True,
                action="Retry as a new request; never assume the first request was free.",
            ) from exc
```

Note `Usage` fields are `input_tokens`/`output_tokens` (`NonNegativeInt | None`), so a missing side maps to `None` naturally; a `ValidationError` from `_USAGE_ADAPTER` (negative or non-integer token counts) is wrapped into `ProviderResponseError` with `paid_request=True` as shown. Add `ValidationError` to the `pydantic` import from Task 6 (`from pydantic import TypeAdapter, ValidationError`).

The resolved model travels unchanged in `body["model"]` and is recorded as `DecisionResult.model`. `request.model` overrides `configured_model` per request. Every successful answer reports `calibrated=false`; confidence and probabilities are never solicited or synthesized; reserved `__abstain__` never appears inside `ChoiceAnswer`, `ScoreAnswer`, or `NoulAnswer`.

- [ ] Verify: `cd /root/business/PROJECTS/bruv && .venv/bin/python -c "import bruv.backends.openrouter" && .venv/bin/python -m pytest tests/unit -q`

Commit: `feat: openrouter adapter with strict schema and abstention`

## Task 8: Error mapping and paid-request lifecycle

- [ ] Add `_raise_for_status` with the body-level `error` override:

```python
def _body_error(response: httpx.Response) -> dict[str, Any] | None:
    try:
        payload = response.json()
    except ValueError:
        return None
    if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
        return payload["error"]
    return None


def _safe_body_message(error_payload: dict[str, Any] | None) -> str | None:
    """Surface a truncated, sanitized body error message; never echo secrets."""
    if error_payload is None:
        return None
    message = error_payload.get("message")
    if not isinstance(message, str) or not message.strip():
        return None
    cleaned = message.strip()[:300]
    code = error_payload.get("code")
    return f"{code}: {cleaned}" if isinstance(code, (str, int)) else cleaned
```

```python
    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        status = response.status_code
        if status < 400:
            return
        error_payload = _body_error(response)
        safe_detail = _safe_body_message(error_payload)
        if status == 401:
            raise ConfigurationError(
                message="OpenRouter rejected the API key as invalid or missing.",
                paid_request=False,
                action="Set a valid OPENROUTER_API_KEY or run `bruv setup`.",
            )
        if status == 402:
            raise BackendUnavailableError(
                message=safe_detail or "OpenRouter credits are exhausted.",
                paid_request=True,
                action="Add credits at https://openrouter.ai/credits and retry as a new request.",
            )
        if status == 403:
            raise BackendUnavailableError(
                message=safe_detail or "OpenRouter blocked the key or flagged the request.",
                paid_request=True,
                action="Review the OpenRouter dashboard; retry as a new request.",
            )
        if status == 408:
            raise ProviderResponseError(
                message=safe_detail or "OpenRouter reported a request timeout.",
                paid_request=True,
                action="Retry as a new request; the sent request may have been billed.",
            )
        if status == 429:
            raise BackendUnavailableError(
                message=safe_detail or "OpenRouter rate limited the request.",
                paid_request=True,
                action="Wait before retrying as a new request.",
            )
        if status in (502, 503):
            raise BackendUnavailableError(
                message=safe_detail or "OpenRouter had no available provider for the request settings.",
                paid_request=True,
                action="Retry later or adjust the model; privacy settings are never relaxed to route.",
            )
        if status == 400:
            raise ProviderResponseError(
                message=safe_detail or "OpenRouter rejected the request as invalid.",
                paid_request=True,
                action="Fix the request and retry as a new request.",
            )
        raise ProviderResponseError(
            message=safe_detail or f"OpenRouter returned status {status}.",
            paid_request=True,
            action="Retry as a new request; report the error if it persists.",
        )
```

Call `_raise_for_status(response)` in `evaluate` between `_send` and `_parse_response`. `paid_request` semantics: `false` in every error raised before send (validation, config, missing key, connect-timeout, DNS/TLS transport failure); conservatively `true` for every error after send, including malformed responses, 408, 429, 5xx, and body-level errors. There is no retry, no idempotency key, no replay: the next attempt is a new user-initiated request. 401 is the documented pre-send configuration-error exception (`paid_request=false`). A body-level `error` object overrides generic status text with its sanitized `code` and `message`.

- [ ] Verify: `cd /root/business/PROJECTS/bruv && .venv/bin/python -m pytest tests/unit -q` — then add focused error-mapping tests in `tests/unit/backends/test_openrouter.py` (table-driven over all statuses, body-error override, transport failures, `paid_request` transitions).

Commit: `feat: openrouter error mapping and paid-request lifecycle`

## Task 9: Doctor diagnostics

- [ ] Make credential and endpoint checks registry-driven in `src/bruv/onboarding/doctor.py`:

```python
def _check_credentials(creds: Credentials, backend: str) -> DiagnosticResult:
    definition = get_backend_definition(backend)
    if not definition.needs_credentials:
        return DiagnosticResult(name="credentials", ok=True, message=f"not required for {backend}")
    env_name = definition.credential_env
    if env_name is not None and creds.has_for(env_name):
        return DiagnosticResult(name="credentials", ok=True, message="present")
    return DiagnosticResult(
        name="credentials",
        ok=False,
        message=f"no API key found for {backend}",
        fix=f"Set {env_name} or run `bruv setup`." if env_name else "Run `bruv setup`.",
    )
```

Add `_check_endpoint` handling for registry-hosted defaults without backend-name branches: add `default_endpoint: str | None = None` to `BackendDefinition`, set `default_endpoint="https://openrouter.ai/api/v1"` on the openrouter definition, and report `using {default_endpoint}` as `ok` when the configured backend has no local endpoint; keep the reachability skip for hosted defaults. For `openrouter`, the config-validity check must include the model rule (nonblank, not blank-padded, not `openrouter/auto`) by calling `require_concrete_model(config.openrouter_model, source="openrouter_model")` inside a try/except that downgrades `ConfigurationError` to a failed diagnostic with its action as `fix`.

- [ ] Add the opt-in, never-paid key check:

```python
def _check_openrouter_key(
    creds: Credentials,
    *,
    enabled: bool,
    fetch: Callable[[str, str], dict[str, Any]] | None,
) -> DiagnosticResult:
    if not enabled:
        return DiagnosticResult(name="openrouter_key", ok=True, message="skipped (not requested)")
    if not creds.has_openrouter:
        return DiagnosticResult(
            name="openrouter_key",
            ok=False,
            message="no OpenRouter API key found",
            fix="Set OPENROUTER_API_KEY or run `bruv setup`.",
        )
    if fetch is None:
        return DiagnosticResult(name="openrouter_key", ok=True, message="key present (live check unavailable)")
    try:
        payload = fetch("https://openrouter.ai/api/v1/key", creds.openrouter_api_key or "")
    except Exception:  # noqa: BLE001
        return DiagnosticResult(
            name="openrouter_key",
            ok=False,
            message="key check request failed",
            fix="Check connectivity and retry. This check is free and never sends a paid request.",
        )
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, dict):
        return DiagnosticResult(
            name="openrouter_key",
            ok=False,
            message="unexpected key-check response",
            fix="Retry later. This check is free and never sends a paid request.",
        )
    return DiagnosticResult(name="openrouter_key", ok=True, message="key valid; credit data returned")
```

Extend `run_doctor(...)` with `check_openrouter_key: bool = False` and an injectable `key_fetch` callable; append the result only when `backend == "openrouter"` (this one branch is acceptable as doctor-level opt-in wiring, not adapter logic). Doctor never sends a paid Chat Completions request; the key check is the documented free `GET https://openrouter.ai/api/v1/key`.

- [ ] Verify: `cd /root/business/PROJECTS/bruv && .venv/bin/python -m pytest tests/integration/test_setup_doctor.py tests/unit/onboarding -q` — then add focused doctor tests (key/config checks, opt-in gate, no paid request, secret values never in `DiagnosticResult`).

Commit: `feat: openrouter doctor diagnostics`

## Task 10: Setup onboarding

- [ ] Add the handler and register it in `_HANDLERS` in `src/bruv/onboarding/setup.py`:

```python
def _openrouter_handler(
    *,
    prompt: PromptFn,
    confirm: ConfirmFn,
    print_line: PrintFn,
    credential_path: Path | None,
) -> SetupResult:
    print_line("OpenRouter is a paid hosted backend: request content is sent to OpenRouter and upstream providers.")
    print_line("Privacy defaults: data_collection=deny, zdr=true. Content is not used for training or retained.")
    print_line("A concrete model is required (openrouter_model); openrouter/auto is not allowed.")
    api_key = prompt("OpenRouter API key (input hidden)")
    if not api_key.strip():
        raise ConfigurationError(
            message="an API key is required for the openrouter backend",
            paid_request=False,
            action="Set OPENROUTER_API_KEY or re-run setup with a key.",
        )
    model = prompt("OpenRouter model (e.g. openai/gpt-4o-mini)").strip()
    if not model or model != model.strip() or model == "openrouter/auto":
        raise ConfigurationError(
            message="a concrete openrouter_model is required",
            paid_request=False,
            action='Set openrouter_model in bruv.toml, e.g. openrouter_model = "openai/gpt-4o-mini".',
        )
    if not confirm("Persist the key to a protected credential file?"):
        print_line("Skipping persistence. Set OPENROUTER_API_KEY in your shell.")
        return SetupResult(backend="openrouter", persisted=False, next_command="bruv doctor")
    save_credentials("OPENROUTER_API_KEY", api_key, path=credential_path)
    return SetupResult(backend="openrouter", persisted=True, next_command="bruv doctor")
```

The backend listing loop already renders every registry entry generically; the new definition appears automatically. Note the setup model prompt is informational guidance; `openrouter_model` in `bruv.toml` remains the source of truth.

- [ ] Verify: `cd /root/business/PROJECTS/bruv && .venv/bin/python -m pytest tests/integration/test_setup_doctor.py -q` — then add the openrouter setup tests (paid warning line, privacy defaults line, key persistence, model rule).

Commit: `feat: openrouter setup onboarding`

## Task 11: Output, fields, and gates for label-only answers

- [ ] Update `src/bruv/output/terminal.py` for label-only and abstain rendering:

```python
    for question_id, answer in result.answers.items():
        if answer.type == "abstain":
            lines.append(f"{question_id}: abstained ({answer.reason})")
        elif isinstance(answer, NoulAnswer):
            if answer.noul is not None:
                lines.append(f"{question_id}: {answer.noul:.3f}")
            elif answer.confidence is not None:
                lines.append(
                    f"{question_id}: {str(answer.value).lower()} "
                    f"(confidence: {answer.confidence:.3f})"
                )
            else:
                lines.append(f"{question_id}: {str(answer.value).lower()}")
        elif isinstance(answer, ChoiceAnswer):
            if answer.confidence is not None:
                lines.append(f"{question_id}: {answer.choice} (confidence: {answer.confidence:.3f})")
            else:
                lines.append(f"{question_id}: {answer.choice} (uncalibrated)")
        elif isinstance(answer, ScoreAnswer):
            if answer.confidence is not None:
                lines.append(f"{question_id}: {answer.score:.3f} (confidence: {answer.confidence:.3f})")
            else:
                lines.append(f"{question_id}: {answer.score:.3f} (uncalibrated)")
```

Import `AbstainAnswer` (or branch on `answer.type == "abstain"` to avoid importing it). An abstained targeted answer renders an explicit abstention notice with the canonical reason; human output for openrouter shows `calibration: uncalibrated` through the existing `calibrated` marker.

- [ ] Update `src/bruv/output/fields.py` `_ANSWER_FIELDS` to include `"value"`, `"reason"`, and `"source_question_type"` so `--field answers.<id>.reason` and abstain gates resolve. Keep private/unknown field rejection unchanged.
- [ ] Update `src/bruv/gates.py`: gates targeting an abstained answer resolve against `AbstainAnswer` fields and produce exit 11 (the existing abstain band):

```python
class _AbstainedTarget(Exception):
    """Internal signal: the targeted answer abstained."""


def _resolve_numeric(result: DecisionResult, field: str) -> float:
    path = field
    if path.endswith(".*.score"):
        for key in result.answers:
            answer = result.answers[key]
            if answer.type == "abstain":
                continue
            if getattr(answer, "type", None) == "score":
                return float(get_field(result, f"answers.{key}.score"))
        if result.answers and all(item.type == "abstain" for item in result.answers.values()):
            raise _AbstainedTarget()
        raise FieldError("no score answer available for wildcard field")
    parts = path.split(".")
    if len(parts) == 3 and parts[0] == "answers" and parts[2] == "score":
        answer = result.answers.get(parts[1])
        if answer is not None and answer.type == "abstain":
            raise _AbstainedTarget()
    value = get_field(result, path)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise FieldError(f"field '{field}' is not numeric")
    return float(value)


def apply_gate(result: DecisionResult, options: GateOptions) -> GateOutcome:
    if options.fail_under is None and options.abstain_band is None:
        return GateOutcome()
    try:
        value = _resolve_numeric(result, options.field)
    except _AbstainedTarget:
        return GateOutcome(abstained=True, value=None)
    if options.abstain_band is not None:
        low, high = options.abstain_band
        if low <= value <= high:
            return GateOutcome(abstained=True, value=value)
    if options.fail_under is not None and value < options.fail_under:
        return GateOutcome(abstained=False, gate_passed=False, value=value)
    return GateOutcome(abstained=False, gate_passed=True, value=value)
```

Confidence gates cannot be expressed against openrouter answers (no confidence exists); score gates keep operating on the selected level value. The CLI already maps `GateOutcome(abstained=True)` to exit 11.

- [ ] Verify: `cd /root/business/PROJECTS/bruv && .venv/bin/python -m pytest tests/unit/test_gates.py tests/unit/output tests/integration/test_cli_commands.py -q` — then add abstain rendering, abstain-field, and abstain-gate tests.

Commit: `feat: label-only output, fields, and abstain gates`

## Task 12: CLI, spec, and machine contracts

- [ ] Inspect backend-name handling: `cd /root/business/PROJECTS/bruv && grep -n "backend" src/bruv/cli.py src/bruv/command_builders.py | head -30`. If `cli.py` declares a static backend list or `Literal`, extend it with `openrouter` or drive it from `bruv.backends.registry.backend_names`. `command_builders.py` is expected backend-agnostic; confirm and leave unchanged otherwise.
- [ ] Update `src/bruv/contracts/spec.py`: `BACKEND_VALUES` already derives from `backend_names` and picks up `openrouter` automatically. Add capability notes:

```python
BACKEND_NOTES = {
    "openrouter": {
        "calibrated": False,
        "runs_local": False,
        "needs_credentials": True,
        "explicit_abstention": True,
        "answer_semantics": "label-only; no confidence or probabilities; explicit abstention with canonical reasons",
    },
}
```

and include `"backend_notes": BACKEND_NOTES` in the `spec()` return. Document in `PAID_REQUEST_SEMANTICS`-adjacent wording that openrouter errors carry conservative post-send `paid_request=true`.

- [ ] Refresh machine contracts: `src/bruv/contracts/schemas.py` output schema updates automatically from the widened answer models; regenerate/update `tests/fixtures/cli/output.schema.json` and `tests/fixtures/cli/spec.json` in the same sorted-compact style, and update `tests/contract/test_machine_contracts.py` expectations for the new fields (`backend_notes`, abstain answer, optional confidence).
- [ ] Verify: `cd /root/business/PROJECTS/bruv && .venv/bin/python -m pytest tests/contract tests/integration -q` — snapshot tests for TypeSafe, Simple Jev, Needle, and RLCD remain unchanged and passing; only machine fixtures change for the new shared modes.

Commit: `feat: openrouter CLI spec and machine contracts`

## Task 13: Adapter unit tests (fake transport)

- [ ] Create `tests/unit/backends/test_openrouter.py`. Behavior is already implemented; write focused tests now:

```python
import json

import httpx
import pytest

from bruv.backends.openrouter import (
    CAPABILITIES,
    ABSTAIN_ID,
    OpenRouterAdapter,
    require_concrete_model,
)
from bruv.application import (
    BackendUnavailableError,
    ConfigurationError,
    ProviderResponseError,
)
from bruv.domain.requests import DecisionRequest


class FakeTransport:
    def __init__(self, response: httpx.Response) -> None:
        self.calls: list[dict[str, object]] = []
        self._response = response

    def post(self, url, *, json, headers):
        self.calls.append({"url": url, "json": json, "headers": headers})
        return self._response


def _response(payload: dict, status: int = 200) -> httpx.Response:
    return httpx.Response(status_code=status, json=payload, request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"))


def _adapter(transport: FakeTransport) -> OpenRouterAdapter:
    return OpenRouterAdapter(
        transport=transport,
        api_key="test-key",
        configured_model="openai/gpt-4o-mini",
    )
```

Cover (each a small test): exact request body (`model`, `response_format` with `strict: true` and literal per-question properties with `additionalProperties: false`, `provider.require_parameters`, `data_collection="deny"`, `zdr=true`); exactly one nonstreaming POST per `evaluate()`; `Authorization: Bearer test-key` present and the key absent from metadata; precedence (`request.model` override, config fallback, `require_concrete_model` rejecting blank/padded/`openrouter/auto` in either source); label-only mappings for `noul`/`choice`/`score` with `calibrated=false` and no confidence/probabilities; strict parsing failures (fenced string content, trailing prose, extra keys, missing keys, selection outside enum, abstention without `abstain_reason`, abstain_reason with substantive selection) each raising `ProviderResponseError` with `paid_request=True` and exactly one parse attempt; schema-selected abstention preserving siblings; refusal (`message.refusal`) and `finish_reason == "content_filter"` mapping affected questions to label-only `AbstainAnswer` with canonical reasons; table-driven error mapping over statuses 400/401/402/403/408/429/502/503 plus body-level `error` override plus `httpx.ConnectTimeout`/`httpx.ConnectError` (knowable pre-send: `BackendUnavailableError`, `paid_request=False`) and `httpx.ReadTimeout` plus other `httpx.TransportError` failures (post-connect send state unknowable: `ProviderResponseError`, `paid_request=True`), asserting the conservative paid boundary per Task 7; invalid usage payloads (negative or non-integer token counts) wrapped as `ProviderResponseError` with `paid_request=True`; the request schema contains no `allOf`/`if/then` with `abstain_reason` optional in the schema while the parser enforces the abstain pairing; `_user_content` renders each question's actual candidate IDs with descriptions or score values; metadata sanitation (tokens, cost when present, privacy disclosure of relaxed settings).

- [ ] Run: `cd /root/business/PROJECTS/bruv && .venv/bin/python -m pytest tests/unit/backends/test_openrouter.py -q` — all pass.

Commit: `test: openrouter adapter and contract coverage` (with Task 14)

## Task 14: Contract tests

- [ ] Create `tests/contract/test_openrouter_contract.py`: canonical JSON output shapes for openrouter results (label-only answers omit `confidence`/`probabilities`/`noul`; abstain answers carry only `type`/`reason`/`source_question_type`; `backend="openrouter"`, `calibrated=false`); label-only `AbstainAnswer` validation rejects `confidence`, `probabilities`, and `legend`; probability-backed `AbstainAnswer` validation unchanged for RLCD; reserved-ID collision rejection (`__abstain__` as an option or level ID is a validation error from capabilities metadata, not a name branch).
- [ ] Create `tests/unit/` mode tests for `ChoiceAnswer`/`ScoreAnswer`/`NoulAnswer`: probability-backed mode unchanged, label-only mode validated, partial/contradictory combinations rejected.
- [ ] Run: `cd /root/business/PROJECTS/bruv && .venv/bin/python -m pytest tests/contract tests/unit -q` — all pass, existing backend snapshots byte-identical.

Commit: `test: openrouter adapter and contract coverage`

## Task 15: Integration tests

- [ ] Extend `tests/integration/test_cli_commands.py`: `--backend openrouter` in dry-run/validate paths; `--field` access on label-only answers and abstain fields; exit 11 for a targeted abstained answer through the gate path; JSON output omitting probability fields rather than emitting `null`.
- [ ] Extend `tests/integration/test_cli_errors.py`: missing key and missing `openrouter_model` produce `configuration_error` with `paid_request=false` before any send; error envelopes carry the conservative `paid_request` states.
- [ ] Extend `tests/integration/test_setup_doctor.py`: openrouter setup flow, doctor key/config/model checks, opt-in key check skipped by default.
- [ ] Run: `cd /root/business/PROJECTS/bruv && .venv/bin/python -m pytest tests/integration -q` — all pass.

Commit: `test: openrouter integration coverage`

## Task 16: Documentation

- [ ] `README.md`: add openrouter to the backend table (hosted, uncalibrated, paid, label-only, explicit abstention) and a usage example with `--backend openrouter`.
- [ ] `docs/configuration.md`: `backend = "openrouter"`, required `openrouter_model` with the no-`openrouter/auto` rule, `openrouter_data_collection = "deny"` default and explicit relax disclosure, `openrouter_zdr = true` default and explicit relax disclosure, key in credential store only (`OPENROUTER_API_KEY`), never in config.
- [ ] `docs/cli-reference.md`: backend flag values, exit 11 abstain behavior, paid-request policy (conservative post-send `paid_request=true`, no retries, next attempt is a new request), privacy defaults.
- [ ] `CHANGELOG.md`: new Unreleased entry covering the backend, the required model config, privacy defaults, paid-request policy, the shared label-only answer modes, and the label-only `AbstainAnswer` mode.
- [ ] Verify: `cd /root/business/PROJECTS/bruv && .venv/bin/python -m pytest tests/contract/test_skill.py tests/contract/test_machine_contracts.py -q` (skill/docs contracts still consistent).

Commit: `docs: openrouter backend`

## Task 17: Optional paid live smoke test (opt-in only)

- [ ] Add `markers = ["live: opt-in tests that contact live services (deselected by default)"]` to the existing `[tool.pytest.ini_options]` table in `pyproject.toml`. Merged result (one table only; never create a duplicate table):

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q --strict-markers"
markers = ["live: opt-in tests that contact live services (deselected by default)"]
```

- [ ] Create `tests/integration/test_openrouter_live_smoke.py`:

```python
"""Opt-in paid smoke test.

WARNING: running this test performs real paid OpenRouter Chat Completions
calls and spends credits. It is excluded from normal CI. Run it only when you
explicitly provide OPENROUTER_API_KEY and BRUV_LIVE_SMOKE_MODEL.
"""

import os

import pytest

from bruv.backends.openrouter import OpenRouterAdapter
from bruv.domain.requests import DecisionRequest

pytestmark = pytest.mark.live


@pytest.fixture(autouse=True)
def _require_opt_in():
    if not os.environ.get("OPENROUTER_API_KEY") or not os.environ.get("BRUV_LIVE_SMOKE_MODEL"):
        pytest.skip("opt-in only: set OPENROUTER_API_KEY and BRUV_LIVE_SMOKE_MODEL to run (this spends credits)")


def test_live_smoke_label_only_choice():
    print("PAID CALL WARNING: this test spends OpenRouter credits.")
    import httpx

    adapter = OpenRouterAdapter(
        transport=httpx.Client(
            base_url="https://openrouter.ai/api/v1",
            timeout=60.0,
        ),
        api_key=os.environ["OPENROUTER_API_KEY"],
        configured_model=os.environ["BRUV_LIVE_SMOKE_MODEL"],
    )
    request = DecisionRequest.model_validate(
        {
            "state": "Duplicate charge on renewal invoice",
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": "Which team owns this?",
                    "criteria": {"sales": "Revenue team", "billing": "Billing team"},
                }
            },
        }
    )
    result = adapter.evaluate(request)
    assert result.backend == "openrouter"
    assert result.calibrated is False
    answer = result.answers["route"]
    assert answer.type in ("choice", "abstain")
```

- [ ] Do not run it without the user's explicit instruction. When the user opts in, run exactly:

```bash
cd /root/business/PROJECTS/bruv
OPENROUTER_API_KEY="<user-provided>" BRUV_LIVE_SMOKE_MODEL="<user-provided>" \
  .venv/bin/python -m pytest tests/integration/test_openrouter_live_smoke.py -m live -q
```

Expected (when opted in): one paid call, label-only choice or abstain result, exit 0. Without opt-in: the test skips.

Commit: `test: opt-in openrouter paid live smoke`

## Task 18: Full verification

- [ ] Run the complete suite:

```bash
cd /root/business/PROJECTS/bruv
.venv/bin/python -m ruff check src tests
.venv/bin/python -m mypy
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest tests/contract -q
```

Expected: ruff clean, mypy strict clean, full suite green, all existing backend snapshots byte-identical, openrouter fixtures updated. Confirm the exclusions: no workspace Classifiers code, no streaming, no retry path, no healing, no default model, no `openrouter/auto` acceptance, no dependency extra added (`grep -n "openrouter" pyproject.toml` shows only the `live` marker context).

No commit (verification only).

---

## Completion notes

- Serializers omit inactive fields only; existing backends stay byte-compatible and their contract tests never change.
- One adapter, one registration, metadata-driven validation; no backend-name branches in config, CLI, contracts, setup, or doctor.
- Malformed provider output always fails as `provider_response_error` with `paid_request=true`; abstentions are per-question; siblings complete.
- The paid live smoke test is the only test that can spend credits and requires explicit user-provided credentials and model.
