# OpenRouter Backend Design

**Date:** 2026-09-22
**Status:** Approved

## Goal

Add `openrouter` as a paid hosted `bruv` backend for `noul`, `choice`, and `score` decisions, using the OpenRouter Chat Completions API with strict structured outputs. One plain HTTP call per request, no SDK, no workspace Classifiers, no default model, no retries, and no fabricated confidence. Abstention is explicit through the shared canonical `AbstainAnswer`.

## User interface

No extra dependency group is required; the adapter reuses the existing `httpx` dependency of core `bruv`. Select the backend through existing CLI and request interfaces:

```bash
bruv choice "Which team owns this?" \
  --option sales \
  --option billing \
  --state "Duplicate charge on renewal invoice" \
  --backend openrouter
```

`openrouter` becomes valid anywhere `typesafe`, `simple-jev`, `needle`, and `rlcd-modernbert` are accepted: CLI flags, `JEV_BACKEND`, config, request validation, setup, doctor, machine-readable spec, and demos where capabilities permit. Missing or invalid credentials produce a configuration error before any request is sent.

## Architecture

### Hosted structured-output adapter

Create an OpenRouter adapter behind the existing `DecisionBackend` port. The adapter performs one nonstreaming `POST https://openrouter.ai/api/v1/chat/completions` per `evaluate()` call, authenticating with `Authorization: Bearer $OPENROUTER_API_KEY`. It uses plain `httpx` only: no OpenRouter SDK, no response streaming, no automatic retry.

This is a Chat Completions adapter, not OpenRouter's async workspace Classifiers product. Classifiers are rejected for this backend (see rejected alternatives).

### Extensible backend registry

Register a new `BackendDefinition` named `openrouter` in the existing registry:

- `calibrated=False`
- `runs_local=False`
- `needs_credentials=True`
- question types `noul`, `choice`, `score`
- no install hint; `httpx` is already a core dependency
- credential source: the existing bruv credential store, surfaced as `OPENROUTER_API_KEY` to the adapter

One adapter, one registration definition, provider-specific tests, and documentation. No new backend-name branches across config, CLI, contracts, setup, and doctor.

### Capability metadata

`openrouter` declares `explicit_abstention=True` and `reserved_answer_ids={"__abstain__"}`. It declares no probability-based capability fields: all OpenRouter answers are label-only. Existing registry-driven validation applies unchanged, including rejection of user-supplied option or level IDs that collide with `__abstain__`.

## Model precedence

OpenRouter models are arbitrary catalog identifiers, so bruv does not pin one model. The resolved model is `request.model or config.openrouter_model`, matching the TypeSafe and Simple Jev precedence:

- `config.openrouter_model` is required, nonblank, has no default value, and must not be `openrouter/auto` or any other router-passthrough alias. Configuration without a concrete model fails validation.
- An explicit `request.model` on a request overrides the configured model for that request. It must be nonblank and must not be `openrouter/auto`; validation rejects otherwise before any request is sent.
- There is no fallback to another backend, no free-model fallback, and no automatic upgrade or downgrade.

The resolved model string travels unchanged in the request body's `model` field and is recorded in result metadata.

## Request contract

### One batched nonstreaming call

One nonstreaming Chat Completions call carries every question of the request. There are no per-question calls. The 256-question request bound already limits batch size.

### Strict JSON Schema

The request uses OpenRouter structured outputs: `response_format` of type `json_schema` with `strict: true`, and provider routing settings `{"require_parameters": true}` so only providers that honor structured outputs serve the request. If no provider can honor the settings, the API fails the request; bruv surfaces that as a backend-unavailable error and never relaxes the schema to get a response.

The schema is a closed top-level object with one literal property per question ID:

- Top level: `properties` contains exactly the request's question IDs; every question ID is required; `additionalProperties: false`. Dynamic question IDs are literal schema properties, not a pattern or map.
- Per question, the value is a closed object (`additionalProperties: false`) with:
  - `selection`: an enum of that question's candidate answer IDs plus the canonical `__abstain__`. For `noul` the enum is `["true", "false", "__abstain__"]`; for `choice` it is the option IDs plus `__abstain__`; for `score` it is the level IDs plus `__abstain__`.
  - `abstain_reason`: an optional property with enum `["insufficient_evidence", "provider_refusal", "content_filter"]`. It is required by validation exactly when `selection` is `__abstain__`; its presence with any other selection is malformed output and fails.
- Instructions and question text become prompt and description content. They never alter enum identifiers.

### Privacy defaults

The request carries `provider.data_collection = "deny"` and `zdr = true` by default: prompt and completion content must not be used for training or retained beyond routing. Relaxing either requires explicit config (see Configuration), and the relaxed choice is stated in the next request's metadata. Restrictive privacy settings can make routing fail when no provider supports them; bruv reports the API's routing failure as a backend-unavailable error and never silently drops a privacy setting to make routing succeed.

### Malformed output fails

The adapter parses the response body as exact JSON exactly once: no fence stripping, no healing plugin, no json-repair, no automatic retry, no second attempt with a relaxed prompt. Missing keys, extra keys, a selection outside the enum, an invalid or missing `abstain_reason` for an abstention, or any other schema mismatch is a provider-response error and fails the request.

## Canonical result mapping

All OpenRouter answers are label-only. The adapter never solicits, accepts, or fabricates confidence or probabilities, and every successful answer reports `calibrated=false`.

- `noul`: selection `"true"` or `"false"` maps to `NoulAnswer` in the new value-only mode with the boolean `value`; `noul` and `confidence` are absent.
- `choice`: selection of an option ID maps to `ChoiceAnswer` with the selected option ID and no `probabilities`.
- `score`: selection of a level ID maps to `ScoreAnswer` with that level's numeric value as `score` and the rubric legend reconstructed from the request's levels, no `probabilities`.
- Selection `__abstain__`: no `NoulAnswer`, `ChoiceAnswer`, or `ScoreAnswer` is emitted for that question; the shared canonical `AbstainAnswer` below is emitted instead, in label-only mode, with `abstain_reason` as the canonical `reason`. Sibling questions in the same request complete normally; one abstained question never aborts the others.

Reserved IDs never appear inside `ChoiceAnswer`, `ScoreAnswer`, or `NoulAnswer`. `__abstain__` appears only in `AbstainAnswer` context, from backends whose capabilities declare explicit-abstention support.

### Shared label-only substantive answer contract

The canonical substantive answer models support probability-backed and label-only modes precisely:

- `ChoiceAnswer.confidence` and `ScoreAnswer.confidence` become optional.
- Probability-backed mode (TypeSafe, Simple Jev, RLCD) is unchanged: `confidence` and `probabilities` are required together and validated together, exactly as today.
- Label-only mode (OpenRouter) requires the selected `choice` for choice, and the selected `score` plus its `legend` for score, with both `confidence` and `probabilities` absent.
- `NoulAnswer` gains a value-only mode: `value` present with both `noul` and `confidence` absent. Its existing probability mode and value-plus-confidence selection mode are unchanged.
- Validators reject partial or contradictory combinations: confidence without probabilities or probabilities without confidence, label-only answers carrying `confidence` or `probabilities`, probability-backed answers missing either field, and `NoulAnswer` instances with overlapping, missing, or empty mode fields.
- Serializers omit inactive fields only; TypeSafe, Simple Jev, Needle, and RLCD payloads remain byte-compatible because those backends still supply their current fields.

Machine-readable schema, snapshot, and contract outputs are updated for the now-optional confidence fields and the new modes; serialized output for existing backends is unchanged.

### Abstention sources

- **Schema-selected abstention:** the model returned structured content and chose `__abstain__` for one or more questions. Those questions become label-only `AbstainAnswer` with the selected canonical reason (`insufficient_evidence` in the normal case); siblings keep their substantive answers.
- **Provider refusal or content filter:** the API or an upstream provider refuses the request or blocks content so that no structured content exists for affected questions. The adapter maps every affected question to label-only `AbstainAnswer` with canonical reason `provider_refusal` or `content_filter` respectively, preserves safe sanitized error metadata, and lets unaffected siblings complete when structured content exists for them.

### Shared canonical AbstainAnswer

OpenRouter emits the shared canonical `AbstainAnswer` contract in label-only mode: required `type` (`"abstain"`), `reason`, and `source_question_type`; `confidence`, `probabilities`, and `legend` are omitted and their presence is a validation error. The probability-backed mode, in which `confidence` and `probabilities` are required together and `probabilities` includes the reserved `__abstain__` key, remains exactly as specified in the RLCD ModernBERT design; RLCD always uses probability-backed `insufficient_evidence`.

Behavior: when the targeted answer abstains, render, JSON, and schema output present the `AbstainAnswer`, gates targeting it resolve against `AbstainAnswer` fields, and the command exits 11 (the existing abstain band). Sibling answers in the same request complete normally.

## Paid status and errors

### paid_request lifecycle

`paid_request` is `false` while the request is being prepared, including validation and configuration failures before send. Once the POST is sent, `paid_request` is conservatively `true` for that request's outcome, including malformed-response, timeout-after-send, and error responses, because credits may already be consumed. There is no retry and no replay: OpenRouter documents no idempotency mechanism for Chat Completions, and a duplicate send risks double spend. The next attempt is a new user-initiated request.

### Error mapping

Stable mapping of HTTP status and the body-level `error` object:

- 400 invalid request or rejected schema: provider-response error, `paid_request` reflects send state.
- 401 invalid or missing API key: configuration error, `paid_request=false` (pre-send).
- 402 insufficient credits: backend-unavailable error naming the credit shortfall, from the body `error` message.
- 403 blocked key or moderator-flagged request: backend-unavailable error with sanitized details.
- 408 request timeout: provider-response error, `paid_request=true` after send.
- 429 rate limited: backend-unavailable error.
- 502 model provider error, 503 no available provider (including privacy-filter routing failures): backend-unavailable error with the body `error` message.
- A body-level `error` object on any status overrides generic status text; its documented `code` and `message` are surfaced sanitized.
- Network-level transport failures (connection, DNS, TLS): backend-unavailable error, `paid_request=false` when the failure occurred before send.

`paid_request` is `false` in every error raised before send and `true` for every error raised after send. Usage and cost metadata returned by the API (token counts, cost when present) are recorded in provider metadata. Credentials, the API key, and full authorization headers are never logged, never stored in metadata, and never echoed in errors; existing redaction covers them.

Exit codes follow the existing policy, including exit 11 for an abstained targeted answer.

## Configuration and onboarding

Config values:

```toml
backend = "openrouter"
openrouter_model = "openai/gpt-4o-mini"   # required, no default
openrouter_data_collection = "deny"       # default; explicit value required to relax
openrouter_zdr = true                     # default; explicit false required to relax
```

`openrouter_model` is required and validated: nonblank, not `openrouter/auto`, and not blank-padded. `openrouter_data_collection` defaults to `"deny"`; setting it to an explicit permissive value relaxes training-data collection for that installation and is disclosed in request metadata. `openrouter_zdr` defaults to `true`; an explicit `false` is required to relax zero-data retention and is disclosed in request metadata. The API key never lives in the config file; setup stores it in the existing credential store.

Setup offers OpenRouter, states that it is a paid hosted backend, explains that request content leaves the machine to OpenRouter and upstream providers, shows the privacy defaults, and prompts for the API key and model. Doctor checks: key presence in the credential store, config validity including the model rule, and, optionally, the documented non-paid `GET https://openrouter.ai/api/v1/key` credit-and-key check when the user opts in. Doctor never sends a paid Chat Completions request.

## Output and gates

Human output shows the selected value, `calibrated: false`, the backend and resolved model, and an explicit abstention notice with the canonical reason when an abstention is targeted. JSON output omits unavailable probability fields rather than emitting `null` or synthetic values, and records usage, cost when present, and the privacy settings used. Existing `--field` access remains valid; confidence gates cannot be expressed against OpenRouter answers because no confidence exists, and score gates continue operating on the selected level value. When the targeted answer is abstained, gates targeting it resolve against `AbstainAnswer` fields and the command exits 11. Schema and CLI spec outputs document OpenRouter's label-only semantics, explicit-abstention support, and the shared label-only substantive answer modes.

## Security

- The API key is stored only in the existing credential store, passed only as the Authorization header, and excluded from logs, metadata, errors, and exports by the existing redaction layer.
- Privacy defaults (`data_collection="deny"`, `zdr=true`) mean request content is not used for training and not retained; relaxing either requires explicit config and is disclosed per request.
- No code is executed from API responses; bodies are parsed as JSON data only, once, with strict schema validation.
- Security reporting follows the repository's `SECURITY.md` process; the backend introduces no new listening surface.

## Testing

Implementation-first, followed by focused tests:

- Adapter unit tests against a fake HTTP transport: exact request body including `model`, structured-output `response_format`, `provider.require_parameters`, and privacy fields; one nonstreaming call per request; Authorization from the credential store.
- Schema construction tests for `noul`, `choice`, and `score`: per-question closed objects, literal question-ID properties, all required, `additionalProperties: false`, selection enums including `__abstain__`.
- Precedence tests: `request.model` override, config fallback, rejection of blank or `openrouter/auto` in either place.
- Strict parsing tests: fences, trailing prose, extra keys, missing keys, invalid selection, and abstention-without-reason all fail as provider-response errors with exactly one parse attempt.
- Abstention tests: schema-selected abstention preserves siblings; API refusal and content-filter paths map affected questions to label-only `AbstainAnswer`; exit 11 and gate/render/JSON behavior.
- Shared `AbstainAnswer` tests: label-only validation rejects `confidence`, `probabilities`, and `legend`; probability-backed validation unchanged for RLCD.
- Substantive answer mode tests: `ChoiceAnswer` and `ScoreAnswer` validate probability-backed mode (confidence plus probabilities together), label-only mode (selected choice; score plus legend, with `confidence` and `probabilities` absent), and reject partial or contradictory combinations; `NoulAnswer` validates value-only mode and preserves its probability and value-plus-confidence modes; serialized TypeSafe, Simple Jev, Needle, and RLCD outputs are byte-identical snapshots.
- Error mapping tests table-driven over 400, 401, 402, 403, 408, 429, 502, 503, body-level `error`, and transport failures, asserting `paid_request` send-state transitions.
- Doctor tests: key/config checks and optional non-paid key check; no paid request ever.
- Registry, config validation, setup, CLI, JSON output, field selection, gates, specs, schemas, and snapshots.
- Existing TypeSafe, Simple Jev, Needle, and RLCD contract tests remain unchanged.

## Compatibility

- Existing answer payload shapes are unchanged for existing backends; TypeSafe, Simple Jev, Needle, and RLCD payloads stay byte-compatible because they still supply their current fields. One new answer type, `AbstainAnswer`, is added and emitted only by backends declaring explicit-abstention support.
- Shared substantive answer contract changes: `ChoiceAnswer.confidence` and `ScoreAnswer.confidence` become optional, `NoulAnswer` gains a value-only mode (`value` present, `noul` and `confidence` absent), and validators enforce exactly one mode per answer, rejecting partial or contradictory combinations. Label-only mode (OpenRouter) carries the selected choice or score plus legend and omits `confidence` and `probabilities`; probability-backed mode is unchanged. Machine schema, snapshot, and contract outputs are updated accordingly.
- The shared canonical `AbstainAnswer` gains the label-only mode described here; RLCD's probability-backed mode and its validation are unchanged.
- Registry metadata advertises `openrouter` with `calibrated=False`, `runs_local=False`, `needs_credentials=True`, `explicit_abstention=True`, and `reserved_answer_ids={"__abstain__"}`; registry-driven validation applies without backend-name branches.
- `DecisionResult.backend` continues to accept registry-advertised values; no contract widening beyond what the RLCD design already defines.
- Schema and CLI spec outputs gain `openrouter` with label-only capability notes.
- Changelog and CLI reference document the backend, the required model config, the privacy defaults, the paid-request policy, and the label-only `AbstainAnswer` mode.

## Out of scope

- OpenRouter workspace Classifiers.
- Streaming responses.
- Automatic retries, idempotency keys, or request replay.
- Response healing, fence stripping, or json-repair.
- Default models, `openrouter/auto`, or free-model fallback.
- Model-reported confidence or per-option probabilities.
- Per-question API calls.
- Selecting or ranking providers beyond the documented routing settings.

## Decisions and rejected alternatives

- **Workspace Classifiers:** rejected. Async workspace classification is a different product surface with different latency, delivery, and pricing semantics; bruv needs one synchronous strict-schema call per request.
- **OpenRouter SDK:** rejected. A single authenticated POST with a JSON body needs no SDK; plain `httpx` keeps the dependency surface unchanged and auditable.
- **Default model or `openrouter/auto`:** rejected. Silent model selection changes cost, capability, and privacy behavior between runs; a concrete model is required from config or the request.
- **Per-question calls:** rejected. One batched call preserves the request as the unit of payment, latency, and paid-status accounting.
- **Model-reported confidence:** rejected. OpenRouter label-only responses carry no calibrated probabilities; soliciting self-reported confidence would fabricate calibrated-looking values with `calibrated=false` semantics violated. Never solicited, never synthesized.
- **Retries:** rejected. OpenRouter documents no idempotency mechanism for Chat Completions; automatic retry risks double spend on an already-sent request.
- **Response healing:** rejected. Fence stripping and json-repair turn malformed output into guessed answers; malformed output fails instead.
- **Relaxing privacy to make routing succeed:** rejected. Dropping `data_collection="deny"` or `zdr=true` silently would leak request content beyond the user's consent; routing failures surface as errors instead.

## Sources

- Structured outputs: https://openrouter.ai/docs/guides/features/structured-outputs
- Provider routing and selection (`require_parameters`, `data_collection`): https://openrouter.ai/docs/guides/routing/provider-selection
- Chat Completions endpoint and API reference: https://openrouter.ai/docs/api_reference/overview
- Errors and debugging: https://openrouter.ai/docs/api_reference/errors-and-debugging
- Limits: https://openrouter.ai/docs/api_reference/limits
- Classifiers (rejected product surface): https://openrouter.ai/docs/guides/features/classifiers
- Key check endpoint: `GET https://openrouter.ai/api/v1/key`
