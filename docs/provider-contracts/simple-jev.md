# Simple Jev provider contract

Source: https://github.com/featherless-ai/simple-jev/blob/57261e8ef3d388fdbb9a4b472cf1a833bc2d4c49/hf-server/API_REFERENCE.md
Accessed: 2026-09-20
Version/commit: HF server API version `0.1.0`; upstream commit `57261e8ef3d388fdbb9a4b472cf1a833bc2d4c49`; document blob `1ec3e3fd05dbad5e2a1e162cbb884af70b329745`.

This note pins bruv's Simple Jev HTTP boundary to upstream `hf-server/API_REFERENCE.md`.

## Transport and endpoints

- `POST /v1/classifier`: score supplied questions. Send `Content-Type: application/json`. No classifier query parameters or required custom headers.
- `GET /health`: after initialization returns `{"status":"ready","model":"<loaded model>"}`. This checks readiness but runs no inference probe.
- Upstream also documents `POST /v1/systemone` as an exact alias, but bruv uses `/v1/classifier`.
- Standalone server supplies no authentication, API-key management, or TLS.

## Exact classifier request fields

Top-level fields are case-sensitive:

- `model`: required nonempty string. Must equal model ID or local path used at server startup; request cannot load or switch models.
- `state`: string, object, array, or null. Supply exactly one non-null `state` or `messages`. Top-level number and boolean are unsupported.
- `messages`: array of one or more message objects, or null, as alternative to `state`. HF implementation accepts only `role` (`system`, `developer`, `user`, or `assistant`) and string `content`.
- `questions`: required object containing 1–256 question IDs at schema level. IDs are nonempty strings. Default runtime branch cap is 100.
- `options`: optional object, default `{"raw_logits": false}`. `raw_logits` is only visible when advanced metrics are enabled.
- `tools`: array of objects or null. Omit or send null/empty; nonempty values are rejected.
- `mm_processor_kwargs`: object or null. Omit or send null/empty; nonempty values are rejected.
- `media_io_kwargs`: object of objects or null. Omit or send null/empty; nonempty values are rejected.

Unknown top-level fields are ignored. Unknown question and option fields are rejected.

Each question requires `type` and `instructions`:

- Choice: `type: "choice"`; required `instructions` entry; required `criteria` object with 2–50 candidate IDs mapped to entries.
- Score: `type: "score"`; required `instructions` entry; required ordered `criteria` array with 2–50 entries.
- Noul: `type: "noul"`; required `instructions` entry; optional `criteria` object containing only `true` and/or `false`; default null.

An entry is string, JSON object, JSON array, or null. Nested JSON may contain normal scalars, but bare numeric or boolean entries are invalid. `instructions` remains required even when its value is null.

## Successful classifier response

Every baseline response contains:

- `model`: request model identifier.
- `answers`: object keyed by request question IDs.
- `usage.input_tokens`: logical unique-prefix token accounting defined by upstream.
- `usage.output_tokens`: always `0` for HF backend.

Answer fields:

- Choice: `type`, `choice`, `confidence`, `probabilities`.
- Score: `type`, `score`, `confidence`, `probabilities`, `legend`. Probability and legend keys are numeric strings from `"0"` through `"N-1"`.
- Noul: `type`, `noul`; no separate confidence field.

Advanced metrics can add answer diagnostics, top-level `metadata`, and top-level metrics when server starts with `ENABLE_OPEN_JEV_ADVANCED_METRICS=1`. bruv must not require these optional fields.

## Error behavior

- `422`: invalid JSON/schema, unknown model, invalid context combination, unsupported chat/media/tool input, branch/token limit, or compiler/backend `ValueError`. Schema and semantic validation normally use upstream `error` envelope, but semantic errors may omit path/details.
- `429`: queue full. Header `Retry-After: 1`; body `{"detail":"Scoring queue is full"}`.
- `499`: client disconnected, if response can still be delivered. Body `{"detail":"Client disconnected"}`.
- `500`: unhandled runtime failure such as model execution error. No stable structured body guaranteed.

## Explicit uncalibrated semantics

Upstream states: “Probabilities are conditional on the candidate/rating token set, not the entire vocabulary, and are not calibrated probabilities of correctness.”

Consequences:

- bruv always maps Simple Jev results to `calibrated: false`.
- Choice `confidence` is winning candidate softmax probability.
- Score `confidence` is largest criterion probability, not confidence interval for expected score.
- Noul is transformed expected nine-bin rating, not binary-token softmax.
- Advanced metadata confirms `metadata.calibration: "not_calibrated"`; answer diagnostics expose `calibrated: false` when enabled. Baseline responses need not contain either optional field.

## Fixture provenance

`tests/fixtures/simple_jev/success.json` is exact synthetic fixture required by bruv plan. It contains no customer data, secrets, or fields outside baseline upstream response shape.
