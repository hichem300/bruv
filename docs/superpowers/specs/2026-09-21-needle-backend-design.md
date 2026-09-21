# Needle 3 Backend Design

## Goal

Add Needle 3 as a first-class local `bruv` backend for `noul`, `choice`, and `score` decisions. Keep core installation lightweight, run inference locally without an API key, and never invent probability distributions Needle does not provide.

## User interface

Install optional support:

```bash
pip install 'bruv[needle]'
```

Select backend through existing CLI and request interfaces:

```bash
bruv choice "Which team?" \
  --option sales \
  --option billing \
  --state "Refund duplicate charge" \
  --backend needle
```

`needle` becomes valid anywhere `typesafe` and `simple-jev` are accepted: CLI flags, `JEV_BACKEND`, config, request validation, setup, doctor, machine-readable spec, and demos where capabilities permit.

## Architecture

### In-process optional backend

Create a Needle adapter behind the existing `DecisionBackend` protocol. The adapter imports `cactus-needle` lazily. Core `bruv` remains installable without Needle or its platform-specific runtime.

The optional dependency is exposed as the `needle` extra. Missing dependency errors instruct users to install `bruv[needle]`.

Before importing or constructing Needle, bruv disables Needle telemetry through the documented environment setting. User prompts, outputs, and runtime activity must remain local.

### Model lifecycle

Use base Needle 3. First construction downloads and caches the approximately 35 MB `.cact` model and platform engine. Later runs use that local cache. No API key or HTTP inference server is required.

The adapter must expose a client/runtime port so tests can use fakes without importing Needle or downloading weights.

### Question execution

Evaluate each question separately. Each question gets its own schema and Needle completion, producing confidence specific to that answer instead of reusing one combined confidence.

- `choice`: constrain output to one enum containing criterion keys.
- `noul`: constrain output to a boolean.
- `score`: constrain output to one enum containing stable string indices for rubric levels; map selected index to numeric score and preserve the rubric legend.

State accepts existing canonical JSON values. Serialize non-string JSON deterministically before passing it to Needle. Instructions and criterion descriptions become schema/tool descriptions; they must never alter enum identifiers.

## Canonical result changes

Needle returns a selected structured value and one calibrated whole-response confidence. It does not return a probability distribution over every allowed value. bruv must not synthesize one.

Extend canonical answer models without changing existing TypeSafe or Simple Jev payloads:

- `NoulAnswer`: support either existing `noul` probability mode or `value` plus `confidence` selection mode. Enforce exactly one mode.
- `ChoiceAnswer`: retain selected `choice` and `confidence`; make `probabilities` optional. Validate sum and selected-key membership only when probabilities exist.
- `ScoreAnswer`: retain `score`, `confidence`, and `legend`; make `probabilities` optional. Validate legend/probability keys and sum only when probabilities exist.

Needle answers omit unavailable probability fields. TypeSafe and Simple Jev continue emitting current fields.

Add `needle` to backend identifiers. Base Needle 3 results report `calibrated: true` because Cactus documents confidence as calibrated for the base model. Provider metadata must state that confidence applies to the selected full response and that per-option probabilities are unavailable. Tuned Needle weights are outside initial scope because their confidence is unavailable.

## Refusal and errors

- Missing optional dependency: configuration error with exact install command.
- Unsupported platform or runtime initialization failure: backend-unavailable error with original details sanitized.
- Download failure: backend-unavailable error explaining cache/model retrieval failed.
- Empty `function_calls`: abstain rather than fabricate an answer.
- `suppressed_calls`: abstain and preserve safe metadata indicating Needle withheld the answer.
- Malformed response, wrong enum, missing confidence, or schema mismatch: provider-response error.
- Tuned weights: rejected in v1 of this backend.

Use existing exit-code policy. No provider call is paid, so Needle errors report `paid_request: false`.

## Configuration and onboarding

Add config values:

```toml
backend = "needle"
needle_model = "Cactus-Compute/needle3"
```

`needle_model` is informational in initial release because only base Needle 3 is supported. Setup offers Needle and explains first-use download. Doctor checks optional package availability, supported platform, writable cache location, and whether model files are already cached. Doctor must not trigger a download.

## Output and gates

Human output displays selected value, Needle confidence, and calibrated status. JSON output omits unavailable probability fields rather than emitting `null` or synthetic values.

Existing `--field` access remains valid. Confidence gates use explicit paths such as `answers.route.confidence`. Existing score gates continue operating on selected rubric index for Needle.

Schema and CLI spec outputs document conditional probability availability and Needle selection semantics.

## Testing

Implementation-first, followed by focused tests:

- Adapter unit tests with fake Needle runtime for choice, noul, and score.
- One-pass-per-question behavior.
- JSON-state serialization.
- Missing dependency, download/runtime failure, refusal, suppression, malformed payload, and missing-confidence paths.
- Canonical model validation for probability-bearing and confidence-only answers.
- Factory, config precedence, setup, doctor, CLI, output, field selection, gates, specs, schemas, and snapshots.
- Ensure existing TypeSafe and Simple Jev contract tests remain unchanged in behavior.
- Optional real-model smoke test marked and excluded from normal CI.
- Full formatting, linting, typing, tests, coverage, build, and clean-install verification.

## Compatibility

Existing provider payloads remain unchanged. Canonical schemas become more permissive because probability fields can be absent for Needle. Consumers that assumed those fields always exist must branch on field presence or backend. Document this as a contract change in changelog and CLI reference.

## Out of scope

- Running Needle as an HTTP service.
- Synthetic per-option probabilities.
- Fine-tuned Needle weights.
- Tool execution or general agent loops.
- Replacing Simple Jev or TypeSafe Jev.
