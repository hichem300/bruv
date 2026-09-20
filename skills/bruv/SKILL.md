---
name: bruv
description: Use when an agent needs a typed yes/no, choice, or score decision from TypeSafe Jev or Simple Jev; validates requests offline, returns deterministic JSON, and reports calibration honestly.
---

# bruv agent skill

bruv gives you typed decisions: `noul` (yes/no confidence), `choice` (pick one
of two or more), and `score` (rate on ordered levels). It wraps TypeSafe Jev and
Simple Jev behind one canonical contract.

## When to use bruv

- You need a structured decision with probabilities, not free text.
- You want to validate the request before any paid call.
- You want deterministic JSON a script can parse.

Do not use bruv for open-ended generation, retrieval, or chat.

## Choose a question type

- `noul`: yes/no with a 0..1 probability. Example: "Does this ask for a refund?"
- `choice`: pick one label from 2..50 options. Example: "Which team handles this?"
- `score`: rate on 2..50 ordered levels. Example: "How urgent is this?"

## Validate before a paid call

Always run `bruv validate -f request.yaml` first. It performs no paid request and
exits 0 for valid input, 2 for invalid input. Fix issues before evaluating.

Use `--dry-run` on any evaluation command to preview the canonical request and
invoke no adapter transport.

## Request machine output

Use `--output json` for scripts. One JSON object on stdout, terminated by a
newline. Diagnostics go to stderr only.

## Handle exit codes

- 0: success.
- 2: usage or validation error (no paid request).
- 3: authentication error.
- 4: backend unavailable.
- 5: provider response error.
- 10: gate failed (threshold not met).
- 11: abstained (value inside the abstain band).
- 70: internal error.

Check `paid_request` in the error envelope to know whether a provider call may
have incurred cost.

## Report calibration honestly

TypeSafe Jev probabilities are provider-documented calibrated probabilities.
Simple Jev is explicitly uncalibrated: bruv always reports `calibrated: false`
for Simple Jev, in JSON and human output. Never present Simple Jev probabilities
as calibrated probabilities of correctness.

## Secret safety

Never pass API keys as CLI flags. bruv reads `TYPESAFE_API_KEY` from the
environment or a protected credential file. Do not print keys, request bodies,
or response bodies in logs.

## Unsupported truth claims

bruv returns model judgments, not ground truth. State results as the model's
decision with calibration context, not as guaranteed facts.