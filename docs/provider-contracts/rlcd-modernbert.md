# RLCD ModernBERT backend contract

Source model: https://huggingface.co/heman10x/rlcd-modernbert-151m
Accessed: 2026-09-22
Pinned revision: `8af2496eb63c7fa66d7d234e1f62629380030eb4` (immutable; never `main`).
License: Apache-2.0 (upstream model card). The model is a GLiClass-ModernBERT
decision model with RLCD calibration. Upstream source of the prompt formatting
and calibration behavior is commit
`465d542f04968e84236d611602e357e7c8d69a15` of
`Heman10x-NGU/Verdict-open-jev`, per the design document in
`docs/superpowers/specs/2026-09-22-rlcd-modernbert-backend-design.md`.

This note pins bruv's local RLCD ModernBERT boundary to the pinned artifacts
and exact upstream behavior.

## Pinned artifacts

bruv downloads exactly four files from `heman10x/rlcd-modernbert-151m` at the
pinned revision via `hf_hub_download`. Each file is verified against a pinned
byte size and SHA-256 before use; nothing unverified is ever executed:

- `model.onnx` (606323181 bytes, ~606 MB)
- `tokenizer.json`
- `tokenizer_config.json`
- `calibrator.json`

There is never a fallback to another revision or a global snapshot. With
`HF_HUB_OFFLINE=1`, bruv uses only the local Hugging Face cache and fails
closed with a remediation message if a pinned artifact is missing.

## Runtime behavior

- Inference runs locally through ONNX Runtime on CPU
  (`CPUExecutionProvider`); no GPU or hosted service is involved.
- Every question is formatted with the pinned `<<LABEL>>`/`<<SEP>>` markers,
  tokenized in one batch, and scored in exactly one ONNX run. Truncation caps
  at 512 tokens.
- Logits are calibrated per choice count with the pinned calibrator
  (`rlcd-calibrator-v1`, per-K temperatures for K in {2, 3, 4, 5, 6, 7, 9,
  11, 17, 25}). The canonical `__abstain__` answer id replaces the upstream
  abstain sentinel everywhere. No fabricated or uncalibrated fallback is
  ever emitted.
- The reserved input markers `<<LABEL>>` and `<<SEP>>` and the reserved
  answer id `__abstain__` are rejected in user-supplied options and levels.

## Answer semantics

- Supported question types: `noul`, `choice`, `score`. The total candidate
  count K includes abstain and must be in {2, 3, 4, 5, 6, 7, 9, 11, 17, 25};
  unsupported totals are rejected before inference (substantive maximum 24).
- Probabilities are calibrated. Substantive probabilities are conditional on
  sufficient evidence: they renormalize the non-abstain mass.
- Abstention is explicit per question. An abstain answer carries the full
  distribution including `__abstain__`, the abstain confidence, and (for
  choice and score) a legend mapping candidate ids to caller-supplied values
  plus `__abstain__`.
- Score values are deterministic zero-based indices into the supplied level
  list; the expected value over the substantive mass is metadata only.

## Pins and privacy

- `rlcd_model` and `rlcd_revision` config keys are pinned fail-closed. There
  is no environment variable or CLI flag to change the model or revision. A
  request-level model override must match `heman10x/rlcd-modernbert-151m`
  exactly or the request is rejected before any artifact or inference work.
- Inference is fully local. Prompts and state never leave your machine. The
  only network contact is the first-use artifact download from Hugging Face
  at the pinned revision.

## Doctor and lazy loading

- `bruv doctor` reports the pinned cache status per artifact (`verified`,
  `missing`, `hash-mismatch`, or `dependency-missing`) with remediation
  hints, without network access, downloads, or model loading.
- `bruv --help`, `bruv spec`, config loading, `--dry-run`, and `doctor` never
  import the RLCD runtime, create a session, download artifacts, or load the
  model. The optional dependencies (`numpy`, `onnxruntime`, `tokenizers`,
  `huggingface_hub`) are imported only when the backend is constructed.
- The platform check accepts Linux, macOS, and Windows on x86_64 and arm64.

## Output and exit codes

- Abstention surfaced through a targeted `--abstain-band` gate produces an
  `AbstainAnswer` and exit code 11. Mixed question batches preserve every
  non-abstaining sibling answer.
- All errors are sanitized: prompt, state, and logits content never enters an
  error message. Result envelopes carry `calibrated: true` and
  `backend: "rlcd-modernbert"`.

## Distinction from Needle and hosted backends

- Needle reports confidence only and never produces probability
  distributions; RLCD reports calibrated probability distributions and
  explicit abstention.
- TypeSafe Jev is hosted with an API key; Simple Jev is uncalibrated. RLCD is
  local, keyless, and calibrated.
