# Changelog

## Unreleased

- New optional RLCD ModernBERT backend: local calibrated ONNX Runtime CPU
  inference, no credential, via `pip install 'bruv[rlcd-modernbert]'`. A
  ~606 MB pinned model artifact plus tokenizer and calibrator files download
  from Hugging Face on first use at an immutable revision, verified by pinned
  sizes and SHA-256 hashes; later runs use the local cache and work offline
  with `HF_HUB_OFFLINE=1`.
- RLCD supports noul, choice, and score with calibrated probabilities and
  explicit per-question abstention; abstain answers carry the full
  distribution including `__abstain__`, unsupported candidate totals are
  rejected before inference, and the request model override must match the
  pinned model exactly.
- New optional Needle backend: local Needle 3 inference, no credential, via
  `pip install 'bruv[needle]'`. About 35 MB of weights download on first use,
  then inference runs fully offline with telemetry disabled.
- Needle reports confidence only. The output schema is relaxed so confidence-
  only answers (no synthetic probabilities) validate; score answers carry the
  selected level index and its legend.
- `setup` and `doctor` derive backend choices, descriptions, and install hints
  from the registry. Doctor adds Needle dependency, platform, and local cache
  checks, plus RLCD ModernBERT dependency, platform, and pinned local cache
  checks; all with no runtime construction or download.

## 0.1.0

Initial public release of bruv.

- One canonical typed-decision contract over TypeSafe Jev and Simple Jev.
- `noul`, `choice`, `score`, `eval`, `validate`, `spec`, `schema`, `setup`,
  `doctor`, `skill`, and `demo` commands.
- Stable error envelope with `paid_request` semantics.
- Simple Jev always reports `calibrated: false`.
- Packaged agent skill with safe installer for Codex, Claude Code, and Pi.
- Funnel audit demo with safe dry-run default.
- Apache-2.0 license; unofficial project disclosure in `NOTICE`.
