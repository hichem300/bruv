# Changelog

## Unreleased

- New optional Needle backend: local Needle 3 inference, no credential, via
  `pip install 'bruv[needle]'`. About 35 MB of weights download on first use,
  then inference runs fully offline with telemetry disabled.
- Needle reports confidence only. The output schema is relaxed so confidence-
  only answers (no synthetic probabilities) validate; score answers carry the
  selected level index and its legend.
- `setup` and `doctor` derive backend choices, descriptions, and install hints
  from the registry, and doctor adds Needle dependency, platform, and local
  cache checks with no runtime construction or download.

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
