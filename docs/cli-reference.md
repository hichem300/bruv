# CLI reference

Run `bruv --help` for the live list. `bruv spec --output json` prints the
machine-readable spec from the same source as help.

## Commands

- `noul INSTRUCTIONS`: yes/no confidence question.
- `choice INSTRUCTIONS --option KEY[=DESC] ...`: pick one of two or more.
- `score INSTRUCTIONS --level LVL ...`: rate on ordered levels.
- `eval FILE`: evaluate a multi-question request from a file.
- `validate -f FILE`: validate a request with no paid call.
- `spec`: print the CLI command spec.
- `schema NAME`: print the request, output, or error JSON Schema.
- `setup`: interactive backend and credential setup.
- `doctor`: run non-paid diagnostics.
- `skill install --target T --scope S`: install the agent skill.
- `demo funnel-audit`: run the safe funnel audit demo.

## Common flags

`--backend`, `--model`, `--output human|json`, `--quiet`, `--no-color`,
`--dry-run`, `--field PATH`, `--fail-under N`, `--abstain-band L:H`.

## Exit codes

0 success, 2 usage/validation, 3 authentication, 4 backend unavailable,
5 provider response, 10 gate failed, 11 abstained, 70 internal error.

Every error envelope includes `paid_request` so you know whether a provider
call may have incurred cost.

## Needle backend

`--backend needle` runs Needle 3 inference locally. No credential needed.
Optional dependency: `pip install 'bruv[needle] @ git+https://github.com/hichem300/bruv.git'`. First use downloads about
35 MB of model weights to your local cache; after that it runs offline with
telemetry disabled (`NEEDLE_TELEMETRY=0`).

Needle output is confidence-only: one confidence value per answer, no
probability distributions, and bruv never synthesizes probabilities. Score
answers report the selected level index plus its legend. Supported on Linux,
macOS, and Windows (x86_64 and arm64). `bruv doctor` checks the dependency,
platform, and local cache without any download.

## RLCD ModernBERT backend

`--backend rlcd-modernbert` runs a pinned GLiClass-ModernBERT decision model
locally via ONNX Runtime on CPU. No credential needed. Optional dependency:
`pip install 'bruv[rlcd-modernbert] @ git+https://github.com/hichem300/bruv.git'`. First use downloads a ~606 MB pinned
model artifact plus tokenizer and calibrator files from Hugging Face; after
that runs use the local cache, and `HF_HUB_OFFLINE=1` works once cached.

RLCD supports noul, choice, and score with calibrated probabilities and
explicit per-question abstention. Unsupported candidate totals are rejected
before inference. Score values are deterministic zero-based indices; the
expected value over substantive mass is metadata only.

When a targeted `--abstain-band` gate resolves to abstention, bruv prints an
`AbstainAnswer` and exits with code 11; mixed question batches keep every
non-abstaining sibling answer. Supported on Linux, macOS, and Windows
(x86_64 and arm64). `bruv doctor` checks the dependency, platform, and pinned
cache status (verified, missing, hash-mismatch, or dependency-missing)
without any download or model load.
