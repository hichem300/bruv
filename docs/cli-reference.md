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
