# bruv

no yap, only fax. bruv gives you typed decisions (yes/no, choice, score) from
TypeSafe Jev or Simple Jev, with one canonical contract, safe onboarding, and
honest calibration disclosure.

> bruv is an independent, unofficial project. It is not affiliated with,
> endorsed by, or sponsored by TypeSafe AI, Featherless AI, or the Qwen Team.

The tagline is brand voice, not an accuracy guarantee.

## 60-second install and first result

```bash
python -m venv .venv && . .venv/bin/activate
pip install bruv

# Simple Jev needs no key
bruv noul "Does this ask for a refund?" --state "I want my money back" --backend simple-jev

# JSON for scripts
bruv choice "Which team?" --option sales --option billing --state "help" --backend simple-jev --output json
```

## Human use

- `bruv setup` (interactive) configures a backend and stores credentials safely.
- `bruv doctor` runs non-paid diagnostics and prints one fix per failed item.
- `bruv validate -f request.yaml` checks a request with no paid call.

## Agent use (Codex, Claude Code, Pi)

Install the packaged skill:

```bash
bruv skill install --target pi --scope user
bruv skill install --target claude --scope project --project-root /path/to/repo
```

See `docs/agent-usage.md`.

## Funnel audit demo

```bash
bruv demo funnel-audit --backend simple-jev            # dry run, no paid call
bruv demo funnel-audit --backend simple-jev --execute  # real evaluation
```

## Backends and calibration

- TypeSafe Jev: hosted, calibrated probabilities (population-level).
- Simple Jev: local/self-hosted, explicitly uncalibrated. bruv always reports
  `calibrated: false` for Simple Jev.

## Security and privacy

bruv never accepts API keys as CLI flags. Use the `TYPESAFE_API_KEY`
environment variable or `bruv setup`. No telemetry is sent. See `SECURITY.md`.

## Contributing

See `CONTRIBUTING.md`. Please report security issues privately via `SECURITY.md`.

## Deeper docs

- `docs/installation.md`
- `docs/configuration.md`
- `docs/cli-reference.md`
- `docs/release-checklist.md`
