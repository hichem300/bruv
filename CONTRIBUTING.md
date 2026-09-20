# Contributing to bruv

Thanks for your interest. bruv is a public Apache-2.0 project.

## Set up

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
```

## Checks

```bash
ruff format .
ruff check .
mypy src/bruv
pytest
```

## Before opening a PR

- Add tests for new behavior.
- Keep Simple Jev `calibrated: false` in every code path.
- Never add API keys as CLI flags.
- Use synthetic fixtures only; no customer data or private endpoints.
- Update `docs/` and `CHANGELOG.md` when behavior changes.
- Confirm `bruv validate`, `spec`, `schema`, `doctor`, and dry-run make no
  paid request.
