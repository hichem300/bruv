## Summary

<!-- One sentence describing the change -->

## Checks

- [ ] Tests added or updated
- [ ] `ruff format . && ruff check . && mypy src/bruv && pytest` pass
- [ ] Simple Jev reports `calibrated: false` in affected paths
- [ ] No API keys added as CLI flags
- [ ] No customer data, private endpoints, or secrets in fixtures/examples
- [ ] Docs and CHANGELOG updated if behavior changed
- [ ] `validate`, `spec`, `schema`, `doctor`, and dry-run make no paid request
