# Release checklist (0.1.0)

- [ ] `ruff format --check . && ruff check . && mypy src/bruv && pytest` pass.
- [ ] `python -m build` produces wheel and sdist.
- [ ] Wheel installs in a clean venv; offline commands succeed.
- [ ] Standalone artifacts built for linux/macOS/windows x86_64 and ARM.
- [ ] SHA-256 checksums generated and signed.
- [ ] CycloneDX SBOM generated.
- [ ] Changelog updated.
- [ ] GitHub Release created from `v*` tag; PR CI uses no secrets.
- [ ] Optional PyPI publish via OIDC in a protected environment.
