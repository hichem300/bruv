# Live UX testing notes — 2026-09-22

Observed while installing and testing bruv directly on the VPS. Treat these as product evidence, not hypothetical improvements.

## Fixed during session

- Managed Simple Jev now installs CPU-only PyTorch on hosts without NVIDIA instead of downloading multi-gigabyte CUDA packages.
- CPU fallback now uses `float32`, matching upstream guidance.
- Managed setup smoke request now sends valid non-null state.
- OpenRouter uses calibrated typed Decisions API at `POST /api/alpha/decisions`, not Chat Completions.
- `bruv setup openrouter` hides API-key input and stores it in protected credentials.
- OpenRouter setup now has a real Typer default for `~typesafe/jev-latest`; pressing Enter accepts it.
- OpenRouter model validation rejects accidental values such as `yes` and `y`.
- OpenRouter response parsing accepts provider `usage.cost` and maps response `id` to canonical `request_id`.

## Remaining high-priority UX improvements

### 1. Make explicit setup direct

`bruv setup simple-jev` and `bruv setup openrouter` still ask `Run interactive setup?`. Backend was already explicitly selected, so this confirmation is redundant. Go directly into backend setup; retain confirmation only immediately before expensive or paid work.

### 2. Make long setup resumable

Simple Jev setup currently combines install, launch, health check, and smoke classification. A late failure encourages rerunning `--repair`, which can rebuild a large environment unnecessarily.

Desired behavior:

- Detect and reuse completed installation.
- Resume from failed stage.
- Show stages: preflight, install, model download, launch, health, smoke.
- Never reinstall because only smoke classification failed.
- Print `installed/running/healthy` state before suggesting repair.

### 3. Add disk preflight and progress

Before local installation:

- Show required and available disk estimates.
- Explain model size separately from Python/runtime size.
- Fail before download when space is clearly insufficient.
- Stream concise progress during package install, model download, and model load.
- Show log path while waiting.

### 4. Improve port-conflict recovery

Current message says port 8000 belongs to “another process.” The live conflict was an old `python3 -m http.server` process.

Desired error:

- Show port, PID, and safe command identity when available.
- Offer copy-pasteable choices: stop identified process, select another port, or configure custom endpoint.
- Never kill process automatically without confirmation.

### 5. Add optional setup smoke commands

OpenRouter setup should offer an explicitly paid smoke request after non-paid checks. Simple Jev setup should run a cheap local smoke without rebuilding anything.

Suggested flow:

```text
Configuration valid. Run a paid OpenRouter smoke request now? [y/N]
```

Success should print one ready-to-copy command and concise result.

### 6. Prefer single-line examples

Multiline shell examples triggered bracketed-paste control characters (`^[[200~` and trailing `~`) and `command not found` in the live terminal.

Documentation and setup output should provide:

- One-line commands first.
- Multiline variants only as secondary readable examples.
- `bruv demo` or `bruv smoke <backend>` commands so users do not need to paste long invocations.

### 7. Improve provider error diagnosis

Generic `OpenRouter returned status 400` hid invalid model configuration. Preflight validation now prevents the observed `yes`/`y` mistake, but provider errors should still include safe structured details such as status category and request ID without exposing response bodies or credentials.

### 8. Surface model-mode quality limitations

`Qwen/Qwen3.5-0.8B` worked well for `choice` but produced nearly identical ~0.01 values for positive and negative `noul` examples.

- Mark model/backend results uncalibrated where appropriate.
- Add a small capability smoke that compares contrasting examples.
- Warn when a mode appears non-discriminating.
- Recommend choice-based yes/no as fallback instead of presenting a broken Noul score as useful.

## Release work still outstanding

- Add focused regression coverage for bugs found in live setup.
- Run formatter, lint, type checking, and full suite once.
- Build wheel and sdist.
- Verify clean-environment install and offline commands.
- Validate standalone artifacts for supported platforms.
- Generate checksums and SBOM.
- Update changelog.
- Tag release and create GitHub Release.
- Decide whether to publish to PyPI.

Current version remains `0.1.0`; no release tag exists yet.
