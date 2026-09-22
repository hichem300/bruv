# Managed Simple Jev Setup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Do not launch nested agents or workflows. User requested code-first execution: do not run tests until final verification task.

**Goal:** Make `bruv setup simple-jev` install and configure isolated local Simple Jev, then auto-start and reuse its server during normal bruv decisions.

**Architecture:** Keep heavyweight Simple Jev dependencies in managed data directory and expose lifecycle through focused runtime service. Persist managed mode in bruv configuration, call idempotent `ensure_running()` from Simple Jev backend factory, and expose explicit `serve simple-jev` controls through Typer.

**Tech Stack:** Python 3.12+, Typer, Pydantic, platformdirs, httpx, stdlib venv/subprocess/pathlib/json/socket.

---

## File structure

- Create `src/bruv/onboarding/simple_jev_runtime.py`: paths, manifest, install/repair, hardware detection, detached process lifecycle, locking, and health checks.
- Modify `src/bruv/config.py`: managed-mode fields, environment precedence, and atomic config persistence helper.
- Modify `src/bruv/onboarding/setup.py`: direct managed setup flow and backward-compatible interactive selection.
- Modify `src/bruv/backends/registry.py`: managed Simple Jev auto-start boundary.
- Modify `src/bruv/cli.py`: `setup [backend] [--repair]` and `serve simple-jev {start|stop|status}`.
- Modify `docs/installation.md`, `docs/configuration.md`, and `README.md`: exact user commands and managed-runtime behavior.
- Create `tests/unit/onboarding/test_simple_jev_runtime.py`: isolated runtime lifecycle tests using injected subprocess/HTTP boundaries.
- Modify `tests/integration/test_setup_doctor.py` and `tests/integration/test_cli_commands.py`: CLI setup/serve/auto-start contracts.

## Task 1: Managed runtime service

**Worker scope:** only `src/bruv/onboarding/simple_jev_runtime.py`. No tests. No other files.

- [ ] Create immutable `ManagedSimpleJevSettings` with model, base URL, device, dtype, source revision, context, and batch defaults.
- [ ] Use `platformdirs.user_data_dir("bruv", appauthor=False) / "simple-jev"` for source, virtual environment, manifest, PID, lock, and log paths.
- [ ] Pin upstream source revision `b02aa81c915a8193759b3cd33fef74721d6e005b`; default model `Qwen/Qwen3.5-0.8B`.
- [ ] Implement atomic JSON manifest read/write. Reject malformed state through unpaid `ConfigurationError` with `bruv setup simple-jev --repair` action.
- [ ] Implement install and repair: validate Python, pre-detect NVIDIA CUDA for setup preview, clone/fetch pinned upstream source, create isolated venv, install `./hf-server`, then verify CUDA through managed Python with `torch.cuda.is_available()` and fall back to CPU safely when unusable.
- [ ] Implement detached launch using managed Python and upstream `hf_server.py`; redirect stdout/stderr to managed log; persist PID atomically.
- [ ] Implement `status()`, `start()`, `stop()`, and `ensure_running()` with health check at `/health`, startup timeout, stale PID cleanup, occupied-port diagnosis, and an atomic-directory startup lock.
- [ ] Never stop process unless stored PID can be matched to Simple Jev command line. Refuse unsafe stop with actionable error.
- [ ] Keep subprocess and health operations injectable through narrow callables/classes so final tests require no install, download, or real daemon.
- [ ] Export only settings, path/service types, installer entry point, and lifecycle entry points needed by setup/CLI/registry.

## Task 2: Configuration, setup, CLI, and auto-start integration

**Worker scope:** `src/bruv/config.py`, `src/bruv/onboarding/setup.py`, `src/bruv/backends/registry.py`, `src/bruv/cli.py`, `README.md`, `docs/installation.md`, `docs/configuration.md`. No tests. Assume Task 1 public interface; adapt imports to actual exported names without editing Task 1 file.

- [ ] Add config fields `simple_jev_managed: bool = False`, `simple_jev_device`, and `simple_jev_dtype`; retain current base URL and model defaults.
- [ ] When `SIMPLE_JEV_BASE_URL` is supplied, force managed mode off so custom endpoint never auto-starts local process.
- [ ] Add atomic config update helper that preserves recognized existing values and writes a complete valid bruv TOML file with owner-only permissions where supported.
- [ ] Extend `run_setup` with optional preselected backend and repair flag while preserving existing interactive `bruv setup` behavior.
- [ ] Managed Simple Jev setup must show chosen model/device and first-download warning, confirm install, call runtime installer, persist backend/model/base URL/managed mode/device/dtype, start server, health-check it, execute one small sample decision, and print one ready command.
- [ ] Change CLI signature to support `bruv setup simple-jev` and `bruv setup simple-jev --repair`; keep TTY refusal for interactive setup but permit explicit backend execution when appropriate prompts still have TTY.
- [ ] Add Typer `serve` command with exact shape `bruv serve simple-jev start|stop|status`; reject other backend/action values with exit 2.
- [ ] In `_build_simple_jev`, call runtime `ensure_running()` only when `config.simple_jev_managed` is true, before constructing HTTP client.
- [ ] Keep all diagnostics on stderr where evaluation output could be JSON. Never expose secrets or raw subprocess environment.
- [ ] Document isolated install location, auto-start behavior, first model download, explicit lifecycle commands, repair, custom endpoint override, and public demo as non-managed alternative.

## Task 3: Integration review before tests

**Worker scope:** review current diff and fix production/docs files only. No tests and no test execution.

- [ ] Trace full flow: direct setup → install → persist → start → health → normal command → registry auto-start → adapter request.
- [ ] Check Linux, macOS, and Windows branches for path, detached process, PID checks, and file permissions.
- [ ] Check JSON stdout remains clean and all failures are unpaid configuration errors with concrete recovery commands.
- [ ] Remove duplicate logic and fix only concrete correctness gaps found. Do not expand scope.

## Task 4: End-only quick tests and verification

**Worker scope:** tests plus minimal fixes required by failures. This is first task allowed to run tests.

- [ ] Add focused runtime tests using temporary directories and fakes for install commands, process state, lock contention, stale PID, health timeout, occupied port, start reuse, safe stop refusal, and repair.
- [ ] Add CLI tests for `setup simple-jev`, `--repair`, serve start/stop/status, invalid action/backend, existing interactive setup compatibility, and non-interactive behavior.
- [ ] Add registry/config tests proving managed mode auto-starts once, healthy reuse is idempotent, custom `SIMPLE_JEV_BASE_URL` disables managed startup, and config writes reload correctly.
- [ ] Run focused suite once:

```bash
cd /root/business/PROJECTS/bruv
.venv/bin/python -m pytest \
  tests/unit/onboarding/test_simple_jev_runtime.py \
  tests/unit/test_config.py \
  tests/unit/backends/test_registry.py \
  tests/integration/test_setup_doctor.py \
  tests/integration/test_cli_commands.py -q
```

Expected: all focused tests pass.

- [ ] Fix only failures caused by this feature, then run full verification once:

```bash
cd /root/business/PROJECTS/bruv
git diff --check
.venv/bin/python -m pytest -q
```

Expected: clean diff and all tests pass.

- [ ] Run no-download smoke commands:

```bash
.venv/bin/bruv setup --help
.venv/bin/bruv serve --help
.venv/bin/bruv noul "Is this a refund request?" --state "refund me" --backend simple-jev --dry-run
```

Expected: setup and serve help render; dry run exits successfully without installing or launching Simple Jev.

- [ ] Commit implementation and tests together after verification, then push `main`.
