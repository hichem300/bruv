# bruv Browser Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `bruv browser`, using any registered bruv backend to choose safe browser actions through canonical Noul and Choice requests.

**Architecture:** Python Playwright provides browser control. bruv observes visible page elements, builds trusted action candidates, asks selected backend to choose among them, validates choice, executes one action, then observes again. Small local daemon keeps browser session alive for `state`, `reply`, `stop`, and `shutdown` commands.

**Tech Stack:** Python 3.11, Typer, Pydantic, httpx, stdlib HTTP server, Python Playwright.

---

## Guardrails

- Clean-room implementation. Do not copy, translate, or vendor `jev-browser-skill` or Cline source.
- No Vercel AI Gateway integration.
- Use existing bruv backend registry, config, credentials, result models, and output conventions.
- Any backend supporting canonical `noul` and `choice` is browser-compatible automatically.
- Never silently switch backends.
- Model selects opaque candidate IDs. It never supplies selectors, JavaScript, shell commands, URLs, or free-form action payloads.
- Implement behavior first, then add focused tests. No TDD sequence.
- Work in current checkout. No worktrees. Preserve unrelated changes.

## Final file layout

```text
src/bruv/browser/
  models.py          shared browser models and statuses
  actions.py         action schema and validation
  observation.py     page snapshot filtering and candidate creation
  selection.py       backend-aware candidate batching/tournament
  driver.py          canonical Noul/Choice calls through registry backend
  executor.py        synchronous Playwright execution
  controller.py      one-observation/one-action loop
  sessions.py        safe persisted session metadata
  daemon.py          authenticated local session process
  artifacts.py       screenshot/transcript paths and privacy warning
  setup_browser.py   Chromium setup and checks
src/bruv/cli_browser.py

tests/unit/browser/
tests/integration/test_browser_cli.py
tests/integration/test_browser_fixture_smoke.py
tests/contract/browser/
docs/browser.md
THIRD_PARTY_NOTICES
```

## Task 1: Core contracts, config, and backend compatibility

**Files:**
- Create: `src/bruv/browser/__init__.py`
- Create: `src/bruv/browser/models.py`
- Create: `src/bruv/browser/actions.py`
- Modify: `src/bruv/config.py`
- Modify: `src/bruv/backends/registry.py`
- Test: `tests/unit/browser/test_models.py`
- Test: `tests/unit/browser/test_config.py`
- Test: `tests/unit/browser/test_registry.py`

- [ ] Define statuses: `running`, `needs_text`, `needs_review`, `done`, `failed`, `stopped`.
- [ ] Define immutable models for goal, observation, element fingerprint, trusted candidate, pending request, executed action, artifacts, and session state.
- [ ] Keep supplied text out of persisted session models. Persist references and safe metadata only.
- [ ] Define validated actions: click, type, navigate, scroll, wait, request text, request review.
- [ ] Add browser config: origin allowlist, step/time limits, headless mode, observation size, trace/video flags, and artifact directory.
- [ ] Extend TOML serialization so origin tuples persist as proper arrays.
- [ ] Add registry helpers that derive browser compatibility from existing `BackendCapabilities`. Do not add independent browser support flags to each backend.
- [ ] Derive valid Choice batch sizes from existing `ChoiceQuestion` bounds, `explicit_abstention`, and `supported_total_candidates`; preserve RLCD `per_k` rules.
- [ ] Add focused model/config/registry tests.

**Verify:**

```bash
cd /root/business/PROJECTS/bruv
.venv/bin/python -m pytest tests/unit/browser/test_models.py tests/unit/browser/test_config.py tests/unit/browser/test_registry.py -q
```

**Commit:** `feat(browser): add core contracts and backend capability derivation`

## Task 2: Page observation and candidate selection

**Files:**
- Create: `src/bruv/browser/observation.py`
- Create: `src/bruv/browser/selection.py`
- Test: `tests/unit/browser/test_observation.py`
- Test: `tests/unit/browser/test_selection.py`

- [ ] Convert raw DOM metadata into bounded observations containing URL, title, visible page text, and interactive elements.
- [ ] Exclude hidden, disabled, password, file, payment-autocomplete, and one-time-code fields.
- [ ] Keep ordinary email, telephone, and search fields available.
- [ ] Build trusted candidates with opaque IDs and element fingerprints. Links become allowlist-checked navigation candidates; controls become click or request-text candidates.
- [ ] Mark login submission, uploads/downloads, payment, destructive actions, public posting, messages/forms, permissions, and CAPTCHAs with `needs_review_kind`.
- [ ] Add bounded scroll and wait candidates. Goal completion comes from Noul; no finish candidate needed.
- [ ] Select one candidate per observation. If candidates exceed backend Choice limits, use recursive batches and finalists. Carry singleton tails without model calls. Never execute one action per batch.
- [ ] Add focused filtering, fingerprint, ordering, batching, and RLCD candidate-count tests.

**Verify:**

```bash
.venv/bin/python -m pytest tests/unit/browser/test_observation.py tests/unit/browser/test_selection.py -q
```

**Commit:** `feat(browser): add safe observation and candidate selection`

## Task 3: Playwright execution and artifacts

**Files:**
- Create: `src/bruv/browser/executor.py`
- Create: `src/bruv/browser/artifacts.py`
- Modify: `pyproject.toml`
- Test: `tests/unit/browser/test_executor.py`
- Test: `tests/unit/browser/test_artifacts.py`

- [ ] Add optional dependency: `browser = ["playwright>=1.49,<2"]`.
- [ ] Use synchronous Playwright inside one dedicated session thread.
- [ ] Launch isolated Chromium context with temporary profile state.
- [ ] Re-read element and compare fingerprint immediately before click/type/navigation. Fail closed on stale candidates.
- [ ] Allow navigation only to run origin or configured origins, and only when URL came from trusted page evidence or initial goal URL.
- [ ] Reject arbitrary JavaScript, raw selectors, shell execution, uploads, and sensitive-field typing.
- [ ] Resolve typed text only from goal data or memory-only user reply. Never include resolved text in action descriptions or transcript.
- [ ] Produce final screenshot and action-only transcript by default. Keep trace/video opt-in.
- [ ] Print privacy warning because screenshots, traces, and videos may capture private page content.
- [ ] Add focused executor and artifact tests using fake page objects.

**Verify:**

```bash
.venv/bin/python -m pytest tests/unit/browser/test_executor.py tests/unit/browser/test_artifacts.py -q
```

**Commit:** `feat(browser): add safe Playwright executor and artifacts`

## Task 4: Backend driver and browser loop

**Files:**
- Create: `src/bruv/browser/driver.py`
- Create: `src/bruv/browser/controller.py`
- Test: `tests/unit/browser/test_driver.py`
- Test: `tests/unit/browser/test_controller.py`

- [ ] Build selected backend through existing registry and require config backend to match CLI selection.
- [ ] Send bounded goal, URL, title, page text, visible element descriptions, and recent actions in canonical requests.
- [ ] Use Noul for goal completion. Accept direct boolean selection. Threshold probability-only Noul only when result is calibrated.
- [ ] For uncalibrated probability-only Noul, use a two-option `complete`/`continue` Choice fallback instead of treating score as probability.
- [ ] Use Choice for candidate selection and map returned opaque ID back to trusted in-memory candidate.
- [ ] Fail closed on abstention, malformed response, unknown option ID, stale action, limit breach, or provider error.
- [ ] Execute exactly one browser action, then observe again.
- [ ] On request text, persist only pending metadata, wait for memory-only reply, convert it into trusted type action, execute in same Playwright thread, and discard supplied text after use.
- [ ] On review-gated candidate, save `needs_review` plus proposed safe metadata and execute nothing.
- [ ] Add focused driver/controller tests, including all fail-closed paths.

**Verify:**

```bash
.venv/bin/python -m pytest tests/unit/browser/test_driver.py tests/unit/browser/test_controller.py -q
```

**Commit:** `feat(browser): add backend-neutral decision loop`

## Task 5: Session store and local daemon

**Files:**
- Create: `src/bruv/browser/sessions.py`
- Create: `src/bruv/browser/daemon.py`
- Test: `tests/unit/browser/test_sessions.py`
- Test: `tests/unit/browser/test_daemon.py`

- [ ] Persist session metadata atomically under bruv user-data directory. Use restrictive permissions where platform supports them.
- [ ] Keep reply text only in active process memory. Never save it to session files, transcripts, connection files, or logs.
- [ ] Run controller and Playwright in same worker thread for full session lifetime.
- [ ] Keep `needs_text` browser session open while worker waits on condition. Reply wakes same worker. Stop wakes blocked worker.
- [ ] Bind daemon to `127.0.0.1` on OS-selected port.
- [ ] Generate random bearer token, store port/PID/token in protected connection file, require token on every endpoint, reject requests carrying browser `Origin` header, and cap POST bodies.
- [ ] Support create session, state, reply, stop, and shutdown endpoints.
- [ ] Add focused lifecycle, authentication, origin, body-size, corruption, and memory-only text tests.

**Verify:**

```bash
.venv/bin/python -m pytest tests/unit/browser/test_sessions.py tests/unit/browser/test_daemon.py -q
```

**Commit:** `feat(browser): add authenticated local session daemon`

## Task 6: CLI, setup, and doctor

**Files:**
- Create: `src/bruv/cli_browser.py`
- Create: `src/bruv/browser/setup_browser.py`
- Modify: `src/bruv/cli.py`
- Modify: `src/bruv/onboarding/doctor.py`
- Test: `tests/integration/test_browser_cli.py`

- [ ] Add commands: `browser run`, `state`, `reply`, `stop`, `shutdown`.
- [ ] Support configured default backend and explicit `--backend` override.
- [ ] Add `bruv setup browser` as dedicated setup branch, not backend registry entry. Detect missing Python extra and instruct `pip install 'bruv[browser]'`; install Chromium only with `python -m playwright install chromium`.
- [ ] Add opt-in `bruv doctor --browser`; normal doctor must not fail when optional browser extra is absent.
- [ ] Use existing human/JSON output and exit-code conventions.
- [ ] Print artifact privacy warning before every run.
- [ ] Add focused CLI tests with fake daemon client.

**Verify:**

```bash
.venv/bin/python -m pytest tests/integration/test_browser_cli.py tests/unit/onboarding/test_doctor.py -q
```

**Commit:** `feat(browser): add CLI setup and diagnostics`

## Task 7: Cross-backend and Chromium verification

**Files:**
- Create: `tests/contract/browser/test_backends.py`
- Create: `tests/fixtures/browser/index.html`
- Create: `tests/fixtures/browser/form.html`
- Create: `tests/integration/test_browser_fixture_smoke.py`

- [ ] Verify canonical Noul/Choice requests against fake ports for every registered backend. No paid calls.
- [ ] Verify RLCD receives only calibrated candidate totals.
- [ ] Serve fixtures from test-local HTTP server; make no external network requests.
- [ ] Run real Chromium journey: navigate, click, type into unsent normal field, then finish through goal-complete decision.
- [ ] Verify password fields are absent and submit candidate reaches `needs_review` without execution.
- [ ] Skip Chromium smoke cleanly when browser binary is unavailable.

**Verify:**

```bash
.venv/bin/python -m pytest tests/contract/browser/test_backends.py tests/integration/test_browser_fixture_smoke.py -q
```

**Commit:** `test(browser): verify every backend and Chromium journey`

## Task 8: Public documentation and release checks

**Files:**
- Create: `docs/browser.md`
- Create: `THIRD_PARTY_NOTICES`
- Modify: `README.md`
- Create: `tests/packaging/test_browser_packaging.py`

- [ ] Document install, commands, supported backends, origin rules, statuses, review behavior, artifacts, privacy limits, and daemon security.
- [ ] State clearly that backend quality and calibration differ; no silent fallback.
- [ ] Record Playwright license and verify licenses for distributed dependencies. Do not copy notices from `jev-browser-skill` because no source is reused.
- [ ] Add packaging test for browser extra and required notice.
- [ ] Run full formatter, lint, type check, tests, and CLI smoke once.

**Final verification:**

```bash
cd /root/business/PROJECTS/bruv
.venv/bin/python -m ruff format --check src tests
.venv/bin/python -m ruff check src tests
.venv/bin/python -m mypy
.venv/bin/python -m pytest
.venv/bin/bruv browser --help
```

Expected: all checks pass; no paid network calls; browser help lists five commands.

**Commit:** `docs(browser): document public browser feature and licensing`

## Completion criteria

- Every registered Noul+Choice backend can drive browser loop.
- One trusted action executes per fresh observation.
- RLCD never uses unsupported candidate total.
- Sensitive fields stay outside model observation.
- Review-gated actions never execute automatically.
- User reply text stays memory-only.
- Local daemon requires bearer token and rejects browser-origin requests.
- Public package installs with `bruv[browser]` and passes full verification.
