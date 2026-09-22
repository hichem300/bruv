# bruv browser — Design Document

Date: 2026-09-22
Status: Approved design (no implementation yet)

## 1. Purpose

`bruv browser` adds a goal-driven, backend-agnostic browser-automation capability to bruv. The agent observes a page, asks the selected bruv backend for decisions, executes safe DOM-derived actions through Playwright, and finishes when the backend confirms the goal. Feature ships as a subcommand group, not a product center.

Provenance: clean-room Python implementation inspired only by the documented, publicly observable behavior of hqman/jev-browser-skill. No source copying, no line-by-line translation, no vendoring, no Cline-derived code.

## 2. Scope

In scope:
- `bruv` CLI subcommands: `setup browser`, `browser run`, `browser state`, `browser reply --text`, `browser stop`, `browser shutdown`.
- Local daemon managing browser sessions and pending `needs_text` requests.
- Backend-agnostic decision loop over the existing bruv backend registry.
- Safety layer: action validation, origin allowlist, sensitive-input filtering, hard limits.
- Artifacts (screenshots and step transcript) under bruv user-data paths. Trace and video capture are opt-in because they can retain page data.
- Tests (written after implementation per project rule) and license inventory.

Non-goals (explicitly out of scope):
- New browser-provider abstraction or Vercel AI Gateway integration.
- Model-generated JavaScript, shell, raw selectors, or invented URLs/text.
- Automated login submission, payments, purchases, bulk posting, CAPTCHA solving.
- Distributed/cloud execution; single local daemon only.
- Changing any existing backend's contracts or calibration behavior.

## 3. Architecture and boundaries

### 3.1 Components

```
CLI (browser group)
  └── BrowserController          # application-layer orchestration
        ├── SessionStore         # sessions, statuses, persistence
        ├── Daemon               # lightweight local process serving sessions
        ├── ObservationBuilder   # DOM -> filtered snapshot -> candidates
        ├── ActionValidator      # allowlist + candidate checks
        ├── PlaywrightExecutor   # isolated context, action execution
        └── BackendDriver        # registry-selected backend, canonical requests
```

- Reuses existing bruv config, credentials, errors, output/JSON conventions, and exit codes. No parallel config system.
- Backend selection is whatever the current registry resolves: same config default backend, same `--backend` override, no silent fallback. All registered backends (TypeSafe, OpenRouter, Simple Jev, Needle, RLCD ModernBERT) participate through the existing canonical Noul and Choice request paths.
- Registry metadata exposes per-backend constraints the browser needs: supported answer modes, calibrated-only flag, candidate-count support (`per_k` calibration requirement for RLCD ModernBERT), context limits. The ObservationBuilder/BackendDriver consult these for batching and candidate-count decisions.
- The daemon is a thin supervisor: holds session state, runs/aborts loops, exposes state over a local socket or HTTP loopback. It never interprets pages itself.

### 3.2 Decision loop

```
observe page (visible DOM)
  -> build finite candidate action list
  -> backend Noul: is goal complete?
       yes -> finish
       no  -> backend Choice: pick next action from candidates
                -> validate action
                -> execute via Playwright
                -> repeat (bounded by max-steps and wall-clock)
  at any point: needs_text -> pause, surface via CLI
```

Candidate actions are restricted to:
- click on a known candidate element
- type into a known field with goal-provided or user-provided text
- navigate to an approved allowlisted URL
- scroll, wait
- `finish`, `request_text`, `request_review`

Never permitted: arbitrary JS evaluation, shell commands, raw selectors, model-invented URLs or text. Text originates only from the goal/command line or from the user via `needs_text`; the model never supplies secrets, payment, or private values.

Large pages: filter hidden and disabled elements, prioritize semantically meaningful elements (headings, buttons, links, form labels), and split remaining candidates into batches sized to the backend's context/candidate-count constraints, issuing multiple Choice rounds if needed.

### 3.3 Data flow

1. `browser run --url URL --goal TEXT [--backend X] [--headed|--headless] [--max-steps N]`
2. Controller validates allowlist, launches an isolated Playwright context with temporary profile state; persistent artifacts use bruv user-data paths. Trace and video are off unless explicitly enabled.
3. Each step produces an observation snapshot; backend calls use canonical Noul/Choice payloads.
4. Session state (`running | needs_text | needs_review | done | failed | stopped`) persists to the session store; CLI `browser state` reads it.
5. `needs_text` pauses the loop; user replies with `browser reply --text "..."`; loop resumes using that text only.
6. `browser stop` requests graceful halt; `browser shutdown` terminates daemon and closes contexts.
7. On completion, enabled artifacts are written under bruv user-data paths. bruv omits sensitive fields and supplied secret values from its own logs, but screenshots, traces, and videos can capture private page content; opt-in capture warns about this risk.

## 4. CLI / UX

- `bruv setup browser` — installs Playwright browser binaries, verifies versions, writes config entries, prints setup doctor summary using existing bruv output styling.
- `bruv browser run --url <url> --goal <text> [--backend <name>] [--headed | --headless] [--max-steps <n>]` — starts session (daemon autostarts if needed), streams progress, returns final status and artifact paths as bruv-conventional JSON.
- `bruv browser state [--session <id>]` — current status, pending request, last actions.
- `bruv browser reply --text <text> [--session <id>]` — answers `needs_text`.
- `bruv browser stop [--session <id>]` — graceful stop.
- `bruv browser shutdown` — stop daemon, close all contexts.

All JSON output follows existing bruv envelope conventions; exit codes reuse `exit_codes.py` semantics (including abstain/exit-11-style handling where a backend legitimately abstains).

## 5. Security

- Origin allowlist: navigation restricted to configured allowed origins. The allowlist must be explicit in config; default empty except the run URL's origin. Documented limitation: an allowlist cannot block every subresource request (images, fetches initiated by the page); it governs navigations and top-level actions only.
- Page content is untrusted input, never instructions.
- Observations omit password, file, hidden inputs, and `autocomplete`-sensitive fields entirely.
- Credentials and supplied sensitive values are never intentionally included in observations, structured output, logs, or step transcripts. Screenshots, traces, and videos may contain private page content, so trace/video capture is opt-in and all artifact-producing modes display that limitation.
- Hard limits: max steps and wall-clock deadline per session; exceeding either fails closed.
- Human review (`needs_review`) is required before: login submission, file upload/download, purchase/payment, destructive actions, public posting, message/form submission, permission grants, CAPTCHA interactions.
- Model review is advisory, not enforcement; validation and allowlisting are the enforcement layer.

## 6. Errors and fail-closed behavior

Fail closed (session -> `failed`, no retry, artifacts preserved) on:
- invalid or malformed backend response
- stale action (element no longer present/valid at execute time)
- disallowed navigation (non-allowlisted origin)
- model-suggested or user-prompted text targeting a sensitive input type
- calibrated-only backend receiving an unsupported candidate count (RLCD ModernBERT requires explicit `per_k` calibration; no fallback to global temperature)
- corrupt session state or artifact
- step/time limit exceeded

Statuses are exactly: `running`, `needs_text`, `needs_review`, `done`, `failed`, `stopped`. Backend quality and calibration differences (calibrated probabilities vs label-only modes) are disclosed in session metadata rather than hidden.

## 7. Test strategy

Tests are written after implementation (project rule), no TDD sequencing:
- DOM filtering (hidden/disabled/sensitive inputs excluded).
- Candidate generation and semantic prioritization.
- Capability/batching against registry metadata (context limits, `per_k` requirement).
- Action validation (allowlist, sensitive-input rejection, stale actions).
- Session store and daemon lifecycle (state transitions, reply, stop, shutdown).
- Fake-backend journeys covering each status and fail-closed path.
- Local fixture Chromium smoke test on a bundled static page.
- Contract checks for every registered backend via canonical Noul/Choice.
- Public install check including license inventory.
- No paid network calls in the normal suite; paid backend calls only in opt-in manual checks.

## 8. Public release and licensing

- Code authored fresh in this repository; contribution and security policies follow existing repo files.
- Dependency license inventory is a release requirement: Python Playwright, httpx, Pydantic, and any new dependency must appear in a maintained `THIRD_PARTY_NOTICES` file (or extend the existing NOTICE) with license identifiers and attribution. Adding a dependency with a license incompatible with the repo's distribution model is rejected at review.
- No content from hqman/jev-browser-skill or Cline is copied, vendored, or translated; the design cites behavior documentation only.

## 9. Acceptance criteria

1. `bruv setup browser` completes on a clean environment and doctor passes.
2. `browser run` drives a fixture page to `done` with a fake backend, producing valid artifacts and JSON output.
3. Each registered backend executes canonical Noul and Choice requests; RLCD ModernBERT refuses uncalibrated candidate counts instead of falling back.
4. Allowlist blocks non-approved navigation; sensitive inputs never appear in observations.
5. `needs_text` / `needs_review` / `stop` / `shutdown` flows work end to end via CLI.
6. All fail-closed triggers above produce `failed` with preserved, secret-free artifacts.
7. Test suite passes without network egress; license inventory is complete and current.