# bruv Browser Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `bruv browser`: a goal-driven, backend-agnostic browser-automation capability. The agent observes a page, asks the selected bruv backend (via existing registry, canonical Noul/Choice) for decisions, executes validated DOM-derived actions through Playwright, and finishes when the backend confirms the goal. Ships as CLI subcommand group plus local hardened loopback daemon.

**Architecture:** New `src/bruv/browser/` package. CLI (browser Typer group) -> BrowserController orchestrates SessionStore, Daemon, ObservationBuilder, ActionValidator, PlaywrightExecutor, BackendDriver. Backend selection via existing registry (`create_registered_backend`); browser compatibility is DERIVED from existing canonical `BackendCapabilities` (backend must support both `noul` and `choice`), so any current or future registered backend qualifies automatically. Domain models extend `bruv.domain.questions.CanonicalModel`. `models.py` owns ALL shared data models (BrowserGoal, PendingRequest, ElementFingerprint, CandidateAction, ElementMeta, Observation, ExecutedAction, ArtifactPaths, SessionState); `actions.py` owns only `ActionKind`/`ACTION_KINDS`/`validate_action_dict`; `observation.py` and `artifacts.py` import shared models from `models.py` and never redefine them. Runtime is fully synchronous: Playwright via `sync_playwright`, controller runs sync inside one dedicated daemon worker thread per session. State persists via `SessionStore` under bruv user-data paths; daemon exposes token-authenticated loopback IPC. Safety: origin allowlist, sensitive-field filtering, needs_review gates, hard step/time limits, fail-closed errors. Reuses config, credentials, output envelopes (`render_success`/`render_error`), exit codes, redaction.

**Tech Stack:** Python 3.11, Typer, Pydantic v2, httpx (client), stdlib `http.server` + `ThreadingHTTPServer` (daemon), Playwright (new optional dependency, Python `playwright` package), platformdirs, `secrets`. No Vercel. No TDD; behavior first, focused tests after.

---

## Project constraints (override generic skill)

- NO TDD / red-green. Implement bounded behavior, then add focused tests.
- Full formatter/lint/typecheck/suite once at final completion; focused checks per task.
- No worktrees. Work in current checkout; preserve unrelated changes. Never `git add -A`; stage only browser feature paths.
- No new provider abstraction; no backend contract changes. Browser support derives from existing capabilities; future registry backends work automatically when canonical capabilities qualify.
- No invented provider context limits. One conservative configurable observation budget shared by all backends by default (`browser_observation_budget_chars` in config).
- No Vercel. Clean-room: no code copied/transliterated from jev-browser-skill or Cline.
- Implementation delegated to `openrouter/z-ai/glm-5.3-flash` workers: narrow, non-overlapping file scopes; exact symbols in every task.
- Style: repo conventions — strict mypy, ruff line-length 100, `from __future__ import annotations`, frozen dataclasses/slots, `CanonicalModel` for wire models.

## File structure map (final state)

```
src/bruv/browser/__init__.py            # package exports
src/bruv/browser/models.py              # ALL shared domain models: BrowserGoal, SessionStatus,
                                        #   SessionState, PendingRequest, ElementFingerprint,
                                        #   ElementMeta, Observation, CandidateAction,
                                        #   ExecutedAction, ArtifactPaths
                                        #   (imports ActionKind + action schema from actions.py;
                                        #   observation.py and artifacts.py import from here,
                                        #   never redefine)
src/bruv/browser/actions.py             # ActionKind, canonical action schema, action dict validation
src/bruv/browser/observation.py         # ObservationBuilder: DOM -> filtered Observation -> candidates
                                        #   (imports ElementMeta/Observation from models.py)
src/bruv/browser/tournament.py          # CandidateTournament: valid batch sizing + recursive Choice tournament
src/bruv/browser/validation.py          # ActionValidator: allowlist, sensitive fields, stale actions, limits
src/bruv/browser/driver.py              # BackendDriver: registry backend -> Noul/Choice canonical calls
src/bruv/browser/executor.py            # PlaywrightExecutor: sync_playwright, isolated context, execution
src/bruv/browser/artifacts.py           # imports ArtifactPaths from models.py; screenshot ON,
                                        #   trace/video opt-in, privacy warning
src/bruv/browser/sessions.py            # SessionStore: statuses, pending requests, atomic persistence
src/bruv/browser/daemon.py              # BrowserDaemon (hardened loopback ThreadingHTTPServer) + DaemonClient
src/bruv/browser/controller.py          # BrowserController: sync decision loop, one action per observation
src/bruv/browser/setup_browser.py       # run_browser_setup (install, verify, opt-in doctor checks)
src/bruv/cli_browser.py                 # Typer browser command group, registered in cli.py
tests/unit/browser/__init__.py
tests/unit/browser/test_models.py
tests/unit/browser/test_observation.py
tests/unit/browser/test_tournament.py
tests/unit/browser/test_validation.py
tests/unit/browser/test_driver.py
tests/unit/browser/test_sessions.py
tests/unit/browser/test_daemon.py
tests/unit/browser/test_controller.py
tests/unit/browser/test_artifacts.py
tests/unit/browser/test_registry_browser.py
tests/unit/browser/test_config_browser.py
tests/integration/test_browser_cli.py
tests/integration/test_browser_fixture_smoke.py   # Playwright fixture Chromium smoke
tests/contract/browser/                           # per-backend canonical Noul/Choice browser checks
tests/fixtures/browser/fixture_site/index.html (+ assets)
docs/browser.md                        # user guide
THIRD_PARTY_NOTICES                    # license inventory (new file; NOTICE not modified)
```

Existing files touched (small diffs only): `src/bruv/cli.py`, `src/bruv/config.py`, `src/bruv/onboarding/doctor.py`, `src/bruv/backends/registry.py`, `pyproject.toml`, `README.md`. `src/bruv/onboarding/setup.py`, `src/bruv/domain/validation.py`, and `NOTICE` are NOT modified.

---

## Task 1: Browser domain models and action schema

**Files:** create `src/bruv/browser/__init__.py`, `src/bruv/browser/actions.py`, `src/bruv/browser/models.py`

**Behavior:** Wire-safe Pydantic models (extend `CanonicalModel` from `bruv.domain.questions`) and exactly the spec statuses/action kinds. Single ownership: `actions.py` owns `ActionKind` and the action dict schema; `models.py` imports them — no duplicate definitions.

Contracts (exact names):

```python
# actions.py
ActionKind = Literal["click", "type", "navigate", "scroll", "wait", "finish",
                     "request_text", "request_review"]

ACTION_KINDS: frozenset[str]  # derived from ActionKind literal values

# Canonical action dict shape (validated here and re-checked by ActionValidator, Task 4):
# {"kind": "click", "ref": "e12"}
# {"kind": "type", "ref": "e7", "text_source": "goal" | "user", "text_index": 0}
# {"kind": "navigate", "url": "https://..."}
# {"kind": "scroll", "direction": "down" | "up"}
# {"kind": "wait", "ms": 250..2000}
# {"kind": "finish"}
# {"kind": "request_text", "ref": "e7", "reason": "..."}
# {"kind": "request_review", "reason": "..."}
# Text is NEVER carried in the action or persisted; type actions reference text
# held only in daemon runtime memory by (text_source, text_index). Model-supplied
# free-form action dicts are never executed; the model only returns an opaque
# option ID (Task 7) mapped to a trusted candidate.

def validate_action_dict(action: dict[str, JsonValue]) -> None:
    """Raise ValueError on unknown kind, missing keys, unknown keys, or bad ranges."""
```

```python
# models.py — single owner of ALL shared browser models.
from bruv.browser.actions import ActionKind, validate_action_dict  # single source

SessionStatus = Literal["running", "needs_text", "needs_review", "done", "failed", "stopped"]

ReviewKind = Literal["login", "payment", "upload", "download", "post", "message",
                     "permission", "captcha", "destructive"]

class BrowserGoal(CanonicalModel):
    url: AnyHttpUrl
    goal: NonBlankString

class PendingRequest(CanonicalModel):
    request_id: NonBlankString
    kind: Literal["needs_text", "needs_review"]
    reason: NonBlankString
    created_at: float  # epoch seconds
    proposed_action: dict[str, JsonValue] | None = None  # trusted action for needs_review

class ElementFingerprint(CanonicalModel):
    tag: str
    role: str | None
    text_or_label: str | None
    href: str | None

class ElementMeta(CanonicalModel):
    ref: NonBlankString
    tag: NonBlankString
    role: str | None = None
    text: str | None = None
    label: str | None = None
    href: str | None = None
    input_type: str | None = None
    autocomplete: str | None = None
    disabled: bool
    visible: bool
    needs_review_kind: ReviewKind | None = None

class Observation(CanonicalModel):
    url: NonBlankString
    title: str
    elements: tuple[ElementMeta, ...]
    text_summary: str

class CandidateAction(CanonicalModel):
    """Trusted, builder/executor-derived candidate. The model never supplies one."""
    candidate_id: NonBlankString
    action: dict[str, JsonValue]      # canonical action dict (see actions.py)
    ref: str | None = None            # element ref for DOM-derived candidates
    fingerprint: ElementFingerprint | None = None  # for stale detection (Task 9)
    needs_review_kind: ReviewKind | None = None    # set for review-gated candidates

class ExecutedAction(CanonicalModel):
    step: int
    action: dict[str, JsonValue]
    ok: bool
    error: str | None = None

class ArtifactPaths(CanonicalModel):
    root: Path
    screenshot: Path
    transcript: Path
    trace: Path | None = None
    video: Path | None = None

class SessionState(CanonicalModel):
    session_id: NonBlankString
    goal: BrowserGoal
    backend: NonBlankString
    status: SessionStatus
    steps: tuple[ExecutedAction, ...] = ()
    pending: PendingRequest | None = None
    allowed_origins: tuple[str, ...] = ()
    max_steps: int
    deadline_epoch: float
    calibrated: bool
    error: str | None = None
    artifacts: ArtifactPaths | None = None
    # NO pending_text / user_text_supply fields: user reply text lives only in
    # daemon SessionRuntime memory (Task 10) and reaches the executor through an
    # injected in-memory text resolver (Task 9). It is never saved in session
    # JSON; if the daemon dies, pending text is lost and the session fails closed.
```

`observation.py` and `artifacts.py` import these models and never redefine them. Tuples are used for immutable `SessionState` collections.

**Steps:**
- [ ] Implement actions/models per contracts above. No placeholders. `validate_action_dict` is the only kind/key/range authority.
- [ ] Focused test after behavior: create `tests/unit/browser/__init__.py`, `tests/unit/browser/test_models.py` asserting: status literal membership, action dict schema round-trip and rejection of invalid dicts via `validate_action_dict`, `CanonicalModel` immutability (mutation raises `TypeError`), `ElementFingerprint` equality semantics, `SessionState` carries no text fields (no `pending_text`/`user_text_supply`), tuple fields, and optional `CandidateAction` synthetic fields (`ref`/`fingerprint`/`needs_review_kind` may be `None`).

**Verify:**
```bash
cd /root/business/PROJECTS/bruv
.venv/bin/python -m pytest tests/unit/browser/test_models.py -q
```
Expected: all pass.

**Commit:**
```bash
git add src/bruv/browser/__init__.py src/bruv/browser/models.py src/bruv/browser/actions.py tests/unit/browser/
git commit -m "feat(browser): domain models, statuses, action schema"
```

---

## Task 2: Derived browser compatibility in the registry (additive only)

**Files:** modify `src/bruv/backends/registry.py` only

**Behavior:** No invented per-provider context limits. No browser fields on `BackendCapabilities` or `BackendDefinition`. Browser compatibility is derived from existing canonical capabilities: a backend is browser-compatible iff `{"noul", "choice"} <= capabilities.question_types`. Candidate batch sizes derive from existing constraints: `ChoiceQuestion.criteria` bounds (min 2, max 50), `capabilities.explicit_abstention`, and `capabilities.supported_total_candidates`. All five current backends support `noul` and `choice`, so all qualify; future registry backends qualify automatically when their canonical capabilities do.

Contracts (additive only in `registry.py`):

```python
CHOICE_MIN_CRITERIA = 2   # ChoiceQuestion.criteria min_length
CHOICE_MAX_CRITERIA = 50  # ChoiceQuestion.criteria max_length

@dataclass(frozen=True, slots=True)
class BrowserBackendMeta:
    """Fully derived from BackendCapabilities; no independent flags."""
    supported: bool                       # {"noul","choice"} <= question_types
    requires_per_k_calibration: bool      # explicit_abstention and
                                          #   supported_total_candidates is not None

def derive_browser_backend_meta(capabilities: BackendCapabilities) -> BrowserBackendMeta:
    """Pure helper deriving BrowserBackendMeta from any BackendCapabilities.
    Lets tests exercise future/hypothetical capability sets without constructing
    BrowserBackendMeta directly."""

def browser_backend_meta(name: str) -> BrowserBackendMeta:
    """derive_browser_backend_meta(get_backend_definition(name).capabilities)."""

def browser_choice_batch_sizes(name: str) -> tuple[int, ...]:
    """Permitted per-batch candidate (criteria) counts, descending, all within
    [CHOICE_MIN_CRITERIA, CHOICE_MAX_CRITERIA].
    - supported_total_candidates is None: range(CHOICE_MIN_CRITERIA, CHOICE_MAX_CRITERIA + 1)
    - else: {total - (1 if explicit_abstention else 0) for total in
      supported_total_candidates} intersected with [2, 50]
    e.g. rlcd-modernbert -> {2,3,4,5,6,8,10,16,24}; others -> 2..50."""
```

No existing backend contract changes; no required constructor args anywhere; existing `BackendCapabilities` untouched.

**Steps:**
- [ ] Add the dataclass and two helpers to `registry.py`. No registry entry edits.
- [ ] Focused test after behavior: `tests/unit/browser/test_registry_browser.py` asserting: all five current names return `supported=True`; `browser_backend_meta("rlcd-modernbert").requires_per_k_calibration is True` and False for the others; `browser_choice_batch_sizes("rlcd-modernbert") == (24, 16, 10, 8, 6, 5, 4, 3, 2)`; sizes for other backends span 2..50; unknown name raises `ConfigurationError`. Also add a derived-compat test using `derive_browser_backend_meta` with a capabilities object carrying only `{"noul"}`, proving `supported=False` (no direct `BrowserBackendMeta` construction, no hypothetical registry entry).

**Verify:**
```bash
.venv/bin/python -m pytest tests/unit/browser/test_registry_browser.py tests/unit/backends/test_registry.py tests/unit/domain/test_validation.py -q
```
Expected: all pass (existing tests unbroken).

**Commit:**
```bash
git add src/bruv/backends/registry.py tests/unit/browser/test_registry_browser.py
git commit -m "feat(browser): derive browser compatibility and batch sizes from registry capabilities"
```

---

## Task 3: Config additions for browser

**Files:** modify `src/bruv/config.py`

**Behavior:** Additive `AppConfig` fields (non-secret, validated). Existing config files without these keys keep loading. One conservative observation budget shared by all backends by default; no per-provider limits.

Contracts:

```python
browser_allowed_origins: tuple[str, ...] = ()        # explicit allowlist; empty default
browser_max_steps: int = Field(default=25, ge=1, le=200)
browser_wall_clock_seconds: int = Field(default=300, ge=10, le=3600)
browser_headless: bool = True
browser_observation_budget_chars: int = Field(default=6000, ge=1000, le=50_000)
    # single conservative budget for page text + element lines sent to ANY backend;
    # deliberately small by default: it is a safety/resource bound, not a provider limit
browser_capture_trace: bool = False                  # opt-in; may capture page data
browser_capture_video: bool = False                  # opt-in; may capture page data
browser_artifacts_dir: Path | None = None            # override; default = user-data path
```

Screenshots are always ON (no config flag); `PRIVACY_WARNING` prints on every run that takes one (Task 8). Trace/video stay opt-in via the flags above.

Also add helper in `config.py`:

```python
def browser_data_dir() -> Path:
    """platformdirs.user_data_dir('bruv', appauthor=False) / 'browser'"""
```

**Steps:**
- [ ] Add fields + validators (origins: each entry must parse via `AnyHttpUrl` or be rejected; strip trailing `/`). Add `browser_data_dir()`. Extend `_toml_value` with tuple/list -> TOML array serialization so `browser_allowed_origins` round-trips as a TOML array (no silent stringification).
- [ ] Focused test after behavior: `tests/unit/browser/test_config_browser.py` asserting defaults, TOML round-trip via `update_config` including a regression that `browser_allowed_origins` persists as a TOML array and reads back as a tuple (never a string), invalid origin rejected with `ConfigurationError`, `browser_data_dir()` returns path containing `browser`.

**Verify:**
```bash
.venv/bin/python -m pytest tests/unit/browser/test_config_browser.py tests/unit/test_config.py -q
```

**Commit:**
```bash
git add src/bruv/config.py tests/unit/browser/test_config_browser.py
git commit -m "feat(browser): config fields for origins, limits, observation budget, capture"
```

---

## Task 4: ActionValidator (allowlist, sensitive fields, stale actions, limits)

**Files:** create `src/bruv/browser/validation.py`

**Behavior:** Pure validation, no I/O. Enforcement layer per spec section 5. Reuses `validate_action_dict` from `actions.py` for shape; adds session/origin/sensitive checks.

Contracts:

```python
class ActionRejected(Exception):
    """Fail-closed rejection; message is safe for output (no secrets)."""
    def __init__(self, code: str, message: str) -> None: ...

SENSITIVE_INPUT_TYPES = frozenset({"password", "file", "hidden"})
SENSITIVE_AUTOCOMPLETE = frozenset({"current-password", "new-password", "cc-number",
    "cc-cvc", "cc-exp", "one-time-code"})
# email, tel, and search inputs and autocomplete="off" are NOT sensitive.

class ActionValidator:
    def __init__(self, allowed_origins: tuple[str, ...], run_origin: str) -> None: ...

    def validate(self, candidate: CandidateAction, session: SessionState) -> None:
        """Validate trusted candidate action shape, session status/step/deadline,
        and action-specific bounds. click/type/request_text require candidate ref and
        fingerprint. navigate URL must equal candidate fingerprint.href or run URL,
        use http/https, and match run origin or explicit allowlist. wait is 250..2000;
        scroll is up/down. Review-gated candidates remain valid choices but controller
        must pause before executor. Model output never constructs CandidateAction."""

    def is_sensitive(self, element_meta: ElementMeta) -> bool:
        """ElementMeta imported from models.py: input type in SENSITIVE_INPUT_TYPES
        or autocomplete in SENSITIVE_AUTOCOMPLETE — exact lists only. Review-gated
        contexts (login/payment/upload/download/post/message/permission/captcha/
        destructive) are needs_review metadata, not sensitivity."""
```

**Steps:**
- [ ] Implement per contract. Origin comparison uses hostname + scheme + port normalization (reuse `urllib.parse.urlparse`; strip trailing slash).
- [ ] Focused test after behavior: `tests/unit/browser/test_validation.py` covering allowlist/run-origin rules, exact sensitive lists, free-form navigate rejection, required ref/fingerprint, malformed action keys, wait/scroll bounds, review-gated candidate acceptance for controller interception, and status/step/deadline fail-closed.

**Verify:**
```bash
.venv/bin/python -m pytest tests/unit/browser/test_validation.py -q
```

**Commit:**
```bash
git add src/bruv/browser/validation.py tests/unit/browser/test_validation.py
git commit -m "feat(browser): action validator with allowlist and sensitive-field rules"
```

---

## Task 5: ObservationBuilder (DOM observation, filtering, prioritization)

**Files:** create `src/bruv/browser/observation.py`

**Behavior:** Convert page DOM into filtered `Observation` + finite candidate list. Pure logic separated from Playwright extraction (extraction lands with executor, Task 9; ObservationBuilder consumes element metadata dicts so unit tests need no browser).

Contracts (`ElementMeta`, `Observation`, and `CandidateAction` come from `models.py`):

```python
class ObservationBuilder:
    def build(self, page_meta: dict[str, JsonValue]) -> Observation:
        """page_meta = {"url","title","elements":[{...}]} produced by executor."""

    def candidates(self, obs: Observation) -> tuple[CandidateAction, ...]:
        """Return at most 200 trusted candidates in priority order.
        DOM actions carry ref + fingerprint. Links become trusted navigate actions;
        controls become click actions. Review-gated controls keep their real action
        plus needs_review_kind so controller can pause without executing. Editable
        fields emit request_text actions containing their ref/fingerprint; after user
        reply, controller converts selected request into a trusted type action with
        memory-only text_source/text_index. Add bounded synthetic scroll-up,
        scroll-down, and wait actions with no ref/fingerprint. Emit no finish action:
        Noul decides completion. Exclude hidden, disabled, sensitive, and unlabeled
        elements."""
```

Filtering rules (spec 5): omit `input type` in {password, file, hidden} and
`autocomplete` in {current-password, new-password, cc-number, cc-cvc, cc-exp,
one-time-code} entirely from `elements`. No other type or autocomplete value is
sensitive (email/tel/search and autocomplete="off" are not).

**Steps:**
- [ ] Implement per contract. Keep file under ~250 lines; no Playwright import here.
- [ ] Focused test after behavior: `tests/unit/browser/test_observation.py` with inline `page_meta` fixtures: hidden/disabled/sensitive excluded; ref sequence stable; priority ordering; fingerprints present and stable; text trimmed.

**Verify:**
```bash
.venv/bin/python -m pytest tests/unit/browser/test_observation.py -q
```

**Commit:**
```bash
git add src/bruv/browser/observation.py tests/unit/browser/test_observation.py
git commit -m "feat(browser): observation builder with filtering and prioritization"
```

---

## Task 6: CandidateTournament (recursive Choice tournament, preserves RLCD per_k)

**Files:** create `src/bruv/browser/tournament.py`

**Behavior:** From ONE page observation, exactly ONE action is ultimately executed. When more candidates exist than one Choice call can rank, run a recursive tournament: partition candidates into valid batches; each multi-candidate batch gets exactly ONE Choice call selecting one finalist; singleton batches are carried forward WITHOUT any backend call; recurse on (finalists + carried singletons) until one candidate remains; then the controller executes it. The tournament NEVER executes actions and NEVER issues one action per batch.

Contracts:

```python
class CandidateTournament:
    def __init__(self, meta: BrowserBackendMeta,
                 choose: Callable[[tuple[CandidateAction, ...]], CandidateAction]) -> None:
        # choose(batch) returns the selected trusted CandidateAction directly;
        # ONLY the BackendDriver (Task 7) maps the model's opaque option ID back
        # to a candidate. Injected for tests with the same direct-return shape.

    def round_batches(self, candidates: tuple[CandidateAction, ...]) -> tuple[
            tuple[CandidateAction, ...], ...]:
        """Partition greedily so every batch's candidate count is in
        meta-derived browser_choice_batch_sizes (registry helper, Task 2).
        If candidates fit in one batch, return a single batch (zero tournament
        rounds beyond the one choose call). If no valid partition exists for a
        remainder (e.g. 1 leftover cannot merge), the leftover becomes a
        singleton batch carried without a call. Raises ActionRejected(
        code="per_k_unsupported", ...) only if a required multi-batch size is
        impossible — never falls back to a global temperature."""

    def select(self, candidates: tuple[CandidateAction, ...]) -> CandidateAction:
        """Recursive tournament:
        - 0 candidates -> ActionRejected("no_candidates", ...)
        - 1 candidate  -> returned immediately, no backend call
        - else partition via round_batches; winners = [the CandidateAction
          returned by choose(batch) for each multi batch] + [the singleton item
          for each singleton batch]; recurse select(winners).
        Invariant: total choose() calls == number of multi-candidate batches
        across all rounds; exactly one candidate survives; no action executed."""
```

RLCD batch sizing: batch candidate count + 1 abstention (`explicit_abstention`) must land in `supported_total_candidates` `{2,3,4,5,6,7,9,11,17,25}`, i.e. criteria sizes `{2,3,4,5,6,8,10,16,24}` — exactly `browser_choice_batch_sizes("rlcd-modernbert")`. `explicit_abstention` comes from `backend_capabilities(name)`; the tournament stays registry-agnostic via `BrowserBackendMeta` + size list passed in by the controller.

**Steps:**
- [ ] Implement per contract.
- [ ] Focused test after behavior: `tests/unit/browser/test_tournament.py`: single candidate -> no call; exact fit -> one call; recursive case (e.g. 60 candidates on rlcd: rounds 24+24+10+2 winners, carried singletons without calls, recurse) asserting the winner, the exact call count, and that every batch passed to `choose` is one of the permitted sizes (the fake `choose` returns a member of the batch directly, mirroring BackendDriver's direct-return contract); unsupported total raises `ActionRejected("per_k_unsupported")`; empty raises.

**Verify:**
```bash
.venv/bin/python -m pytest tests/unit/browser/test_tournament.py -q
```

**Commit:**
```bash
git add src/bruv/browser/tournament.py tests/unit/browser/test_tournament.py
git commit -m "feat(browser): recursive Choice tournament with per-backend batch constraints"
```

---

## Task 7: BackendDriver (canonical Noul/Choice)

**Files:** create `src/bruv/browser/driver.py`

**Behavior:** Build registry backend via existing `create_registered_backend(bruv.backends.registry.BackendBuildContext(config, credentials))` and issue canonical `DecisionRequest`s. No new provider abstraction. Requests carry bounded page text, title, URL, visible element descriptions, the goal, and recent actions. The model NEVER supplies an action dict — it returns an opaque option ID mapped back to a trusted candidate.

Contracts:

```python
class BackendDriver:
    def __init__(self, backend_name: str, config: AppConfig,
                 credentials: Credentials, *, client_port: object | None = None) -> None:
        # require config.backend == backend_name (CLI loads with backend_override);
        # then build via create_registered_backend(BackendBuildContext(
        #     config=config, credentials=credentials,
        #     selected_dependency_factory=(lambda c, k: client_port) if client_port else None))
        # Store definition = get_backend_definition(backend_name).

    def is_goal_complete(self, goal: BrowserGoal, obs: Observation,
                         recent_actions: tuple[ExecutedAction, ...]) -> bool:
        """One canonical Noul: questions={"goal": NoulQuestion(instructions=...)}
        with state = {"goal": goal.goal, "url": obs.url, "title": obs.title,
        "page_text": <obs.text_summary trimmed to browser_observation_budget_chars>,
        "elements": [<compact visible element descriptions: ref, tag, role,
        text-or-label, href>],
        "recent_actions": [...last 5 ExecutedAction.action dicts...]}.
        Reads DecisionResult.answers["goal"]:
        - NoulAnswer with value: use selected boolean directly
        - calibrated probability-only NoulAnswer: compare noul to 0.5
        - uncalibrated probability-only NoulAnswer: never threshold it; issue one
          canonical Choice with `complete` and `continue`, then use selected label
        - AbstainAnswer or malformed/wrong type: fail closed.
        This preserves honest uncalibrated semantics while keeping every backend usable."""

    def choose_action(self, goal: BrowserGoal, obs: Observation,
                      batch: tuple[CandidateAction, ...],
                      recent_actions: tuple[ExecutedAction, ...]) -> CandidateAction:
        """One canonical Choice for this tournament batch:
        criteria = {option_id: json.dumps(candidate.action, sort_keys=True)}
        with opaque sequential option ids "c0".."cN" (N < CHOICE_MAX_CRITERIA);
        state = same bounded context shape as is_goal_complete.
        Reads answers' ChoiceAnswer: answer.choice is an opaque option ID string;
        map it back to the trusted CandidateAction via an in-memory
        {option_id: candidate} dict built for this call.
        Unknown option ID -> ProviderResponseError (fail closed; the model can
        never inject its own action).
        AbstainAnswer -> raise ActionRejected("backend_abstained", ...)."""
```

Both methods reuse `bruv.domain.requests.DecisionRequest` and the backend's existing `evaluate()` path; validation via `validate_request(request, definition.capabilities)` before the call so RLCD per_k violations fail closed at build time (existing `total_candidates_not_supported` issue).

**Steps:**
- [ ] Implement per contract. Build one private `_state(...)` helper so Noul and Choice share the same bounded context (page text, title, URL, element descriptions, goal, recent actions).
- [ ] Focused test after behavior: `tests/unit/browser/test_driver.py` records canonical requests and asserts bounded state, opaque Choice mapping, unknown IDs/abstain/malformed fail closed, separate value-only Noul behavior, calibrated probability-only 0.1/0.5 behavior, and uncalibrated probability-only Noul triggering `complete|continue` Choice fallback without numeric thresholding. Also reject constructor config/backend mismatch.

**Verify:**
```bash
.venv/bin/python -m pytest tests/unit/browser/test_driver.py -q
```

**Commit:**
```bash
git add src/bruv/browser/driver.py tests/unit/browser/test_driver.py
git commit -m "feat(browser): backend driver issuing canonical Noul and Choice requests"
```

---

## Task 8: Artifacts and privacy defaults

**Files:** create `src/bruv/browser/artifacts.py`

**Behavior:** Import `ArtifactPaths` from `models.py`. Final screenshot and action-only transcript are created by default; trace/video remain opt-in. Screenshots can contain private page data. Transcript contains action metadata only, never page text or supplied text; page-visible typed text can still appear in screenshot/trace/video.

Contracts:

```python
def artifact_paths(session_id: str, *, capture_trace: bool,
                   capture_video: bool, base: Path | None = None) -> ArtifactPaths:
    """root = (base or browser_data_dir()) / session_id; screenshot/transcript
    are always present; trace/video paths exist only when enabled."""

def write_transcript(path: Path, steps: tuple[ExecutedAction, ...]) -> None:
    """JSONL action metadata only: step, canonical action refs/indexes, ok, error.
    Never write page text, daemon reply text, or resolved typed values."""

PRIVACY_WARNING = (
    "Browser artifacts can retain private page content. A final screenshot and "
    "action transcript are created for every run; trace and video are opt-in."
)
```

**Steps:**
- [ ] Implement per contract. Show `PRIVACY_WARNING` for every run before browser launch, not only when trace/video flags are enabled.
- [ ] Focused test after behavior: `tests/unit/browser/test_artifacts.py`: default includes screenshot/transcript but no trace/video; opt-in sets trace/video paths; transcript excludes page text and supplied text; path layout under base.

**Verify:**
```bash
.venv/bin/python -m pytest tests/unit/browser/test_artifacts.py -q
```

**Commit:**
```bash
git add src/bruv/browser/artifacts.py tests/unit/browser/test_artifacts.py
git commit -m "feat(browser): artifact paths and honest opt-in capture privacy defaults"
```

---

## Task 9: PlaywrightExecutor (synchronous)

**Files:** create `src/bruv/browser/executor.py`

**Behavior:** Isolated Chromium context via `sync_playwright()`; ALL methods and the context manager are synchronous — no async anywhere in the browser package. Executes validated actions; extracts page metadata for ObservationBuilder; re-validates trusted candidates against fresh element fingerprints at execute time; enforces origin at navigation.

Contracts:

```python
class ExecutorError(Exception):
    def __init__(self, code: str, message: str) -> None: ...

class PlaywrightExecutor:
    """Synchronous context manager. All methods sync; runs inside the daemon's
    dedicated worker thread for the session."""

    def __enter__(self) -> "PlaywrightExecutor":
        # self._playwright = sync_playwright().start(); launch Chromium from it;
        # create a temporary isolated context; enable tracing ONLY if capture_trace;
        # record_video ONLY if capture_video. Keep all objects on this worker thread;
        # close context/browser and call self._playwright.stop() in close/__exit__.

    def extract(self) -> dict[str, JsonValue]:
        """JS evaluation returning {"url","title","text", "elements":[ElementMeta-shaped dicts]}.
        Selects visible, enabled interactive elements: a[href], button, input:not([type=hidden]),
        select, textarea, [role=button|link|textbox]. Collects tag/role/text/label/href/
        input_type/autocomplete/disabled/visible. "text" is the bounded page text
        (trimmed to browser_observation_budget_chars). No third-party page JS injected
        beyond this read-only snapshot query."""

    def execute(self, candidate: CandidateAction, session: SessionState,
                resolve_text: Callable[[str, int], str]) -> ExecutedAction:
        """Re-validate the trusted candidate, re-extract live element, and compare
        ElementFingerprint (tag/role/text-or-label/href); mismatch/missing raises
        ExecutorError("stale_action"). Controller never calls execute when
        candidate.needs_review_kind is set.
        Execute click/type/navigate/scroll/wait. Navigate only to trusted observed
        href or run URL and re-check origin. Type resolves goal/user text through
        the in-memory callback; action/transcript never contains resolved text."""

    def screenshot(self, path: Path) -> None: ...
    def close(self, save_trace_to: Path | None) -> None: ...
    def __exit__(self, *exc: object) -> None: ...
```

Hard limits live in the controller; executor honors deadline via per-action Playwright timeouts (default 10s per action, `page.goto` 20s).

**Dependency:** Add to `pyproject.toml` under a new optional extra:
```toml
browser = ["playwright>=1.49,<2"]
```
(not in core dependencies; import Playwright lazily inside executor with a clear `ConfigurationError` when missing).

**Steps:**
- [ ] Add `browser` extra to `pyproject.toml` `[project.optional-dependencies]`.
- [ ] Implement executor per contract. Read-only `extract()` JS inline in executor. Sync API only.
- [ ] Focused test after behavior (no network, no browser launch): `tests/unit/browser/test_executor_unit.py` for lazy-import error mapping, fingerprint-match -> execute vs fingerprint-mismatch -> `ExecutorError("stale_action")`, and action->ExecutedAction mapping with a fake page object (struct implementing `goto`, `click`, `fill`, `keyboard`, `title`, `url` attributes used by executor).

**Verify:**
```bash
.venv/bin/python -m pytest tests/unit/browser/test_executor_unit.py -q
```

**Commit:**
```bash
git add pyproject.toml src/bruv/browser/executor.py tests/unit/browser/test_executor_unit.py
git commit -m "feat(browser): synchronous Playwright executor with fingerprint revalidation"
```

---

## Task 10: SessionStore and hardened Daemon/IPC

**Files:** create `src/bruv/browser/sessions.py`, `src/bruv/browser/daemon.py`

**Behavior:** Sessions persist as JSON under `browser_data_dir() / "sessions" / "<session_id>.json"`. Daemon is a thin supervisor on loopback IPC with mandatory bearer-token auth. Loopback binding alone is NOT treated as sufficient: every endpoint requires the token.

Contracts:

```python
# sessions.py
class SessionStore:
    def __init__(self, root: Path | None = None) -> None:  # default browser_data_dir()/sessions
    def create(self, goal: BrowserGoal, backend: str, allowed_origins: tuple[str, ...],
               max_steps: int, deadline_epoch: float, calibrated: bool) -> SessionState
    def load(self, session_id: str) -> SessionState   # corrupt file -> raise SessionCorruptError
    def save(self, state: SessionState) -> None       # atomic tmp+replace, 0600 perms
    def latest(self) -> SessionState                  # most recent by created order file mtime
class SessionCorruptError(Exception): ...

# daemon.py (no fixed DEFAULT_PORT)
CONNECTION_FILE = "connection.json"  # under browser_data_dir()/"daemon"/

@dataclass(frozen=True, slots=True)
class DaemonConnection:
    port: int
    pid: int
    token: str

@dataclass(slots=True)
class SessionRuntime:
    thread: Thread
    stop_event: Event
    condition: Condition
    controller: BrowserController
    reply_text: str | None = None  # memory-only, consumed and cleared by worker

def write_connection(conn: DaemonConnection, path: Path) -> None:
    """Atomic tmp+replace, 0600 perms; content: {"port","pid","token"}."""

def read_connection(path: Path | None = None) -> DaemonConnection:
    """Default path browser_data_dir()/"daemon"/connection.json; missing/corrupt ->
    ConfigurationError('daemon not running')."""

class BrowserDaemon:
    """stdlib http.server ThreadingHTTPServer bound to ("127.0.0.1", 0)
    (OS-assigned port). Hardening, all enforced in one request wrapper:
    - Authorization: Bearer <token> required on EVERY endpoint; random token via
      secrets.token_urlsafe(32), generated at startup
    - any request bearing an Origin header (browser context) -> 403
    - POST bodies over 1 MiB -> 413; malformed required JSON -> 400; empty-body
      stop/shutdown requests are allowed
    - connection file written atomically before serving; token never logged
    Endpoints (JSON responses; CLI renders envelopes via render_success/render_error):
      POST /sessions            -> start controller loop in a dedicated worker thread
      GET  /sessions/{id}       -> SessionState
      GET  /sessions            -> list summaries
      POST /sessions/{id}/reply {"text": ...}   -> answer needs_text
      POST /sessions/{id}/stop  -> graceful stop flag
      POST /shutdown            -> abort loops, close contexts, exit server
    One daemon-owned SessionRuntime per active session contains worker thread,
    stop Event, Condition, one-shot reply slot, and controller reference. Controller
    plus Playwright stay in that same worker thread. /reply stores text in runtime
    memory and signals Condition; /stop sets Event and signals Condition. HTTP
    handler threads never call Playwright. Reply text never enters SessionStore,
    connection files, transcripts, or logs; daemon death loses it and fails the
    waiting session closed. Daemon never interprets pages."""

class DaemonClient:
    """Sync httpx client; reads DaemonConnection from the connection file (or an
    explicit port/token for tests). is_alive() -> bool; autostart handled in CLI
    layer by spawning [sys.executable, '-m', 'bruv.browser.daemon'] detached
    (stdout/stderr to browser_data_dir()/daemon.log), then polling the connection
    file with a short deadline."""

def serve(store: SessionStore | None = None,
          connection_path: Path | None = None) -> None:
    """Blocking entry point for `python -m bruv.browser.daemon`."""
```

**Steps:**
- [ ] Implement `sessions.py` (store + corrupt handling + atomic 0600 writes).
- [ ] Implement `daemon.py` (server, token/Origin/size middleware, endpoints, connection file, client, module entry point).
- [ ] Focused tests after behavior: `tests/unit/browser/test_sessions.py` (create/save/load round-trip, corrupt raises, atomic write permissions asserted on POSIX); `tests/unit/browser/test_daemon.py` (OS-assigned port, temp connection path, reply wakes same worker, stop wakes waiter, shutdown transitions; reply text absent from saved JSON/logs; missing/wrong token -> 401; `Origin` header -> 403 even with valid token; oversized POST -> 413; empty stop/shutdown accepted).

**Verify:**
```bash
.venv/bin/python -m pytest tests/unit/browser/test_sessions.py tests/unit/browser/test_daemon.py -q
```

**Commit:**
```bash
git add src/bruv/browser/sessions.py src/bruv/browser/daemon.py tests/unit/browser/test_sessions.py tests/unit/browser/test_daemon.py
git commit -m "feat(browser): session store and token-authenticated loopback daemon"
```

---

## Task 11: BrowserController decision loop

**Files:** create `src/bruv/browser/controller.py`

**Behavior:** Implements spec 3.2 loop with statuses and fail-closed triggers (spec 6). Coordinates Tasks 2–10 pieces. ONE observation leads to EXACTLY ONE executed action: goal check (one Noul), then tournament (possibly several Choice calls) selects ONE finalist, then that one action executes; the loop re-observes after every executed action. Runs synchronously inside the daemon worker thread.

Contracts:

```python
class BrowserController:
    def __init__(self, session: SessionState, store: SessionStore,
                 config: AppConfig, credentials: Credentials,
                 *, driver_factory: Callable[[str], object] | None = None,
                 executor_factory: Callable[[], object] | None = None) -> None:
        # factories inject fakes in tests; default builds BackendDriver + PlaywrightExecutor

    def run(self, stop_check: Callable[[], bool],
            wait_for_text: Callable[[float], str | None],
            resolve_text: Callable[[str, int], str]) -> SessionState:
        """Sync loop per spec 3.2, held inside one worker thread and one live
        Playwright context:
        - extract -> ObservationBuilder.build -> goal-complete Noul
        - if incomplete, tournament selects ONE trusted candidate
        - needs_review_kind/request_review: persist needs_review with proposed_action,
          end without executing it; no approval command exists in v1
        - request_text: persist needs_text with target ref/fingerprint, then block
          through wait_for_text(deadline). Append returned text to a controller-local
          list, create a trusted type CandidateAction for same ref/fingerprint with
          text_source="user" and its memory index, then execute through resolve_text.
          None from waiter means stop/deadline/daemon loss -> fail closed
        - otherwise validate -> executor.execute(candidate, session, resolve_text),
          record action, then re-observe before selecting another
        - stop Event -> stopped; provider/action/limit/session error -> failed;
          goal complete -> final screenshot + transcript -> done
        Exactly one action executes per observation; gated actions never execute."""

    def request_stop(self) -> None:
        """Set stop state consumed by daemon runtime's Event-backed stop_check."""
```

needs_review gating set (spec 5, enforced at executor + controller): login submission, file upload/download, purchase/payment, destructive actions, public posting, message/form submission, permission grants, CAPTCHA interactions.

**Steps:**
- [ ] Implement per contract; keep under ~300 lines; inject factories.
- [ ] Focused test after behavior: `tests/unit/browser/test_controller.py` with fake driver + fake executor covering: done journey with one action per observation; recursive tournament; `needs_text` persists only target metadata, blocks, converts memory-only reply into trusted type action, resolves text only at executor call, and writes no text to state/transcript; stop wakes waiter; `needs_review` preserves proposed action and executes nothing; daemon-loss waiter returns `None` and fails closed; provider/action/max-step failures preserve safe metadata.

**Verify:**
```bash
.venv/bin/python -m pytest tests/unit/browser/test_controller.py -q
```

**Commit:**
```bash
git add src/bruv/browser/controller.py tests/unit/browser/test_controller.py
git commit -m "feat(browser): sync controller loop, one action per observation, fail-closed paths"
```

---

## Task 12: CLI browser group and setup browser (NOT a backend handler)

**Files:** create `src/bruv/cli_browser.py`, `src/bruv/browser/setup_browser.py`; modify `src/bruv/cli.py` (add sub-Typer + setup branch), `src/bruv/onboarding/doctor.py`. `src/bruv/onboarding/setup.py` is NOT modified with a `_browser_handler`.

**Behavior:** Spec section 4 commands; JSON via `render_success`/`render_error`; exit codes via `exit_code_for`. `browser` is NOT a backend: it must not enter `_HANDLERS`, the setup dispatch table, or the backend registry. `cli.setup()` branches on the exact argument `browser` to a dedicated `run_browser_setup` BEFORE normal backend dispatch; normal backend setup is unchanged.

Contracts:

```python
# cli_browser.py: browser_typer = typer.Typer(help="Browser automation sessions")
# Commands:
# run(url: str, goal: str, backend: str|None, headed: bool, headless: bool,
#     max_steps: int|None)   -> daemon autostart, POST /sessions, then poll
#     GET /sessions/{id} streaming progress lines to stderr (human) or final JSON to stdout;
#     maps abstain from backend to ABSTAINED (11) semantics via CommandOutcome(abstained=...)
# state(session: str|None)  -> latest or given session JSON envelope
# reply(text: str, session: str|None) -> POST reply; only valid from needs_text (else exit 2)
# stop(session: str|None)   -> POST stop
# shutdown()                -> POST shutdown
# No other new commands.

# cli.py:
# app.add_typer(browser_typer, name="browser")
# in setup(): first branch —
#   if backend == "browser":
#       result = run_browser_setup(prompt=_masked_prompt, confirm=typer.confirm,
#                                  print_line=typer.echo)
#       print result and return without calling run_setup()
# then existing backend dispatch remains unchanged.

# setup_browser.py
def run_browser_setup(*, prompt: PromptFn, confirm: ConfirmFn, print_line: PrintFn,
                      config_path: Path | None = None) -> SetupResult:
    """Dedicated browser setup, never registered as a decision backend.
    If Python Playwright is absent, fail with install action:
    pip install 'bruv[browser]'. Do not install Python packages automatically.
    After confirmation, install Chromium only with
    [sys.executable, '-m', 'playwright', 'install', 'chromium']; verify using
    the same checks as `bruv doctor --browser`; optionally persist a
    comma-separated origin allowlist through update_config(config_path); return
    SetupResult(backend="browser", persisted=..., next_command="bruv doctor --browser")."""

# doctor.py: browser checks are OPT-IN ONLY: add a --browser flag to the doctor
# command; when set, append _check_browser() results (playwright import present?
# chromium binary present via `python -m playwright install --dry-run` parse or
# importlib.metadata). Default run_doctor() item list unchanged, so ordinary
# installs without the browser extra never fail doctor.
```

`run` flags: `--headed/--headless` mutually exclusive (error if both, exit 2); `--capture-trace` / `--capture-video` remain opt-in. Every run prints `PRIVACY_WARNING` before starting because final screenshot/transcript are enabled by default.

**Steps:**
- [ ] Implement `cli_browser.py` with the five commands + `run`. Use `render_success`/`render_error` envelopes.
- [ ] Wire sub-Typer into `cli.py`; add the exact-argument `browser` branch in `cli.setup()`; add opt-in doctor check behind `--browser`.
- [ ] Focused tests after behavior: `tests/integration/test_browser_cli.py` using existing CLI test patterns from `tests/integration/test_cli_commands.py` (invoke via `typer.testing.CliRunner`): `browser state` with empty store, `reply` invalid-status error envelope, `shutdown` with daemon not running (graceful message, exit 0), `run` with injected fake (monkeypatch `DaemonClient`), `--headed --headless` rejection, `bruv setup browser` invokes `run_browser_setup` and does NOT hit the backend dispatch, `bruv doctor` without `--browser` contains no browser check.

**Verify:**
```bash
.venv/bin/python -m pytest tests/integration/test_browser_cli.py tests/unit/onboarding/test_doctor.py -q
```

**Commit:**
```bash
git add src/bruv/cli_browser.py src/bruv/browser/setup_browser.py src/bruv/cli.py src/bruv/onboarding/doctor.py tests/integration/test_browser_cli.py
git commit -m "feat(browser): CLI command group, dedicated setup browser, opt-in doctor checks"
```

---

## Task 13: Backend contract checks for browser driver

**Files:** create `tests/contract/browser/__init__.py`, `tests/contract/browser/test_browser_driver_contracts.py`

**Behavior:** Every registered backend executes canonical Noul and Choice through `BackendDriver` using existing fake/fixture patterns (`tests/contract/_shared.py`, `tests/contract/backend_contract.py`, provider fakes under `tests/fixtures/`). Asserts RLCD ModernBERT refuses uncalibrated candidate counts (existing `total_candidates_not_supported` path) instead of falling back. No paid network calls: use the same fake-client injection pattern as `tests/contract/_shared.py`.

**Steps:**
- [ ] Implement contract test: parametrize over `backend_names`; for each, build driver with injected fake dependency via `BackendBuildContext.selected_dependency_factory` (mirroring existing unit backend tests' fake-port pattern); assert `is_goal_complete` and `choose_action` produce `DecisionRequest` objects passing `validate_request` against that backend's `capabilities`, and results map correctly (including abstain fail-closed on both paths).
- [ ] Explicit rlcd-modernbert case: candidate count not in `supported_total_candidates` -> validation issue -> driver raises fail-closed error (no fallback).
- [ ] Derived-compatibility case: assert `browser_backend_meta(name).supported` is True for every registered backend and that a hypothetical capabilities object lacking `choice` derives `supported=False`.

**Verify:**
```bash
.venv/bin/python -m pytest tests/contract/browser/test_browser_driver_contracts.py -q
```

**Commit:**
```bash
git add tests/contract/browser/
git commit -m "test(browser): canonical Noul/Choice contract checks for every registered backend"
```

---

## Task 14: Fixture Chromium smoke test

**Files:** create `tests/fixtures/browser/fixture_site/index.html`, `tests/fixtures/browser/fixture_site/form.html`, `tests/integration/test_browser_fixture_smoke.py`

**Behavior:** Bundled static pages served by a test-local `ThreadingHTTPServer` on `127.0.0.1` with no external network: index has heading/buttons/link/text input; form has labeled field/submit. Smoke launches real headless Chromium against that HTTP origin and uses a fake backend. Skip cleanly when Playwright/Chromium is unavailable:

```python
pytestmark = pytest.mark.skipif(not _playwright_available(), reason="playwright/chromium not installed")
```

**Steps:**
- [ ] Write fixture pages (plain HTML, no external assets).
- [ ] Implement success smoke: fake backend navigates, clicks, types into an unsent ordinary field, then returns goal-complete through Noul. Assert `done`, final screenshot/transcript exist, JSON envelope renders, and typed text is absent from transcript.
- [ ] Add safety smoke: password field is excluded; selecting submit produces `needs_review` with proposed action and executor records zero execution for it.

**Verify:**
```bash
.venv/bin/python -m pytest tests/integration/test_browser_fixture_smoke.py -q
```
Expected: pass when chromium installed; skip otherwise. Install for local run: `.venv/bin/pip install -e '.[browser]' && .venv/bin/python -m playwright install chromium`.

**Commit:**
```bash
git add tests/fixtures/browser/ tests/integration/test_browser_fixture_smoke.py
git commit -m "test(browser): fixture Chromium smoke journey"
```

---

## Task 15: Docs, THIRD_PARTY_NOTICES, public install

**Files:** create `docs/browser.md`, `THIRD_PARTY_NOTICES`; modify `README.md`

**Steps:**
- [ ] `docs/browser.md`: user guide covering six commands, origin allowlist, memory-only `needs_text`, terminal `needs_review` behavior, artifact locations, warning shown every run, final screenshot/transcript enabled by default, trace/video off by default, private-page capture risk, allowlist limitation, token-authenticated loopback daemon, one-action-per-observation tournament, statuses/failures, calibration disclosure, and automatic compatibility for any registry backend supporting canonical Noul+Choice.
- [ ] `THIRD_PARTY_NOTICES`: entries for Python Playwright (Apache-2.0), plus verify existing inventory covers httpx (BSD-3), Pydantic (MIT), platformdirs (MIT), pyyaml, rich, typer. License identifiers + attribution lines; note that any added dependency with an incompatible license is rejected.
- [ ] `README.md`: add short "Browser" section linking `docs/browser.md`, one example `bruv browser run` command.
- [ ] Public install check: extend existing packaging test pattern — `tests/packaging/test_browser_packaging.py` asserting the `browser` extra resolves (parse `pyproject.toml`, check `browser` extra exists and licenses of its deps recorded in THIRD_PARTY_NOTICES) following the style of `tests/packaging/test_public_repo.py`.

**Verify:**
```bash
.venv/bin/python -m pytest tests/packaging/ -q
```

**Commit:**
```bash
git add docs/browser.md THIRD_PARTY_NOTICES README.md tests/packaging/
git commit -m "docs(browser): user guide, license inventory, packaging check"
```

---

## Task 16: Final full verification

**Steps:**
- [ ] Full formatter, lint, type check, and suite (project rule: once at completion):

```bash
cd /root/business/PROJECTS/bruv
.venv/bin/python -m ruff format --check src tests
.venv/bin/python -m ruff check src tests
.venv/bin/python -m mypy
.venv/bin/python -m pytest
```
Expected: all pass, no network egress (paid backends only via fake ports). Smoke `bruv --help` shows `browser` group:
```bash
.venv/bin/bruv browser --help
```
- [ ] Acceptance checklist against spec section 9 (1–7) verified by the tests above; record results in commit message body.
- [ ] Commit any final formatting fixes. NEVER `git add -A`: stage only browser feature paths, preserving unrelated working-tree files:

```bash
git add src/bruv/browser/ src/bruv/cli_browser.py src/bruv/cli.py \
  src/bruv/config.py src/bruv/onboarding/doctor.py \
  src/bruv/backends/registry.py pyproject.toml README.md \
  THIRD_PARTY_NOTICES docs/browser.md \
  tests/unit/browser/ tests/integration/test_browser_cli.py \
  tests/integration/test_browser_fixture_smoke.py \
  tests/contract/browser/ tests/fixtures/browser/ tests/packaging/test_browser_packaging.py
git commit -m "chore(browser): final formatting and verification pass"
```
- [ ] Before committing, run `git status --short` and confirm every staged path belongs to the browser feature; if unrelated modified files appear, leave them unstaged and untouched.