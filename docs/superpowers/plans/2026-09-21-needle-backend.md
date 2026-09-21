# Needle 3 Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional, local Needle 3 backend supporting bruv `noul`, `choice`, and `score` decisions without fabricating probability distributions.

**Architecture:** Add a lazy in-process adapter behind `DecisionBackend`, with a narrow injectable runtime port around `cactus-needle`. Evaluate each question independently using constrained extraction, then map selected values plus Needle whole-response confidence into expanded canonical answer contracts. Keep Needle optional through `bruv[needle]`, disable telemetry, and leave existing TypeSafe and Simple Jev payloads unchanged.

**Tech Stack:** Python 3.11+, Pydantic 2, Typer, cactus-needle optional runtime, pytest, Ruff, mypy, Hatchling.

**Execution constraints:** Work in current checkout. Do not create a worktree. Implement behavior first, then add/update tests. Use subagents sequentially with `openai-codex/gpt-5.6-sol`; delegated agents must not launch subagents or workflows.

---

## File map

**Create**
- `src/bruv/backends/needle.py` — runtime protocol, lazy Cactus runtime wrapper, question schemas, response mapping, errors.
- `tests/unit/backends/test_needle.py` — fake-runtime adapter coverage.
- `tests/contract/test_needle_contract.py` — canonical backend contract and confidence-only JSON shape.

**Modify**
- `pyproject.toml` — optional `needle` dependency extra.
- `src/bruv/domain/results.py` — confidence-only canonical answer variants and `needle` backend identifier.
- `src/bruv/config.py` — `needle` backend/config support.
- `src/bruv/backends/factory.py` — capability and adapter construction.
- `src/bruv/onboarding/setup.py` — Needle setup path.
- `src/bruv/onboarding/doctor.py` — offline Needle dependency/platform/cache checks.
- `src/bruv/output/terminal.py` — selected value plus confidence rendering.
- `src/bruv/contracts/spec.py` — advertise Needle backend and conditional probability semantics.
- `src/bruv/cli.py` — accept Needle in typed backend CLI options if choices are declared locally.
- `tests/unit/test_config.py`, `tests/unit/backends/test_factory.py`, `tests/unit/test_application.py`, `tests/unit/test_smoke.py` — domain/config/factory checks.
- `tests/integration/test_cli_commands.py`, `tests/integration/test_setup_doctor.py`, `tests/integration/test_cli_errors.py` — CLI/onboarding/error behavior.
- `tests/contract/test_machine_contracts.py`, `tests/contract/test_contract_snapshots.py`, `tests/fixtures/cli/output.schema.json`, `tests/fixtures/cli/spec.json` — machine contract updates.
- `README.md`, `docs/installation.md`, `docs/configuration.md`, `docs/cli-reference.md`, `CHANGELOG.md` — install, use, semantics, compatibility.

## Task 1: Expand canonical results honestly

- [ ] Add Needle backend identifier and confidence-only answer modes in `src/bruv/domain/results.py`.

Use these shapes:

```python
class NoulAnswer(CanonicalModel):
    type: Literal["noul"] = "noul"
    noul: Probability | None = None
    value: bool | None = None
    confidence: Probability | None = None

    @model_validator(mode="after")
    def validate_answer_mode(self) -> NoulAnswer:
        probability_mode = self.noul is not None
        selection_mode = self.value is not None and self.confidence is not None
        if probability_mode == selection_mode:
            raise ValueError("noul answer must contain exactly one answer mode")
        return self
```

For `ChoiceAnswer` and `ScoreAnswer`, change `probabilities` to an optional field defaulting to `None`. Run distribution validation only when present. Preserve required `choice`/`score`, `confidence`, and `legend`. Expand `DecisionResult.backend` to `Literal["typesafe", "simple-jev", "needle"]`.

- [ ] Update `src/bruv/output/terminal.py`: render Needle noul as `true|false (confidence: 0.XXX)` and append confidence to confidence-only choice/score answers; preserve existing provider output exactly.
- [ ] Run `ruff format` and `mypy` on changed files.
- [ ] Commit: `feat: support confidence-only decision answers`.

## Task 2: Implement Needle adapter and runtime boundary

- [ ] Create `src/bruv/backends/needle.py` with:

```python
class NeedleRuntime(Protocol):
    def classify(self, *, text: str, schema: dict[str, object], description: str) -> NeedleSelection: ...

@dataclass(frozen=True, slots=True)
class NeedleSelection:
    value: str | bool
    confidence: float
    suppressed: bool = False

class CactusNeedleRuntime:
    def __init__(self) -> None: ...
    def classify(self, *, text: str, schema: dict[str, object], description: str) -> NeedleSelection: ...

class NeedleAdapter(DecisionBackend):
    capabilities = BackendCapabilities(
        backend="needle",
        question_types=frozenset({"noul", "choice", "score"}),
        calibrated=True,
        allows_json_state=True,
    )

    def __init__(self, runtime: NeedleRuntime, model: str = "Cactus-Compute/needle3") -> None: ...
    def evaluate(self, request: DecisionRequest) -> DecisionResult: ...
```

- [ ] Set `NEEDLE_TELEMETRY=0` with `os.environ.setdefault` before lazy `import needle`. Never overwrite explicit stricter `DO_NOT_TRACK` settings.
- [ ] Build one raw JSON schema per question. Choice enum values are criterion keys. Noul schema uses boolean. Score enum values are string indices (`"0"`, `"1"`, ...). Include instructions and JSON-serialized criterion descriptions in description text.
- [ ] Serialize string state unchanged and non-string state via deterministic `json.dumps(..., sort_keys=True, separators=(",", ":"), ensure_ascii=False)`.
- [ ] Use one runtime `classify` call per question. Map:
  - noul → `NoulAnswer(value=selected_bool, confidence=confidence)`
  - choice → `ChoiceAnswer(choice=selected_key, confidence=confidence, probabilities=None)`
  - score → selected integer index as float, confidence, indexed legend, no probabilities
- [ ] Return `DecisionResult(backend="needle", model="Cactus-Compute/needle3", calibrated=True, ...)` with provider metadata declaring `confidence_scope="whole_response"` and `probabilities_available=False`.
- [ ] Map missing package to `ConfigurationError` with action `Install Needle support with: pip install 'bruv[needle]'`. Map initialization/download/platform failures to `BackendUnavailableError`; empty/suppressed calls and malformed values to non-paid `ProviderResponseError` with safe messages. Never include prompts or provider output in errors.
- [ ] Run targeted Ruff/mypy import checks.
- [ ] Commit: `feat: add local Needle backend adapter`.

## Task 3: Wire dependency, config, and factory

- [ ] Add optional extra to `pyproject.toml`:

```toml
needle = [
  "cactus-needle>=3.0,<4",
]
```

- [ ] Extend `BackendName` and config validation to include `needle`; add `needle_model: str = "Cactus-Compute/needle3"`. Keep `JEV_BACKEND` precedence unchanged.
- [ ] In `src/bruv/backends/factory.py`, return Needle capabilities without importing `cactus-needle`. Construct `CactusNeedleRuntime` and `NeedleAdapter` only when selected. Keep injected `client_factory` behavior for current HTTP/SDK backends; add a dedicated optional `needle_runtime_factory` parameter instead of overloading client types.
- [ ] Update CLI backend type declarations or validation in `src/bruv/cli.py` so every command accepts `--backend needle`.
- [ ] Confirm `bruv validate ... --backend needle` performs no model import/download.
- [ ] Commit: `feat: wire Needle backend configuration`.

## Task 4: Add setup, doctor, output contract, and docs

- [ ] Update setup choices to `typesafe|simple-jev|needle`. Needle path persists no credential, explains optional install and first-use ~35 MB download, and returns `bruv doctor`.
- [ ] Update doctor:
  - credentials: `not required for needle`
  - endpoint/reachability: local runtime, no network probe
  - dependency check via `importlib.util.find_spec("needle")`
  - supported platform check for Linux x86_64/arm64, macOS arm64/x86_64, and Windows x86_64/arm64
  - writable cache parent check without downloading model
  - capability result included in `run_doctor`
- [ ] Add `needle` to `BACKEND_VALUES`; add contract note that probability fields are backend-dependent and Needle confidence covers selected full response.
- [ ] Update README and docs with exact commands:

```bash
pip install 'bruv[needle]'
bruv noul "Does this request a refund?" --state "Refund me" --backend needle
```

Document first-use download, local execution, telemetry disabled by bruv, confidence-only fields, score selected-index semantics, and unsupported synthetic probabilities.
- [ ] Add changelog entry identifying relaxed output schema and consumer migration: branch on backend or field presence.
- [ ] Commit: `docs: document Needle backend`.

## Task 5: Add adapter and canonical tests after implementation

- [ ] Add `tests/unit/backends/test_needle.py` with fake runtime assertions for:
  - choice, noul, score mappings
  - one runtime call per question
  - deterministic JSON-state serialization
  - descriptions preserve IDs and include human labels
  - missing package, initialization failure, refusal, suppression, wrong enum, wrong type, out-of-range score, missing/invalid confidence
  - prompt/output data absent from errors
- [ ] Add canonical result tests proving existing probability-bearing answers remain valid and Needle confidence-only answers serialize with omitted probability fields.
- [ ] Add `tests/contract/test_needle_contract.py` using shared request fixtures and `assert_backend_contract`.
- [ ] Run:

```bash
.venv/bin/pytest tests/unit/backends/test_needle.py tests/contract/test_needle_contract.py -q
```

Expected: all selected tests pass.
- [ ] Commit: `test: cover Needle backend adapter`.

## Task 6: Update integration and machine contracts

- [ ] Extend config/factory/setup/doctor tests for Needle paths and prove doctor does not construct runtime or download weights.
- [ ] Extend CLI tests with injected Needle backend for human output, JSON output, field extraction, confidence gates, score gates, validation-only, dry-run, and safe errors.
- [ ] Update spec/schema snapshots from runtime-generated canonical data, not hand-authored guesses. Verify output schema expresses optional probabilities and dual noul modes.
- [ ] Verify TypeSafe and Simple Jev snapshots retain their prior serialized payloads.
- [ ] Run:

```bash
.venv/bin/pytest tests/unit tests/integration tests/contract -q
```

Expected: all selected tests pass.
- [ ] Commit: `test: verify Needle CLI contracts`.

## Task 7: Package and real-runtime smoke

- [ ] Recreate project venv if needed and install editable Needle extra:

```bash
python3 -m venv --clear .venv
. .venv/bin/activate
python -m pip install -e '.[dev,needle]'
```

- [ ] Confirm package availability without downloading through doctor and `python -c 'import needle'`.
- [ ] Because server disk is nearly full, inspect free space before model fetch. Do not download when available disk is under 1 GB beyond measured package/model/cache requirement. Report blocker instead of filling root filesystem.
- [ ] If space permits, run one real Needle choice request and capture elapsed time, peak memory reported by runtime, selected answer, and confidence. Never claim model smoke success if download or inference was skipped.
- [ ] Build wheel/sdist and inspect wheel metadata to confirm Needle stays optional.
- [ ] Commit any packaging corrections: `build: package optional Needle support`.

## Task 8: Full verification and release readiness

- [ ] Run full verification serially:

```bash
.venv/bin/ruff format --check .
.venv/bin/ruff check .
.venv/bin/mypy src/bruv
.venv/bin/pytest --cov=bruv
.venv/bin/python -m build
```

Expected: every command exits 0; tests report zero failures; coverage remains at project threshold; wheel and sdist build.

- [ ] Run clean-install test suite and CLI smoke without Needle extra, proving core install still works and missing Needle gives exact install action.
- [ ] Run clean-install smoke with Needle extra but do not force model download if disk guard fails.
- [ ] Check `git status --short`, diff for secrets/model binaries/cache artifacts, and malformed tokens.
- [ ] Record exact verified and skipped checks in `docs/launch-evidence/0.1.0.md` or a new versioned evidence file if version changes.
- [ ] Commit: `chore: record Needle backend verification`.
