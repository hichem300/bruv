# bruv CLI Design

**Date:** 2026-09-20
**Status:** Approved

## Brand

- Product, distribution, and executable: `bruv`
- Display name: **bruv**
- Tagline: **no yap, only fax**

The tagline is brand voice, not an accuracy guarantee. Results remain model decisions. Output and documentation must preserve calibration disclosures.

## Goal

Build a Python CLI distributed and invoked as `bruv`. It provides one Jev-compatible command interface over two decision backends:

1. TypeSafe Jev hosted API.
2. Simple Jev local or remote HTTP API.

Preserve existing community `jev-cli` command syntax, stdout/stderr behavior, and core exit-code contracts where practical. Provider-specific differences must remain visible, especially probability calibration.

## Scope

### Version 1 includes

- `noul`, `choice`, `score`, `eval`, and `validate` commands.
- TypeSafe Jev and Simple Jev backends.
- Backend selection through flag, environment, or configuration.
- Canonical typed requests and results.
- Human-readable terminal output and machine-readable JSON.
- Stable field extraction and exit-code behavior.
- Offline validation.
- Sync HTTP execution.
- Packaged Agent Skills-standard `SKILL.md` for Codex, Claude Code, and Pi.
- Self-describing JSON schemas and command specification for agent discovery.
- Explicit non-interactive behavior for AI agents and automation.
- Beginner-friendly installation on Windows, macOS, and Linux.
- Guided human setup, diagnostics, and copy-paste examples.
- Bundled funnel-audit agent example suitable for a YouTube walkthrough.
- Public GitHub repository with Apache-2.0 licensing, contributor guidance, security policy, CI, and release automation.

### Version 1 excludes

- Dynamic third-party plugin loading.
- MCP server.
- Batch/resume processing.
- Local model lifecycle management.
- System One Adapter backend.
- Caching, telemetry, and generalized middleware.
- Full byte-for-byte compatibility with every existing `jev-cli` output.

These can be added after core compatibility and provider behavior are proven.

## Architecture

Use ports-and-adapters architecture. Refactoring.Guru's Adapter pattern is primary. Facade provides one deep application interface. Backend injection supplies Strategy-like runtime variation without a separate strategy hierarchy.

```text
CLI commands
    |
    v
DecisionFacade
    |
    v
DecisionBackend port
    |----------------------|
    v                      v
TypeSafeJevAdapter    SimpleJevAdapter
    |                      |
    v                      v
TypeSafe API          Simple Jev HTTP API
```

Output formatting and exit-code policy consume canonical results, never provider response objects.

## Module Interfaces

### Domain models

`DecisionRequest` contains:

- `state`: string or JSON-compatible value.
- `model`: optional provider model identifier.
- `questions`: mapping from stable question IDs to typed questions.

Question variants:

- `NoulQuestion`: instructions plus optional true/false criteria.
- `ChoiceQuestion`: instructions plus two or more named options.
- `ScoreQuestion`: instructions plus ordered rubric levels.

`DecisionResult` contains:

- canonical answers keyed by question ID;
- backend identifier;
- resolved model identifier;
- `calibrated`: boolean;
- token or request usage when supplied;
- latency when supplied;
- provider request ID when supplied;
- provider metadata in a namespaced optional field.

Canonical domain models must not import provider SDK types.

### Backend port

```python
class DecisionBackend(Protocol):
    def evaluate(self, request: DecisionRequest) -> DecisionResult: ...
```

One method keeps interface small. Configuration, authentication, HTTP lifecycle, request translation, and response translation remain adapter implementation details.

Adapters also expose immutable backend capabilities during construction or through a small metadata value used by validation. Capabilities include supported question types and calibration status. Do not add methods for each question type.

### Decision facade

```python
class DecisionFacade:
    def evaluate(self, request: DecisionRequest) -> DecisionResult: ...

    def validate(self, request: DecisionRequest) -> ValidationResult: ...
```

Facade responsibilities:

- run provider-independent validation;
- run selected-backend capability validation;
- invoke backend only for evaluation;
- return canonical results;
- translate expected backend failures into stable application errors.

Facade does not format terminal output or terminate process.

### Composition factory

A small factory selects concrete adapter at process startup:

```python
def create_backend(config: AppConfig) -> DecisionBackend: ...
```

This is composition-root wiring, not an Abstract Factory hierarchy.

## Backend Adapters

### TypeSafeJevAdapter

- Uses TypeSafe SDK or documented HTTPS interface.
- Reads credentials from environment or protected config, never command flags.
- Maps canonical questions to TypeSafe question types.
- Marks results `calibrated=True` only when TypeSafe contract warrants it.
- Preserves model, usage, latency, and request ID when available.

### SimpleJevAdapter

- Calls `/v1/classifier`; `/v1/systemone` compatibility may be configurable later.
- Supports local and remote base URLs.
- Maps Simple Jev answers into canonical answers.
- Always marks v1 results `calibrated=False` because Simple Jev explicitly states its distributions are not calibrated probabilities of correctness.
- Does not manage model downloads or server startup.

## CLI Contract

Distribution name: `bruv`.

Console executable: `bruv`.

Primary commands:

```text
bruv noul
bruv choice
bruv score
bruv eval
bruv validate
bruv backend list
bruv backend check
bruv setup
bruv doctor
bruv demo funnel-audit
bruv spec
bruv schema request|output|error
bruv skill install
```

Compatibility priorities:

1. Preserve familiar positional question text and flags such as `--state`, `--state-file`, `--option`, `--level`, `--field`, `--fail-under`, and `--quiet` where semantics match.
2. Preserve stdout as result data and stderr as diagnostics.
3. Preserve core exit codes:
   - `0`: success or gate passed.
   - `2`: usage or validation error; no request billed.
   - `3`: missing or rejected authentication.
   - `10`: evaluation succeeded but gate failed.
   - `11`: evaluation succeeded but result fell inside abstain band.
4. Introduce documented codes only for transport or backend availability errors.
5. Default interactive terminal output to readable table/text.
6. Default piped output to JSON only if behavior can be made deterministic and tested; otherwise require explicit `--output json` in initial release.

Backend selection precedence:

1. `--backend` command flag.
2. `JEV_BACKEND` environment variable.
3. Config file.
4. Default `typesafe`.

Backend values: `typesafe`, `simple-jev`.

## Human Installation and Onboarding

bruv targets non-developers following a YouTube tutorial as well as developers.

### Installation paths

Primary installation uses checksum-verified standalone releases with no prior Python setup:

- macOS/Linux: one copy-paste shell installer.
- Windows: one copy-paste PowerShell installer.
- Install into a user-owned binary directory; do not require administrator rights by default.
- Installer detects architecture, verifies checksum/signature, explains PATH changes, and prints the exact next command.

Secondary installation supports Python users:

```text
pipx install bruv
uv tool install bruv
```

Release automation builds and smoke-tests Windows, macOS, and Linux artifacts. Documentation also provides uninstall and upgrade commands. Installation scripts must fail safely and never pipe an unverified download directly into execution.

### First-run path

`bruv setup` runs an optional guided wizard only in an interactive terminal:

1. explain TypeSafe versus Simple Jev in plain language;
2. select backend;
3. collect endpoint/model settings;
4. accept secrets without echoing them;
5. store credentials with restrictive permissions or supported OS credential storage;
6. test connectivity;
7. run one small example;
8. print next commands.

`bruv doctor` checks installation, configuration, credentials, endpoint reachability, backend capabilities, and version. Every failed check includes one concrete fix command. Neither command prints secrets.

### User-facing behavior

- Help text leads with common jobs, not architecture terms.
- Every command includes one copy-paste example.
- Errors state what failed, whether a paid request occurred, and exact next action.
- Human output uses readable labels and optional color; meaning never depends on color.
- Progress indicators appear only on interactive stderr.
- `--help`, `--version`, `validate`, `doctor`, and local schema/spec commands work without credentials.
- README starts with a five-minute quickstart and links deeper reference material afterward.
- Release documentation includes a short YouTube-ready install-to-first-result path for all three operating systems.

## Human and AI-Agent Interface

bruv must work without prompts, terminal emulation, or prose parsing.

Agent-facing contract:

- `--output json` always emits one documented JSON value to stdout.
- `--no-color` disables ANSI output.
- `--quiet` suppresses non-result output.
- stdin supports bounded state input.
- `bruv validate` performs offline validation.
- `--dry-run` shows redacted canonical request and selected backend without spending money.
- `bruv spec --output json` describes commands, flags, defaults, and exit codes.
- `bruv schema ...` emits versioned JSON Schemas.
- errors use stable codes and machine-readable JSON when JSON output is selected.
- no confirmation prompt appears in non-interactive mode; unsafe or incomplete requests fail closed.

### Packaged skill

Repository and distribution include `skills/bruv/SKILL.md`, conforming to Agent Skills standard. Same canonical skill supports Codex, Claude Code, and Pi.

Skill content stays concise and teaches agents:

- when bruv fits better than generative LLM output;
- installation and backend configuration;
- choosing `noul`, `choice`, or `score`;
- validating and dry-running before paid evaluation;
- requesting JSON and interpreting stable exit codes;
- TypeSafe versus Simple Jev calibration difference;
- protecting secrets and avoiding unsupported truth claims.

`bruv skill install --target codex|claude|pi --scope user|project` installs packaged skill into target's supported skill location. Installer previews destination, refuses silent overwrite, and supports an explicit `--force` replacement. Package also documents manual installation.

Skill must be tested with baseline agent scenarios before authoring, then rerun against Codex, Claude Code, and Pi instructions. This follows RED-GREEN-REFACTOR for process documentation.

## Funnel-Audit Agent Example

Ship one complete, runnable example under `examples/funnel-audit-agent/`. It demonstrates bruv as typed decision layer inside a broader audit agent, not as source of guaranteed facts.

Example contents:

```text
examples/funnel-audit-agent/
├── README.md
├── AGENT_PROMPT.md
├── audit.yaml
├── sample-funnel.md
├── expected-output.json
├── run.sh
└── run.ps1
```

Audit questions cover:

- whether offer is understandable;
- whether primary CTA is visible and specific;
- whether proof supports main claim;
- whether obvious conversion friction exists;
- highest-priority leak as a `choice`;
- leak severity as a `score`.

`bruv demo funnel-audit` copies or runs bundled sample with explicit backend and dry-run options. It must never silently spend money. First execution shows canonical request and requires explicit evaluation flag in an interactive terminal; non-interactive use requires `--execute`.

Walkthrough must show both audiences:

1. Human runs sample commands and reads terminal output.
2. Codex, Claude Code, or Pi loads `SKILL.md`, runs validation/dry-run, executes bruv with JSON output, and uses structured answers to decide next audit step.

Example output labels model judgments, backend, model, and calibration status. Documentation warns users to verify business conclusions against source evidence.

## Configuration

Configuration contains non-secret defaults:

- selected backend;
- TypeSafe endpoint and model;
- Simple Jev base URL and model;
- request timeout;
- output mode.

Secrets remain in environment variables or protected provider credential storage. Dry-run, validation, errors, and debug output must redact secrets.

## Data Flow

1. CLI framework parses arguments.
2. Command builds canonical `DecisionRequest`.
3. Composition root loads config and constructs backend adapter.
4. `DecisionFacade.validate()` checks domain rules and backend capabilities.
5. For evaluation, facade invokes `DecisionBackend.evaluate()`.
6. Adapter sends provider request and translates response.
7. Output formatter renders canonical result.
8. Exit-code policy computes process status from canonical result and gate options.

No formatter, gate, or CLI command branches on provider response shape.

## Error Model

Stable application errors:

- `UsageError`: malformed CLI arguments.
- `ValidationError`: invalid canonical request or unsupported capability.
- `AuthenticationError`: credentials missing or rejected.
- `BackendUnavailableError`: timeout, refused connection, or temporary provider failure.
- `ProviderResponseError`: malformed or incompatible provider response.
- `ConfigurationError`: invalid configuration.

Adapters catch only expected provider, HTTP, and schema failures. Unexpected programming errors retain tracebacks in debug mode and produce a concise internal-error message otherwise.

Errors go to stderr. Machine-readable mode emits stable error JSON without secrets or raw credential-bearing headers.

## Output and Calibration Safety

Every machine-readable result includes:

```json
{
  "backend": "simple-jev",
  "model": "Qwen/Qwen3.5-0.8B",
  "calibrated": false,
  "answers": {}
}
```

Human-readable Simple Jev output displays an uncalibrated marker. Documentation must say thresholds are backend-specific and should be validated on representative task data. Switching backend must never silently imply equivalent probability meaning.

## Testing Strategy

### Domain tests

- question and request validation;
- canonical result invariants;
- gate and abstain-band behavior;
- exit-code mapping.

### Shared backend contract tests

Run same behavioral contract against each adapter using fake HTTP responses:

- all three question types map correctly;
- provider metadata does not leak into domain fields;
- calibration flag is correct;
- expected failures become stable application errors;
- secrets never appear in errors.

### Adapter tests

- exact outbound request fixtures;
- exact inbound response translation;
- authentication and timeout handling;
- malformed response handling.

### CLI tests

- command parsing compatibility;
- stdout/stderr separation;
- JSON fixtures;
- field extraction;
- exit codes;
- backend precedence.

### Live integration tests

Optional and environment-gated. Normal test suite must not require credentials, network access, model downloads, or GPU hardware.

## Pattern Decisions

- **Adapter:** required because TypeSafe and Simple Jev expose differing provider contracts.
- **Facade:** required to hide validation, backend invocation, and stable error translation behind small interface.
- **Strategy:** achieved through injected `DecisionBackend`; no duplicate strategy classes.
- **Factory Method:** limited to backend construction at composition root.
- **Command:** delegated to Python CLI framework; no custom command-object hierarchy in v1.
- **Decorator:** deferred until retries, rate limiting, caching, or telemetry become demonstrated shared concerns.
- **Plugin architecture:** rejected for v1 because only two known adapters exist.
- **Singleton, Bridge, Mediator, Visitor, Abstract Factory:** rejected because they solve no current requirement.

## Public GitHub Repository

Repository is public and licensed under Apache License 2.0.

Required public-facing files:

```text
README.md
LICENSE
NOTICE
CONTRIBUTING.md
CODE_OF_CONDUCT.md
SECURITY.md
CHANGELOG.md
SUPPORT.md
.github/
├── ISSUE_TEMPLATE/
│   ├── bug.yml
│   ├── feature.yml
│   └── config.yml
├── PULL_REQUEST_TEMPLATE.md
├── dependabot.yml
└── workflows/
    ├── ci.yml
    └── release.yml
```

README must explain in this order:

1. what bruv does in one sentence;
2. honest Jev-compatible and unofficial-project disclosure;
3. 60-second install and first result;
4. human use;
5. AI-agent use with Codex, Claude Code, and Pi;
6. funnel-audit demo;
7. backend and calibration differences;
8. security, privacy, contribution, and deeper documentation links.

Repository metadata includes concise description, relevant topics, screenshots or terminal recording, supported platforms, package/release links, and Apache-2.0 license detection. Do not imply affiliation with TypeSafe AI, Featherless AI, or upstream projects. Preserve third-party notices and licenses where required.

CI runs on pull requests and main branch across Windows, macOS, and Linux. It verifies formatting, linting, typing, unit tests, adapter contracts, CLI behavior, schemas, skill packaging, example integrity, package builds, and clean installs. Live provider tests remain optional and secret-gated.

Release workflow:

- starts from a version tag;
- builds wheel, source distribution, and standalone platform artifacts;
- generates checksums and software bill of materials;
- smoke-tests artifacts in clean environments;
- publishes GitHub Release notes from changelog;
- publishes Python package only through protected trusted publishing when configured;
- never exposes provider credentials to pull-request workflows.

`SECURITY.md` provides private vulnerability reporting and supported-version policy. `CONTRIBUTING.md` gives copy-paste environment, test, documentation, and pull-request steps. Public examples and fixtures contain no real customer data, tokens, endpoints, or proprietary funnel material.

## Suggested Layout

```text
README.md
LICENSE
NOTICE
CONTRIBUTING.md
CODE_OF_CONDUCT.md
SECURITY.md
CHANGELOG.md
SUPPORT.md
.github/
src/bruv/
├── cli.py
├── application.py
├── config.py
├── exit_codes.py
├── domain/
│   ├── requests.py
│   ├── results.py
│   └── errors.py
├── backends/
│   ├── interface.py
│   ├── factory.py
│   ├── typesafe.py
│   └── simple_jev.py
└── output/
    ├── json_output.py
    ├── terminal.py
    └── fields.py
skills/
└── bruv/
    └── SKILL.md
examples/
└── funnel-audit-agent/
    ├── README.md
    ├── AGENT_PROMPT.md
    ├── audit.yaml
    ├── sample-funnel.md
    ├── expected-output.json
    ├── run.sh
    └── run.ps1
```

Tests mirror source modules under `tests/`. Agent-skill behavior has baseline and post-skill scenario fixtures.

## Acceptance Criteria

- Same `bruv` command evaluates `noul`, `choice`, and `score` questions through either backend.
- Backend changes require no changes to command handlers, formatters, or gate logic.
- TypeSafe and Simple Jev responses become one documented canonical JSON shape.
- Simple Jev results are visibly marked uncalibrated.
- Offline validation performs no HTTP request.
- Existing compatible `jev-cli` argument shapes behave equivalently after replacing executable name with `bruv` for supported v1 flags.
- Codex, Claude Code, and Pi can discover and use packaged `bruv` skill.
- `spec`, `schema`, JSON output, dry-run, and non-interactive behavior let agents operate without prose parsing.
- Fresh users can install a verified standalone release on Windows, macOS, or Linux without preparing Python.
- `setup` and `doctor` guide users to a working first evaluation without exposing secrets.
- Funnel-audit demo runs on Windows, macOS, and Linux and demonstrates both human and agent workflows without silent spending.
- Public repository carries Apache-2.0 license, clear unofficial-project disclosure, contribution/security policies, issue templates, and cross-platform CI.
- Tagged releases produce verified, smoke-tested artifacts without exposing secrets to public pull requests.
- Automated tests require no live credentials or model downloads.
- Secrets never appear in stdout, stderr, exceptions, dry runs, or fixtures.

## Sources

- Refactoring.Guru pattern catalog: https://refactoring.guru/design-patterns/catalog
- Adapter pattern: https://refactoring.guru/design-patterns/adapter
- Strategy pattern: https://refactoring.guru/design-patterns/strategy
- Command pattern: https://refactoring.guru/design-patterns/command
- Community Jev CLI: https://github.com/shaharia-lab/jev-cli
- Simple Jev: https://github.com/featherless-ai/simple-jev
- System One Adapter reference: https://github.com/typesafe-ai/system-one-adapter-python
