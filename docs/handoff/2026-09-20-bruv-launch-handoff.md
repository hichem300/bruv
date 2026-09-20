# bruv Launch Handoff

**Date:** 2026-09-20  
**Project root:** `/root/business/PROJECTS/jev-test`  
**State:** Design approved. Implementation not started. Directory is not currently a Git repository.

## Mission

Launch **bruv**, Python CLI for human users and AI coding agents.

**Tagline:** no yap, only fax

bruv exposes one typed-decision CLI over:

1. TypeSafe Jev hosted API.
2. Simple Jev local or remote HTTP API.

It preserves useful `jev-cli` argument shapes while using its own `bruv` executable.

## Source of Truth

Read first:

`docs/superpowers/specs/2026-09-20-bruv-cli-design.md`

Do not redesign settled architecture unless implementation evidence exposes a conflict.

## Settled Decisions

- Language: Python.
- Package and executable: `bruv`.
- Display name: bruv.
- Architecture: Ports + Adapters + Facade.
- Primary pattern: Adapter.
- Backends: TypeSafe Jev and Simple Jev.
- Question types: `noul`, `choice`, `score`.
- Primary commands: `noul`, `choice`, `score`, `eval`, `validate`.
- Agent commands: `spec`, `schema`, `skill install`.
- Machine contract: deterministic JSON, stable exit codes, stdout/stderr separation.
- Simple Jev output must always disclose `calibrated: false`.
- MCP, batch/resume, plugins, and model lifecycle management are outside v1.
- Windows, macOS, and Linux are required v1 platforms.
- Primary install path uses verified standalone releases; `pipx` and `uv tool` remain secondary paths.
- Human onboarding includes `bruv setup` and `bruv doctor`.
- Ship runnable funnel-audit agent example for YouTube demonstration.
- Publish as public GitHub repository under Apache-2.0.
- Repository must include public contribution, conduct, support, security, issue, CI, and release infrastructure.

## Public Repository Requirement

Prepare public repo with `LICENSE`, `NOTICE`, `README.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `SUPPORT.md`, `CHANGELOG.md`, issue forms, pull-request template, Dependabot config, cross-platform CI, and protected release workflow.

README must lead viewers from project promise to installation, first result, human workflow, AI-agent workflow, and funnel-audit demo. State clearly that bruv is unofficial and not affiliated with TypeSafe AI, Featherless AI, or upstream projects. Public fixtures must contain no customer data or secrets.

CI must run safely on untrusted public pull requests without provider credentials. Tagged releases build and smoke-test standalone artifacts plus Python distributions, then publish checksums, software bill of materials, and release notes.

## User-Friendly Release Requirement

Viewers must be able to go from video description to first result without knowing Python packaging.

Required experience:

- one copy-paste installer for macOS/Linux;
- one copy-paste PowerShell installer for Windows;
- no administrator rights by default;
- checksum/signature verification before execution;
- clear PATH, upgrade, and uninstall guidance;
- five-minute README quickstart;
- guided `bruv setup` flow with plain-language backend choices;
- `bruv doctor` with one concrete fix per failed check;
- readable help and errors that state whether a paid request occurred;
- interactive progress only on stderr, with non-interactive mode staying deterministic.

## Agent-First Requirement

bruv must be easy for Codex, Claude Code, and Pi to discover and invoke.

Ship canonical Agent Skills-standard file:

`skills/bruv/SKILL.md`

Provide:

```text
bruv spec --output json
bruv schema request
bruv schema output
bruv schema error
bruv validate
bruv skill install --target codex|claude|pi --scope user|project
```

Agent operation must require no terminal prompts or prose parsing. Include `--output json`, `--no-color`, `--quiet`, stdin support, redacted `--dry-run`, stable errors, and fail-closed behavior.

Skill installer must preview destination, refuse silent overwrite, support explicit `--force`, and document manual installation.

## Architecture Target

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

CLI handlers, formatters, gates, and exit-code policy consume canonical domain models only. Provider types stay inside adapters.

## Required Safety

- Never expose API keys in flags, logs, dry runs, errors, fixtures, or tracebacks.
- Keep stdout for result data and stderr for diagnostics.
- Do not describe Simple Jev probabilities as calibrated.
- Treat tagline as brand voice, not factual guarantee.
- Offline validation sends no request and incurs no cost.
- Non-interactive incomplete requests fail instead of prompting.

## Recommended Launch Sequence

1. Turn approved design into implementation plan with small verifiable steps.
2. Initialize Python package and Git repository for planned public release under Apache-2.0.
3. Add public repository foundation: license, notice, README, contribution, conduct, support, security, issue templates, pull-request template, Dependabot, CI, and release workflows.
4. Implement canonical domain models and validation.
5. Implement backend port, facade, and composition factory.
6. Implement Simple Jev adapter against fake HTTP fixtures.
7. Implement TypeSafe adapter against fake SDK/HTTP fixtures.
8. Implement CLI, JSON output, field extraction, gates, and exit codes.
9. Add `spec`, `schema`, and dry-run contracts.
10. Create skill using skill TDD:
   - run baseline agent scenarios without skill;
   - record failures;
   - write minimal `SKILL.md`;
   - rerun scenarios against Codex, Claude Code, and Pi expectations;
   - close discovered gaps.
11. Add skill installer with safe destination and overwrite handling.
12. Add guided setup, doctor checks, and human-focused help/error messages.
13. Package verified standalone releases for Windows, macOS, and Linux; retain `pipx` and `uv tool` paths.
14. Add `examples/funnel-audit-agent/` with README, agent prompt, audit questions, sample funnel, expected JSON, Bash runner, and PowerShell runner.
15. Add `bruv demo funnel-audit` with dry-run-first and explicit `--execute` behavior.
16. Run focused unit, contract, CLI, packaging, clean-install, public-repo, and cross-platform checks.
17. Perform optional live smoke tests only when credentials/endpoints are provided.

## Verification Gates

Before calling v1 ready:

- Both adapters pass same backend contract suite.
- CLI tests verify stdout/stderr and exit codes.
- JSON output validates against emitted schemas.
- `validate` and `--dry-run` make no paid request.
- Secret-redaction tests pass.
- Simple Jev results show `calibrated: false` in human and JSON output.
- Wheel installs in clean virtual environment.
- Installed `bruv --help`, `bruv setup`, `bruv doctor`, `bruv spec`, and `bruv schema output` work.
- Fresh Windows, macOS, and Linux environments can install without preinstalled Python.
- Installer verifies artifacts and gives working PATH, upgrade, and uninstall instructions.
- Funnel-audit example runs through Bash and PowerShell paths without silent spending.
- Packaged skill is present in standalone artifacts, wheel, and source distribution.
- Skill installation is tested for all three target layouts without overwriting existing files.
- Agent scenarios demonstrate correct question selection, JSON use, exit-code handling, and calibration disclosure.
- Public pull-request CI passes without access to repository or provider secrets.
- Apache-2.0 license and required notices are detected correctly.
- README public quickstart works from clean Windows, macOS, and Linux environments.
- Tagged release workflow emits checksums, software bill of materials, and smoke-tested artifacts.

## References

- TypeSafe docs: https://docs.typesafe.ai/
- Existing community CLI: https://github.com/shaharia-lab/jev-cli
- Simple Jev: https://github.com/featherless-ai/simple-jev
- System One Adapter reference: https://github.com/typesafe-ai/system-one-adapter-python
- Pattern catalog: https://refactoring.guru/design-patterns/catalog
- Agent Skills specification: https://agentskills.io/specification

## First Instruction for Next Session

> Read `docs/superpowers/specs/2026-09-20-bruv-cli-design.md` and this handoff. Create implementation plan for bruv v1. Preserve Ports + Adapters + Facade, two-backend scope, Apache-2.0 public GitHub release, beginner-friendly Windows/macOS/Linux installation, guided setup and doctor commands, funnel-audit agent demo, agent-facing JSON/schema/spec contracts, packaged `SKILL.md`, calibration disclosure, and secret safety. Do not implement until plan is reviewed.
