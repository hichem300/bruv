# Managed Simple Jev Setup Design

## Goal

Make local Simple Jev usable through one beginner-friendly command:

```bash
bruv setup simple-jev
```

After setup, normal `bruv` decision commands automatically start and reuse managed Simple Jev server when backend is `simple-jev`.

## User experience

`bruv setup simple-jev`:

1. Detects Python and NVIDIA CUDA availability.
2. Chooses safe default model `Qwen/Qwen3.5-0.8B` and CPU or CUDA device automatically.
3. Shows selected model, device, managed installation location, and warning that first use downloads model weights.
4. Requests confirmation before installing.
5. Creates isolated managed virtual environment under platform user-data directory.
6. Installs pinned Simple Jev source into that environment.
7. Persists managed runtime configuration.
8. Starts server, waits for `/health`, and reports exact recovery guidance on failure.
9. Runs one small sample decision and prints ready-to-use command.

Existing `bruv setup` remains interactive backend selection. Existing custom Simple Jev endpoint configuration remains supported. `bruv setup simple-jev` directly selects managed local setup.

Normal use stays unchanged:

```bash
bruv noul "Is this a refund request?" \
  --state "Give me my money back" \
  --backend simple-jev
```

When configured endpoint is managed local Simple Jev and health check fails, bruv starts server, waits until healthy, then executes request. Healthy server is reused.

Management commands:

```bash
bruv serve simple-jev status
bruv serve simple-jev start
bruv serve simple-jev stop
bruv setup simple-jev --repair
```

## Architecture

### Managed runtime service

New onboarding/runtime module owns:

- platform-specific data, environment, config, PID, and log paths;
- isolated virtual-environment creation;
- pinned Simple Jev installation;
- CPU/CUDA detection and launch arguments;
- process start, stop, status, stale-PID cleanup, and health waiting;
- a cross-process startup lock so concurrent bruv calls do not launch duplicate servers.

Heavy Simple Jev, PyTorch, and Transformers packages never enter bruv's own environment.

### Configuration

Persist managed settings separately from credentials:

- mode: managed;
- base URL, default `http://127.0.0.1:8000`;
- exact model ID;
- device and dtype;
- context and batch defaults;
- pinned Simple Jev source revision;
- managed environment path.

Manual `SIMPLE_JEV_BASE_URL` configuration remains authoritative and does not trigger managed auto-start.

### CLI integration

- `setup` accepts optional backend argument and `--repair`.
- `serve simple-jev {start|stop|status}` controls only bruv-managed process.
- Simple Jev adapter construction calls lightweight `ensure_running()` only for managed configuration.
- Existing backend registry remains source of backend metadata and selection.

## Process lifecycle

Server runs detached with PID and logs in bruv data directory. Start succeeds only after `/health` responds before timeout. Stop signals recorded process, waits briefly, then reports failure rather than killing unrelated processes. PID identity is checked before signaling. Stale PID state is removed safely.

Auto-start is local-only, idempotent, and silent during healthy reuse. Startup diagnostics go to stderr so JSON stdout remains machine-safe.

## Errors and recovery

Actionable failures cover:

- unsupported Python;
- insufficient disk or failed package/model download;
- missing or unusable CUDA;
- occupied port;
- corrupt managed environment;
- startup timeout;
- stale PID;
- failed health check.

Each error names next command, usually `bruv setup simple-jev --repair`, `bruv serve simple-jev status`, or log path inspection. Setup never deletes existing environment without explicit repair confirmation.

## Scope boundaries

First release supports one managed local Simple Jev instance and one configured model. No Docker, multi-model pool, service-manager installation, remote deployment, GUI, or automatic system startup. Public demo and custom endpoints remain manual configuration choices.

## Verification

Implementation is code-first. One quick final pass verifies:

- focused setup and process-management tests;
- CLI contract for direct backend argument, repair, and serve commands;
- auto-start with mocked process and health boundaries;
- existing suite once at end;
- one manual dry setup/status smoke path without downloading full model weights.
