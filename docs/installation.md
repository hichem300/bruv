# Installation

bruv runs on Python 3.11+.

## From PyPI

```bash
python -m venv .venv && . .venv/bin/activate
pip install bruv
bruv --help
```

## Standalone installer (no Python required)

macOS/Linux:

```bash
curl -fsSL https://bruv.dev/install.sh | bash
```

Windows (PowerShell):

```powershell
irm https://bruv.dev/install.ps1 | iex
```

Installers verify the SHA-256 checksum before placing the binary. They install
to a user-owned directory and never require sudo/administrator.

## Verify

```bash
bruv --help
bruv doctor
```

## Managed Simple Jev backend (optional)

Simple Jev talks to a local Simple Jev server. bruv can install and manage
that server for you:

```bash
bruv setup simple-jev
```

- Installs an isolated runtime under your platform user-data directory at
  `bruv/simple-jev`. The upstream checkout, virtual environment, model cache,
  state files, and logs all live inside that one managed root; nothing is
  written outside it.
- Device selection is `auto`: bruv probes CUDA through the managed torch and
  falls back to CPU safely when CUDA is unusable.
- On CPU-only hosts, install bootstraps the official CPU torch wheel
  (`--index-url https://download.pytorch.org/whl/cpu`) so the resolver never
  pulls a mismatched CUDA build. All pip steps use `--no-cache-dir`, so bruv
  leaves no persistent pip cache on disk.
- The first install downloads the default `Qwen/Qwen3.5-0.8B` model from
  Hugging Face into the managed cache. This can take several minutes.
- During setup, bruv warns that the default Qwen model in Noul mode can be
  non-discriminating (it may score very similar options alike). The same
  warning is shown once on your first affected use.
- Simple Jev reports `calibrated: false`; its score is not a verified
  real-world probability and must not be presented as probability or
  calibrated confidence.
- After setup, normal simple-jev calls auto-start the managed server and
  reuse it if it is already healthy. Rerunning `bruv setup simple-jev` is safe:
  it verifies the existing install and resumes automatically, skipping steps
  that are already complete.
- Use `bruv setup simple-jev --repair` to force a full rebuild of managed
  state when something is broken or you want a clean rebuild.
- Before downloading packages or the model, setup runs a disk-space preflight
  and fails early with the required and available space if disk is too low.
- Setup streams visible progress to your terminal: git and pip output, plus
  model-loading progress, so you can see each step as it runs.

Control the managed server directly:

```bash
bruv serve simple-jev status
bruv serve simple-jev start
bruv serve simple-jev stop
```

Check everything with `bruv doctor` after setup. git is required for the
managed install.

## OpenRouter backend (optional, paid)

OpenRouter provides paid hosted access to TypeSafe Jev through OpenRouter's
calibrated typed Decisions API at `POST /api/alpha/decisions`. This is a
dedicated typed-decisions endpoint, not Chat Completions. It returns
calibrated probabilities, so bruv reports `calibrated: true`.

```bash
bruv setup openrouter
bruv choice "Which team?" --option sales --option billing --state "help" --backend openrouter
```

- `bruv setup openrouter` prompts for your OpenRouter key, validates it, and
  stores it in the protected credential file (mode 0600 on POSIX, user-only
  ACL on Windows). The key is hidden while typed and is never accepted as a
  CLI flag or written to the config file. If secure permissions cannot be
  applied, setup refuses persistence and recommends `OPENROUTER_API_KEY`.
- Setup makes no paid call. There is no setup smoke test against the billing
  API; your first real decision command is the first paid request.
- The default model is `~typesafe/jev-latest`. Pricing per the current model
  page: $0.042 per 1M input tokens and zero cost for output tokens.

Check everything with `bruv doctor` after setup.

## Needle backend (optional)

Needle runs the Needle 3 model fully on your machine. No credential needed.

```bash
pip install 'bruv[needle]'
```

- First use downloads about 35 MB of model weights to your local cache.
  Later runs use the cached weights; no download happens again.
- All inference is local. Needle telemetry is disabled (`NEEDLE_TELEMETRY=0`),
  and no data leaves your machine.
- Needle reports calibrated confidence only. It does not produce probability
  distributions, and bruv never synthesizes them. Score answers carry the
  selected level index and its legend.
- Supported platforms: Linux, macOS, and Windows on x86_64 and arm64.

Check everything with `bruv doctor` after selecting the Needle backend.

## RLCD ModernBERT backend (optional)

RLCD ModernBERT runs a pinned GLiClass-ModernBERT decision model fully on
your machine through ONNX Runtime on CPU. No credential needed.

```bash
pip install 'bruv[rlcd-modernbert]'
```

- First use downloads a ~606 MB pinned `model.onnx` plus tokenizer and
  calibrator files from Hugging Face at an immutable revision. Every file is
  verified against a pinned size and SHA-256 before use.
- Later runs load from the local Hugging Face cache. With `HF_HUB_OFFLINE=1`,
  bruv works fully offline once artifacts are cached; a missing cache fails
  closed with a remediation message instead of downloading.
- All inference is local. Prompts and state never leave your machine; the
  only network contact is the first-use artifact download from Hugging Face.
- Supported platforms: Linux, macOS, and Windows on x86_64 and arm64.

Check everything with `bruv doctor` after selecting the RLCD ModernBERT
backend.
