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
