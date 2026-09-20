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
