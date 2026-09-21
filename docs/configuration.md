# Configuration

bruv stores non-secret configuration in a TOML file and resolves backends with
this precedence: `--backend` flag > `JEV_BACKEND` env > config file > `typesafe`.

## Config file

Default location: the `bruv` config directory reported by your platform
(`~/.config/bruv/bruv.toml` on Linux).

```toml
backend = "simple-jev"
request_timeout_seconds = 30.0
output = "human"
```

No credential is ever stored in the config file.

## Environment variables

- `JEV_BACKEND`: `typesafe`, `simple-jev`, `needle`, or `rlcd-modernbert`.
- `TYPESAFE_API_KEY`: TypeSafe API key (never a CLI flag).
- `TYPESAFE_ENDPOINT`: optional TypeSafe base URL.
- `SIMPLE_JEV_BASE_URL`: Simple Jev base URL.
- `BRUV_OUTPUT`: `human` or `json`.

## Credentials

`bruv setup` stores the TypeSafe key in a protected credential file (mode 0600
on POSIX, user-only ACL on Windows). If secure permissions cannot be applied,
setup refuses persistence and recommends the environment variable.

## Needle backend

Set `backend = "needle"` in the config file or pass `--backend needle`. No
credential or endpoint is needed: inference runs entirely on your machine.

Needle requires the optional dependency (`pip install 'bruv[needle]'`). The
first run downloads about 35 MB of model weights into your local cache; after
that everything is offline and telemetry stays disabled
(`NEEDLE_TELEMETRY=0`).

Needle reports a single confidence value per answer. It does not emit
probability distributions, and bruv never synthesizes them. For `score`
questions Needle selects one level index from the supplied legend; the answer
carries that index and the legend itself.

## RLCD ModernBERT backend

Set `backend = "rlcd-modernbert"` in the config file or pass
`--backend rlcd-modernbert`. No credential or endpoint is needed: inference
runs entirely on your machine through ONNX Runtime on CPU.

RLCD requires the optional dependency (`pip install 'bruv[rlcd-modernbert]'`).
The first run downloads a ~606 MB pinned model artifact plus tokenizer and
calibrator files from Hugging Face; after that runs use the local cache, and
`HF_HUB_OFFLINE=1` works once artifacts are cached.

Two config keys are informational and pinned fail-closed:

```toml
rlcd_model = "heman10x/rlcd-modernbert-151m"
rlcd_revision = "8af2496eb63c7fa66d7d234e1f62629380030eb4"
```

There is no environment variable or CLI flag to change the model or revision.
Any request-level `--model` value must match the pinned model ID exactly; any
divergent value is rejected before inference.
