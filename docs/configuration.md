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

- `JEV_BACKEND`: `typesafe`, `openrouter`, `simple-jev`, `needle`, or `rlcd-modernbert`.
- `TYPESAFE_API_KEY`: TypeSafe API key (never a CLI flag).
- `TYPESAFE_ENDPOINT`: optional TypeSafe base URL.
- `OPENROUTER_API_KEY`: OpenRouter API key (never a CLI flag).
- `OPENROUTER_MODEL`: OpenRouter model slug. Defaults to `~typesafe/jev-latest`;
  an explicit model slug must be configured when using any other model.
- `SIMPLE_JEV_BASE_URL`: Simple Jev base URL. Setting it configures a custom
  endpoint and disables managed auto-start (see below).
- `BRUV_OUTPUT`: `human` or `json`.

## Credentials

`bruv setup` stores keys in a protected credential file (mode 0600 on POSIX,
user-only ACL on Windows). The file can contain both the TypeSafe key and the
OpenRouter key. If secure permissions cannot be applied, setup refuses
persistence and recommends the environment variable.

## Managed Simple Jev

With `bruv setup simple-jev` (see `docs/installation.md`), the setup writes
`simple_jev_managed = true` into the config file:

```toml
backend = "simple-jev"
simple_jev_managed = true
```

While that flag is set, every normal simple-jev call auto-starts the managed
server under your platform user-data directory at `bruv/simple-jev` and
reuses it when healthy. Device (`auto` with CUDA detection and CPU fallback),
dtype, and model can be tuned in the config file:

```toml
simple_jev_model = "Qwen/Qwen3.5-0.8B"
simple_jev_device = "auto"   # auto (CUDA with CPU fallback), cpu, or cuda
simple_jev_dtype = "bfloat16"
```

The managed server is controlled with `bruv serve simple-jev
status|start|stop`, and `bruv setup simple-jev --repair` rebuilds broken
managed state.

### Custom endpoint

Setting `SIMPLE_JEV_BASE_URL` (or `simple_jev_base_url` in the config file)
to a non-default value disables managed auto-start: bruv talks to that
endpoint and never starts a local server. The default base URL is
`http://127.0.0.1:8000`.

A related key sets the model name sent to the endpoint:

```toml
simple_jev_model = "Qwen/Qwen3.5-0.8B"
```

### Public demo endpoint

No local runtime or key is needed if you point Simple Jev at the public
Featherless demo endpoint:

```toml
simple_jev_base_url = "https://simple-jev-demo-api.featherless.ai/v1/"
simple_jev_model = "featherless-ai/gemma-4-26B-A4B-classifier"
```

The demo endpoint needs no key but is rate limited, so it is suitable for
trying bruv out, not for sustained use. The managed Simple Jev backend stays
uncalibrated in all cases: bruv always reports `calibrated: false`.

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

## OpenRouter backend

Set `backend = "openrouter"` in the config file or pass `--backend openrouter`.
OpenRouter uses calibrated typed Decisions API at `POST /api/alpha/decisions`,
not Chat Completions.

Config keys:

```toml
openrouter_model = "~typesafe/jev-latest"
openrouter_data_collection = false
openrouter_zdr = false
```

- `openrouter_model`: model slug sent to OpenRouter. Defaults to
  `~typesafe/jev-latest`; an explicit slug is required for other models.
- `openrouter_data_collection`: OpenRouter data-collection preference.
- `openrouter_zdr`: OpenRouter zero-data-retention preference.

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
