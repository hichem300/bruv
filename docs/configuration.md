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

- `JEV_BACKEND`: `typesafe` or `simple-jev`.
- `TYPESAFE_API_KEY`: TypeSafe API key (never a CLI flag).
- `TYPESAFE_ENDPOINT`: optional TypeSafe base URL.
- `SIMPLE_JEV_BASE_URL`: Simple Jev base URL.
- `BRUV_OUTPUT`: `human` or `json`.

## Credentials

`bruv setup` stores the TypeSafe key in a protected credential file (mode 0600
on POSIX, user-only ACL on Windows). If secure permissions cannot be applied,
setup refuses persistence and recommends the environment variable.
