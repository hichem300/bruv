"""Non-secret configuration and backend precedence."""

from __future__ import annotations

import json
import math
import os
import tempfile
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import platformdirs
from pydantic import AnyHttpUrl, BaseModel, Field, ValidationError, field_validator

from bruv.application import ConfigurationError

BackendName = str


class AppConfig(BaseModel):
    """Non-secret runtime configuration. No credentials live here."""

    model_config = {"extra": "forbid", "frozen": True}

    backend: BackendName = "typesafe"
    typesafe_endpoint: AnyHttpUrl | None = None
    typesafe_model: str | None = "jev-latest"
    simple_jev_base_url: AnyHttpUrl = AnyHttpUrl("http://127.0.0.1:8000")
    simple_jev_model: str = "Qwen/Qwen3.5-0.8B"
    simple_jev_managed: bool = False
    simple_jev_device: Literal["auto", "cpu", "cuda"] = "auto"
    simple_jev_dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16"
    openrouter_model: str | None = "~typesafe/jev-latest"
    openrouter_data_collection: Literal["deny", "allow"] = "deny"
    openrouter_zdr: bool = True
    needle_model: str = "Cactus-Compute/needle3"
    rlcd_model: str = "heman10x/rlcd-modernbert-151m"
    rlcd_revision: str = "8af2496eb63c7fa66d7d234e1f62629380030eb4"
    request_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    output: Literal["human", "json"] = "human"

    @field_validator("backend")
    @classmethod
    def _registered_backend(cls, value: str) -> str:
        from bruv.backends.registry import backend_names

        if value not in backend_names:
            supported = ", ".join(repr(name) for name in backend_names)
            raise ValueError(f"backend must be one of: {supported}")
        return value

    @field_validator("openrouter_model")
    @classmethod
    def _openrouter_model_concrete(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value.strip():
            raise ValueError("openrouter_model must not be blank")
        if value != value.strip():
            raise ValueError("openrouter_model must not have surrounding whitespace")
        if value == "openrouter/auto":
            raise ValueError("openrouter_model must name a concrete model, not 'openrouter/auto'")
        return value

    @field_validator("needle_model")
    @classmethod
    def _nonblank_needle_model(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("needle_model must not be blank")
        return value

    @field_validator("rlcd_model", "rlcd_revision")
    @classmethod
    def _nonblank_rlcd_fields(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("rlcd model fields must not be blank")
        return value


def config_dir() -> Path:
    return Path(platformdirs.user_config_dir("bruv", appauthor=False))


def config_path() -> Path:
    return config_dir() / "bruv.toml"


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigurationError(
            message=f"Could not read config file at {path}.",
            paid_request=False,
            action="Fix or remove the config file and retry.",
        ) from exc
    if "tool" in data and isinstance(data.get("tool"), dict) and "bruv" in data["tool"]:
        tool = data["tool"]
        if isinstance(tool, dict):
            return dict(tool["bruv"]) if isinstance(tool["bruv"], dict) else {}
    return data


def load_config(
    *,
    backend_override: BackendName | None = None,
    env: Mapping[str, str] | None = None,
    path: Path | None = None,
) -> AppConfig:
    """Resolve configuration with backend precedence flag > env > file > default."""
    environment = env if env is not None else __import__("os").environ
    file_data = _read_toml(path if path is not None else config_path())

    env_backend = environment.get("JEV_BACKEND")

    data: dict[str, Any] = {}
    data.update(file_data)
    if env_backend:
        data["backend"] = env_backend
    if backend_override is not None:
        data["backend"] = backend_override

    if "TYPESAFE_ENDPOINT" in environment:
        data["typesafe_endpoint"] = environment["TYPESAFE_ENDPOINT"]
    if "OPENROUTER_MODEL" in environment:
        data["openrouter_model"] = environment["OPENROUTER_MODEL"]
    if "SIMPLE_JEV_MODEL" in environment:
        data["simple_jev_model"] = environment["SIMPLE_JEV_MODEL"]
    if "SIMPLE_JEV_BASE_URL" in environment:
        data["simple_jev_base_url"] = environment["SIMPLE_JEV_BASE_URL"]
        data["simple_jev_managed"] = False
    if "BRUV_OUTPUT" in environment:
        data["output"] = environment["BRUV_OUTPUT"]

    try:
        return AppConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigurationError(
            message="Configuration values are invalid.",
            paid_request=False,
            action="Fix the config file or environment variables and retry.",
        ) from exc


def _toml_value(value: object) -> str:
    """Serialize one model field value as a TOML-compatible literal."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)  # JSON string quoting is valid TOML basic string
    if isinstance(value, Path):
        return json.dumps(str(value))
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"cannot serialize non-finite float: {value!r}")
        return repr(value)
    # AnyHttpUrl and similar pydantic URL types stringify cleanly.
    return json.dumps(str(value))


def update_config(
    values: Mapping[str, object],
    *,
    path: Path | None = None,
) -> AppConfig:
    """Merge values into the existing config file and atomically rewrite it."""
    target = path if path is not None else config_path()

    def _fail(action: str, exc: Exception) -> ConfigurationError:
        raise ConfigurationError(
            message=f"Could not update config file at {target}.",
            paid_request=False,
            action=action,
        ) from exc

    existing = _read_toml(target)
    known_fields = AppConfig.model_fields

    unknown = sorted(set(values) - set(known_fields))
    if unknown:
        raise ConfigurationError(
            message="Unknown configuration keys: " + ", ".join(repr(k) for k in unknown) + ".",
            paid_request=False,
            action="Use only documented AppConfig field names and retry.",
        )

    merged: dict[str, Any] = dict(existing)
    merged.update(dict(values))

    try:
        config = AppConfig.model_validate(merged)
    except ValidationError as exc:
        raise ConfigurationError(
            message="Configuration values are invalid.",
            paid_request=False,
            action="Fix the offending values and retry.",
        ) from exc

    lines: list[str] = []
    for name, field in known_fields.items():
        value = getattr(config, name)
        if value is None:
            continue
        lines.append(f"{name} = {_toml_value(value)}")
    payload = "\n".join(lines) + "\n"

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=target.parent, prefix=".bruv.toml.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(tmp_name, 0o600)
            os.replace(tmp_name, target)
            tmp_name = None  # type: ignore[assignment]
        finally:
            if tmp_name is not None:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
        # fsync the parent directory so the rename is durable, where supported.
        try:
            dir_fd = os.open(target.parent, os.O_RDONLY)  # type: ignore[arg-type]
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except (OSError, AttributeError):
            pass
    except OSError as exc:
        raise _fail(
            "Check directory permissions and free disk space, then retry.", exc
        ) from exc

    return config


__all__ = [
    "AppConfig",
    "BackendName",
    "config_dir",
    "config_path",
    "load_config",
    "update_config",
]
