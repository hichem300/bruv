"""Non-secret configuration and backend precedence."""

from __future__ import annotations

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
    if "SIMPLE_JEV_BASE_URL" in environment:
        data["simple_jev_base_url"] = environment["SIMPLE_JEV_BASE_URL"]
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


__all__ = ["AppConfig", "BackendName", "config_dir", "config_path", "load_config"]
