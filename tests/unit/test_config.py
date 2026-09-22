"""AppConfig precedence and validation tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from bruv.application import ConfigurationError
from bruv.config import AppConfig, load_config


def write_config(path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_default_backend_is_typesafe() -> None:
    config = load_config(env={}, path=type("P", (), {"is_file": lambda self: False})())
    assert config.backend == "typesafe"
    assert config.output == "human"
    assert config.needle_model == "Cactus-Compute/needle3"
    assert config.rlcd_model == "heman10x/rlcd-modernbert-151m"
    assert config.rlcd_revision == "8af2496eb63c7fa66d7d234e1f62629380030eb4"


def test_rlcd_fields_must_be_nonblank() -> None:
    with pytest.raises(ValidationError, match="rlcd_model"):
        AppConfig(rlcd_model="   ")
    with pytest.raises(ValidationError, match="rlcd_revision"):
        AppConfig(rlcd_revision=" ")


def test_openrouter_privacy_defaults() -> None:
    config = AppConfig()
    assert config.openrouter_model is None
    assert config.openrouter_data_collection == "deny"
    assert config.openrouter_zdr is True


def test_openrouter_model_accepts_concrete_slug() -> None:
    config = AppConfig(openrouter_model="anthropic/claude-3.5-haiku")
    assert config.openrouter_model == "anthropic/claude-3.5-haiku"


def test_openrouter_model_rejects_blank_padded_alias() -> None:
    with pytest.raises(ValidationError, match="openrouter_model must not be blank"):
        AppConfig(openrouter_model="   ")
    with pytest.raises(
        ValidationError, match="openrouter_model must not have surrounding whitespace"
    ):
        AppConfig(openrouter_model=" anthropic/claude-3.5-haiku")
    with pytest.raises(ValidationError, match="concrete model"):
        AppConfig(openrouter_model="openrouter/auto")


def test_file_config_sets_needle_model(tmp_path) -> None:
    cfg = tmp_path / "bruv.toml"
    write_config(cfg, 'needle_model = "custom-needle"\n')
    config = load_config(env={}, path=cfg)
    assert config.needle_model == "custom-needle"


def test_needle_model_must_be_nonblank() -> None:
    with pytest.raises(ValidationError, match="needle_model must not be blank"):
        AppConfig(needle_model="   ")


def test_file_config_sets_backend(tmp_path) -> None:
    cfg = tmp_path / "bruv.toml"
    write_config(cfg, 'backend = "simple-jev"\n')
    config = load_config(env={}, path=cfg)
    assert config.backend == "simple-jev"


def test_env_overrides_file(tmp_path) -> None:
    cfg = tmp_path / "bruv.toml"
    write_config(cfg, 'backend = "simple-jev"\n')
    config = load_config(env={"JEV_BACKEND": "typesafe"}, path=cfg)
    assert config.backend == "typesafe"


def test_flag_overrides_env(tmp_path) -> None:
    config = load_config(
        backend_override="simple-jev", env={"JEV_BACKEND": "typesafe"}, path=tmp_path / "none.toml"
    )
    assert config.backend == "simple-jev"


def test_invalid_env_backend_rejected(tmp_path) -> None:
    with pytest.raises(ConfigurationError):
        load_config(env={"JEV_BACKEND": "bogus"}, path=tmp_path / "none.toml")


def test_timeout_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"request_timeout_seconds": 0})


def test_timeout_capped() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"request_timeout_seconds": 301})


def test_config_has_no_secret_fields() -> None:
    fields = set(AppConfig.model_fields)
    assert "api_key" not in fields
    assert "token" not in fields
    assert "secret" not in fields


def test_invalid_config_raises_configuration_error(tmp_path) -> None:
    cfg = tmp_path / "bruv.toml"
    write_config(cfg, 'request_timeout_seconds = "not-a-number"\n')
    with pytest.raises(ConfigurationError):
        load_config(env={}, path=cfg)
