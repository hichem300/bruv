"""Browser config field and TOML serialization tests."""

from __future__ import annotations

import tomllib

import pytest
from pydantic import ValidationError

from bruv.config import (
    AppConfig,
    default_browser_artifact_dir,
    load_config,
    update_config,
)


def test_browser_defaults() -> None:
    config = AppConfig()
    assert config.browser_allowed_origins == ()
    assert config.browser_max_steps == 25
    assert config.browser_time_limit_seconds == 600.0
    assert config.browser_headless is True
    assert config.browser_observation_max_elements == 60
    assert config.browser_page_text_max_chars == 6_000
    assert config.browser_capture_trace is False
    assert config.browser_capture_video is False
    assert config.browser_artifact_dir is None


def test_origins_must_be_http_s(tmp_path) -> None:
    with pytest.raises(ValidationError, match="http:// or https://"):
        AppConfig(browser_allowed_origins=("ftp://example.com",))
    with pytest.raises(ValidationError, match="whitespace"):
        AppConfig(browser_allowed_origins=(" https://example.com",))


def test_origins_must_be_bare_origins() -> None:
    with pytest.raises(ValidationError, match="no path, query, or fragment"):
        AppConfig(browser_allowed_origins=("https://example.com/app",))
    with pytest.raises(ValidationError, match="no path, query, or fragment"):
        AppConfig(browser_allowed_origins=("https://example.com/?x=1",))
    with pytest.raises(ValidationError, match="no path, query, or fragment"):
        AppConfig(browser_allowed_origins=("https://example.com#frag",))
    with pytest.raises(ValidationError, match="credentials"):
        AppConfig(browser_allowed_origins=("https://user:pass@example.com",))
    with pytest.raises(ValidationError, match="include a host"):
        AppConfig(browser_allowed_origins=("https://",))


def test_origins_are_normalized_and_deduped() -> None:
    with pytest.raises(ValidationError, match="duplicates"):
        AppConfig(browser_allowed_origins=("https://a.com", "https://a.com/"))
    config = AppConfig(browser_allowed_origins=("https://a.com/", "http://b.org"))
    assert config.browser_allowed_origins == ("https://a.com", "http://b.org")


def test_browser_limits_are_bounded() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"browser_max_steps": 0})
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"browser_time_limit_seconds": 0})
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"browser_observation_max_elements": 0})
    assert AppConfig.model_validate({"browser_max_steps": 1}).browser_max_steps == 1


def test_load_origins_from_toml_file(tmp_path) -> None:
    cfg = tmp_path / "bruv.toml"
    cfg.write_text(
        'browser_allowed_origins = ["https://example.com", "https://app.example.com"]\n',
        encoding="utf-8",
    )
    config = load_config(env={}, path=cfg)
    assert config.browser_allowed_origins == ("https://example.com", "https://app.example.com")


def test_update_config_writes_origins_as_toml_array(tmp_path) -> None:
    cfg = tmp_path / "bruv.toml"
    update_config({"browser_allowed_origins": ("https://example.com", "http://b.org")}, path=cfg)
    raw = cfg.read_text(encoding="utf-8")
    assert "browser_allowed_origins = " in raw
    parsed = tomllib.loads(raw)
    assert parsed["browser_allowed_origins"] == ["https://example.com", "http://b.org"]


def test_origins_round_trip_through_toml(tmp_path) -> None:
    cfg = tmp_path / "bruv.toml"
    update_config(
        {
            "browser_allowed_origins": ("https://example.com",),
            "browser_capture_trace": True,
            "browser_artifact_dir": str(tmp_path / "bruv-artifacts"),
        },
        path=cfg,
    )
    config = load_config(env={}, path=cfg)
    assert config.browser_allowed_origins == ("https://example.com",)
    assert config.browser_capture_trace is True
    assert str(config.browser_artifact_dir) == str(tmp_path / "bruv-artifacts")


def test_default_browser_artifact_dir_is_under_user_data() -> None:
    path = default_browser_artifact_dir()
    assert "bruv" in path.parts
    assert path.name == "artifacts"
