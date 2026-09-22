"""OpenRouter concrete-model helper tests."""

from __future__ import annotations

import pytest

from bruv.backends.openrouter import RESERVED_ROUTER_ALIAS, require_concrete_model


def test_alias_constant() -> None:
    assert RESERVED_ROUTER_ALIAS == "openrouter/auto"


def test_valid_concrete_model_returns_value() -> None:
    for source in ("config", "request"):
        assert require_concrete_model("anthropic/claude-3.5-haiku", source) == (
            "anthropic/claude-3.5-haiku"
        )


def test_blank_model_rejected() -> None:
    for source in ("config", "request"):
        with pytest.raises(Exception) as exc_info:
            require_concrete_model("   ", source)
        assert getattr(exc_info.value, "paid_request", True) is False
        assert "must not be blank" in str(exc_info.value)


def test_padded_model_rejected() -> None:
    with pytest.raises(Exception) as exc_info:
        require_concrete_model(" anthropic/claude-3.5-haiku ", "config")
    assert getattr(exc_info.value, "paid_request", True) is False
    assert "surrounding whitespace" in str(exc_info.value)


def test_router_alias_rejected() -> None:
    with pytest.raises(Exception) as exc_info:
        require_concrete_model("openrouter/auto", "request")
    assert getattr(exc_info.value, "paid_request", True) is False
    assert "openrouter/auto" in str(exc_info.value)


def test_error_action_names_source() -> None:
    from bruv.application import ConfigurationError

    with pytest.raises(ConfigurationError) as exc_info:
        require_concrete_model("openrouter/auto", "config")
    assert "config" in exc_info.value.action
