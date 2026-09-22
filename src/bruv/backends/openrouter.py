"""OpenRouter backend support: concrete-model gate shared by config and requests.

OpenRouter routes unclassified model aliases to whatever model it currently
prefers, which would make bruv decisions non-reproducible and could silently
send data somewhere the operator did not choose. Every OpenRouter model
reference in bruv must therefore name one concrete model; the shared
:func:`require_concrete_model` helper enforces that at both config load and
per-request resolution time. No adapter lives here yet.
"""

from __future__ import annotations

from bruv.application import ConfigurationError

RESERVED_ROUTER_ALIAS = "openrouter/auto"


def require_concrete_model(model: str, source: str) -> str:
    """Return ``model`` when it names one concrete OpenRouter model.

    ``source`` names where the model came from (for example ``"config"`` or
    ``"request"``) so the error can point at the right knob. Raises an unpaid
    :class:`ConfigurationError` for blank values, surrounding whitespace, or
    the reserved router alias.
    """
    stripped = model.strip()
    if not stripped:
        raise ConfigurationError(
            message=f"OpenRouter model from {source} must not be blank.",
            paid_request=False,
            action=f"Set a concrete OpenRouter model in {source}, "
            f"such as 'anthropic/claude-3.5-haiku'.",
        )
    if stripped != model:
        raise ConfigurationError(
            message=f"OpenRouter model from {source} must not have surrounding whitespace.",
            paid_request=False,
            action=f"Remove leading or trailing spaces from the model in {source}.",
        )
    if model == RESERVED_ROUTER_ALIAS:
        raise ConfigurationError(
            message=f"OpenRouter model from {source} is the reserved router "
            f"alias '{RESERVED_ROUTER_ALIAS}'.",
            paid_request=False,
            action=f"Set a concrete OpenRouter model in {source} instead of "
            f"'{RESERVED_ROUTER_ALIAS}'.",
        )
    return model


__all__ = ["RESERVED_ROUTER_ALIAS", "require_concrete_model"]
