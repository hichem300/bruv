"""Composition root: build a backend from config and credentials.

Command handlers never import provider adapter classes directly. This factory
owns HTTPX and SDK lifecycle setup. For TypeSafe, the real SDK is imported
lazily; tests inject a ``client_factory`` to avoid the SDK dependency.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx

from bruv.application import ConfigurationError
from bruv.backends.interface import DecisionBackend
from bruv.backends.simple_jev import SimpleJevAdapter
from bruv.backends.typesafe import TypeSafeJevAdapter
from bruv.config import AppConfig
from bruv.domain.validation import BackendCapabilities
from bruv.onboarding.credentials import Credentials

ClientFactory = Callable[[AppConfig, Credentials], Any]


def _default_simple_jev_client(config: AppConfig) -> httpx.Client:
    return httpx.Client(
        follow_redirects=False,
        verify=True,
        timeout=config.request_timeout_seconds,
    )


def backend_capabilities(config: AppConfig) -> BackendCapabilities:
    """Return capabilities for the configured backend without any client."""
    if config.backend == "simple-jev":
        return SimpleJevAdapter.capabilities
    return TypeSafeJevAdapter.capabilities


def create_backend(
    config: AppConfig,
    credentials: Credentials,
    *,
    client_factory: ClientFactory | None = None,
) -> DecisionBackend:
    """Select and construct the configured backend exactly once."""
    if config.backend == "simple-jev":
        client: Any = (
            client_factory(config, credentials)
            if client_factory is not None
            else _default_simple_jev_client(config)
        )
        return SimpleJevAdapter(
            base_url=str(config.simple_jev_base_url),
            model=config.simple_jev_model,
            client=client,
            timeout_seconds=config.request_timeout_seconds,
        )

    if config.backend == "typesafe":
        client = (
            client_factory(config, credentials)
            if client_factory is not None
            else _default_typesafe_client(config, credentials)
        )
        return TypeSafeJevAdapter(client, model=config.typesafe_model)

    raise ConfigurationError(
        message=f"Unknown backend {config.backend!r}.",
        paid_request=False,
        action="Choose 'typesafe' or 'simple-jev'.",
    )


def _default_typesafe_client(config: AppConfig, credentials: Credentials) -> object:
    """Build a real TypeSafe SDK client, importing the SDK lazily.

    The SDK is an optional runtime dependency; if it is missing, guide the user
    to install it or switch to Simple Jev.
    """
    try:
        from typesafe_sdk import TypeSafeClient  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ConfigurationError(
            message="TypeSafe SDK is not installed.",
            paid_request=False,
            action="Install typesafe-sdk or switch to the simple-jev backend.",
        ) from exc

    if not credentials.has_typesafe:
        raise ConfigurationError(
            message="No TypeSafe API key found.",
            paid_request=False,
            action="Set TYPESAFE_API_KEY or run `bruv setup`.",
        )

    kwargs: dict[str, object] = {"api_key": credentials.typesafe_api_key}
    if config.typesafe_endpoint is not None:
        kwargs["base_url"] = str(config.typesafe_endpoint)
    if config.typesafe_model is not None:
        kwargs["model"] = config.typesafe_model
    return TypeSafeClient(**kwargs)


__all__ = ["backend_capabilities", "create_backend"]
