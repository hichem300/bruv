"""Compatibility wrappers around the backend registry composition root."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from bruv.backends.interface import DecisionBackend
from bruv.backends.registry import BackendBuildContext, create_registered_backend
from bruv.backends.registry import backend_capabilities as registered_backend_capabilities
from bruv.config import AppConfig
from bruv.domain.validation import BackendCapabilities
from bruv.onboarding.credentials import Credentials

ClientFactory = Callable[[AppConfig, Credentials], Any]


def backend_capabilities(config: AppConfig) -> BackendCapabilities:
    """Return configured backend capabilities without constructing dependencies."""
    return registered_backend_capabilities(config.backend)


def create_backend(
    config: AppConfig,
    credentials: Credentials,
    *,
    client_factory: ClientFactory | None = None,
) -> DecisionBackend:
    """Construct configured backend, preserving legacy dependency injection."""
    return create_registered_backend(
        BackendBuildContext(
            config=config,
            credentials=credentials,
            selected_dependency_factory=client_factory,
        )
    )


__all__ = ["backend_capabilities", "create_backend"]
