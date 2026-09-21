"""Internal registry for backend metadata and adapter construction."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, cast

import httpx

from bruv.application import ConfigurationError
from bruv.backends.interface import DecisionBackend
from bruv.config import AppConfig
from bruv.domain.validation import BackendCapabilities
from bruv.onboarding.credentials import Credentials

SelectedDependencyFactory = Callable[[AppConfig, Credentials], object]
BackendBuilder = Callable[["BackendBuildContext"], DecisionBackend]


@dataclass(frozen=True, slots=True)
class BackendBuildContext:
    """Inputs shared by every registered backend builder."""

    config: AppConfig
    credentials: Credentials
    selected_dependency_factory: SelectedDependencyFactory | None = None

    def selected_dependency(self) -> object | None:
        if self.selected_dependency_factory is None:
            return None
        return self.selected_dependency_factory(self.config, self.credentials)


@dataclass(frozen=True, slots=True)
class BackendDefinition:
    """Static backend metadata plus its lazy adapter builder."""

    name: str
    capabilities: BackendCapabilities
    build: BackendBuilder
    setup_description: str
    install_hint: str | None = None
    needs_credentials: bool = False
    runs_local: bool = False
    optional_module: str | None = None


class _BackendRegistry:
    def __init__(self, definitions: Iterable[BackendDefinition]) -> None:
        indexed: dict[str, BackendDefinition] = {}
        for definition in definitions:
            if definition.name != definition.capabilities.backend:
                raise RuntimeError(
                    "Backend registry name must match capabilities backend: "
                    f"{definition.name!r} != {definition.capabilities.backend!r}"
                )
            if definition.name in indexed:
                raise RuntimeError(f"Duplicate backend registry name: {definition.name!r}")
            indexed[definition.name] = definition
        self._definitions = indexed
        self.names = tuple(indexed)

    def get(self, name: str) -> BackendDefinition:
        try:
            return self._definitions[name]
        except KeyError:
            supported = ", ".join(repr(item) for item in self.names)
            raise ConfigurationError(
                message=f"Unknown backend {name!r}.",
                paid_request=False,
                action=f"Choose one of: {supported}.",
            ) from None


def _build_typesafe(context: BackendBuildContext) -> DecisionBackend:
    from bruv.backends.typesafe import TypeSafeClientPort, TypeSafeJevAdapter

    dependency = context.selected_dependency()
    client = (
        cast(TypeSafeClientPort, dependency)
        if dependency is not None
        else cast(TypeSafeClientPort, _default_typesafe_client(context.config, context.credentials))
    )
    return TypeSafeJevAdapter(client, model=context.config.typesafe_model)


def _build_simple_jev(context: BackendBuildContext) -> DecisionBackend:
    from bruv.backends.simple_jev import SimpleJevAdapter

    dependency = context.selected_dependency()
    client = (
        cast(httpx.Client, dependency)
        if dependency is not None
        else _default_simple_jev_client(context.config)
    )
    return SimpleJevAdapter(
        base_url=str(context.config.simple_jev_base_url),
        model=context.config.simple_jev_model,
        client=client,
        timeout_seconds=context.config.request_timeout_seconds,
    )


def _build_needle(context: BackendBuildContext) -> DecisionBackend:
    from bruv.backends.needle import CactusNeedleRuntime, NeedleAdapter, NeedleRuntime

    dependency = context.selected_dependency()
    runtime = cast(NeedleRuntime, dependency) if dependency is not None else CactusNeedleRuntime()
    return NeedleAdapter(runtime, model=context.config.needle_model)


def _default_simple_jev_client(config: AppConfig) -> httpx.Client:
    return httpx.Client(
        follow_redirects=False,
        verify=True,
        timeout=config.request_timeout_seconds,
    )


def _default_typesafe_client(config: AppConfig, credentials: Credentials) -> object:
    try:
        from typesafe_sdk import TypeSafeClient  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ConfigurationError(
            message="TypeSafe SDK is not installed.",
            paid_request=False,
            action="Install typesafe-sdk or switch to another backend.",
        ) from exc

    if not credentials.has_typesafe:
        raise ConfigurationError(
            message="No TypeSafe API key found.",
            paid_request=False,
            action="Set TYPESAFE_API_KEY or run `bruv setup`.",
        )

    kwargs: dict[str, Any] = {"api_key": credentials.typesafe_api_key}
    if config.typesafe_endpoint is not None:
        kwargs["base_url"] = str(config.typesafe_endpoint)
    if config.typesafe_model is not None:
        kwargs["model"] = config.typesafe_model
    return TypeSafeClient(**kwargs)


_REGISTRY = _BackendRegistry(
    (
        BackendDefinition(
            name="typesafe",
            capabilities=BackendCapabilities(
                backend="typesafe",
                question_types=frozenset({"noul", "choice", "score"}),
                calibrated=True,
                allows_json_state=True,
            ),
            build=_build_typesafe,
            setup_description="hosted, calibrated. Needs TYPESAFE_API_KEY.",
            install_hint="Install typesafe-sdk.",
            needs_credentials=True,
        ),
        BackendDefinition(
            name="simple-jev",
            capabilities=BackendCapabilities(
                backend="simple-jev",
                question_types=frozenset({"noul", "choice", "score"}),
                calibrated=False,
                allows_json_state=True,
            ),
            build=_build_simple_jev,
            setup_description="local/self-hosted, uncalibrated. No key needed.",
        ),
        BackendDefinition(
            name="needle",
            capabilities=BackendCapabilities(
                backend="needle",
                question_types=frozenset({"noul", "choice", "score"}),
                calibrated=True,
                allows_json_state=True,
            ),
            build=_build_needle,
            setup_description="local Needle 3 inference. No key needed.",
            install_hint="Install Needle support with: pip install 'bruv[needle]'",
            runs_local=True,
            optional_module="needle",
        ),
    )
)

backend_names = _REGISTRY.names


def get_backend_definition(name: str) -> BackendDefinition:
    return _REGISTRY.get(name)


def backend_capabilities(name: str) -> BackendCapabilities:
    return get_backend_definition(name).capabilities


def create_registered_backend(context: BackendBuildContext) -> DecisionBackend:
    return get_backend_definition(context.config.backend).build(context)


__all__ = [
    "BackendBuildContext",
    "BackendDefinition",
    "backend_capabilities",
    "backend_names",
    "create_registered_backend",
    "get_backend_definition",
]
