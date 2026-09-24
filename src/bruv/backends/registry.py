"""Internal registry for backend metadata and adapter construction."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, cast

import httpx

from bruv.application import ConfigurationError
from bruv.backends.interface import DecisionBackend
from bruv.config import AppConfig
from bruv.domain.questions import ChoiceQuestion
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
    runtime_packages: tuple[str, ...] = ()


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
    if context.config.simple_jev_managed:
        from urllib.parse import urlparse

        from bruv.onboarding.simple_jev_runtime import (
            ManagedSimpleJevRuntime,
            ManagedSimpleJevSettings,
        )

        parsed = urlparse(str(context.config.simple_jev_base_url))
        settings = ManagedSimpleJevSettings(
            model=context.config.simple_jev_model,
            host=parsed.hostname or "127.0.0.1",
            port=parsed.port or 8000,
            device=context.config.simple_jev_device,
            dtype=context.config.simple_jev_dtype,
        )
        ManagedSimpleJevRuntime(settings=settings).ensure_running()

    from bruv.backends.simple_jev import SimpleJevAdapter
    from bruv.onboarding.simple_jev_runtime import SimpleJevPaths
    from bruv.onboarding.simple_jev_warning import stderr_sink

    dependency = context.selected_dependency()
    client = (
        cast(httpx.Client, dependency)
        if dependency is not None
        else _default_simple_jev_client(context.config)
    )
    if context.config.simple_jev_managed:
        return SimpleJevAdapter(
            base_url=str(context.config.simple_jev_base_url),
            model=context.config.simple_jev_model,
            client=client,
            timeout_seconds=context.config.request_timeout_seconds,
            warning_sink=stderr_sink(),
            warning_root=SimpleJevPaths.default().root,
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


def _build_rlcd_modernbert(context: BackendBuildContext) -> DecisionBackend:
    from bruv.backends.rlcd_modernbert import build_adapter

    return build_adapter(model=context.config.rlcd_model, revision=context.config.rlcd_revision)


def _default_simple_jev_client(config: AppConfig) -> httpx.Client:
    return httpx.Client(
        follow_redirects=False,
        verify=True,
        timeout=config.request_timeout_seconds,
    )


def _build_openrouter(context: BackendBuildContext) -> DecisionBackend:
    from bruv.backends.openrouter import OpenRouterAdapter, require_concrete_model

    if not context.credentials.has_openrouter:
        raise ConfigurationError(
            message="No OpenRouter API key found.",
            paid_request=False,
            action="Set OPENROUTER_API_KEY or run `bruv setup openrouter`.",
        )
    model = context.config.openrouter_model
    if not model:
        raise ConfigurationError(
            message="No OpenRouter model configured.",
            paid_request=False,
            action="Set openrouter_model in config or run `bruv setup openrouter`.",
        )
    require_concrete_model(model, "config")
    dependency = context.selected_dependency()
    client = (
        cast(httpx.Client, dependency)
        if dependency is not None
        else _default_openrouter_client(context.config)
    )
    return OpenRouterAdapter(
        client=client,
        api_key=context.credentials.openrouter_api_key,
        model=model,
        timeout=context.config.request_timeout_seconds,
    )


def _default_openrouter_client(config: AppConfig) -> httpx.Client:
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
            install_hint="Install Needle support with: pip install 'bruv[needle] @ git+https://github.com/hichem300/bruv.git'",
            runs_local=True,
            optional_module="needle",
        ),
        BackendDefinition(
            name="rlcd-modernbert",
            capabilities=BackendCapabilities(
                backend="rlcd-modernbert",
                question_types=frozenset({"noul", "choice", "score"}),
                calibrated=True,
                allows_json_state=True,
                explicit_abstention=True,
                supported_total_candidates=frozenset({2, 3, 4, 5, 6, 7, 9, 11, 17, 25}),
                reserved_input_markers=("<<LABEL>>", "<<SEP>>"),
                reserved_answer_ids=frozenset({"__abstain__"}),
            ),
            build=_build_rlcd_modernbert,
            setup_description=(
                "local RLCD ModernBERT, calibrated, ~606 MB first-use HF download, "
                "local/free inference."
            ),
            install_hint="Install RLCD support with: pip install 'bruv[rlcd-modernbert] @ git+https://github.com/hichem300/bruv.git'",
            runs_local=True,
            runtime_packages=("onnxruntime", "tokenizers", "numpy", "huggingface_hub"),
        ),
        BackendDefinition(
            name="openrouter",
            capabilities=BackendCapabilities(
                backend="openrouter",
                question_types=frozenset({"noul", "choice", "score"}),
                calibrated=True,
                allows_json_state=True,
            ),
            build=_build_openrouter,
            setup_description=(
                "hosted, paid, calibrated via OpenRouter's typed Decisions API. "
                "Needs OPENROUTER_API_KEY."
            ),
            needs_credentials=True,
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


# --- Browser compatibility, derived from existing capabilities only. ---


def _choice_option_bounds() -> tuple[int, int]:
    """Read ChoiceQuestion criteria bounds from the canonical model itself."""
    minimum, maximum = 2, 50
    for item in ChoiceQuestion.model_fields["criteria"].metadata:
        min_length = getattr(item, "min_length", None)
        max_length = getattr(item, "max_length", None)
        if isinstance(min_length, int):
            minimum = min_length
        if isinstance(max_length, int):
            maximum = max_length
    return minimum, maximum


def browser_supports_noul_choice(capabilities: BackendCapabilities) -> bool:
    """A backend is browser-compatible when it answers canonical noul and choice."""
    return {"noul", "choice"} <= capabilities.question_types


def browser_compatible_backends() -> tuple[str, ...]:
    """Names of registered backends the browser can drive, registry order."""
    return tuple(
        name
        for name in backend_names
        if browser_supports_noul_choice(get_backend_definition(name).capabilities)
    )


def browser_choice_batch_sizes(capabilities: BackendCapabilities) -> tuple[int, ...]:
    """Valid candidate batch sizes for one Choice call on this backend.

    Batch size k produces k candidate options plus one abstain option when the
    backend declares ``explicit_abstention``; the total must stay within the
    canonical ChoiceQuestion bounds. ``explicit_abstention`` shifts the total
    but never makes a one-criterion question valid: criteria counts remain the
    canonical 2..50. When ``supported_total_candidates`` is set (RLCD
    ModernBERT's per_k calibration contract), only criteria counts whose
    total (k + abstain offset) is supported are allowed. Unsupported candidate
    totals are simply absent from the result:
    callers must reject them, never fall back to a global temperature.
    """
    abstain = 1 if capabilities.explicit_abstention else 0
    minimum, maximum = _choice_option_bounds()
    sizes: list[int] = []
    for criteria in range(minimum, maximum + 1):
        total = criteria + abstain
        if (
            capabilities.supported_total_candidates is not None
            and total not in capabilities.supported_total_candidates
        ):
            continue
        sizes.append(criteria)
    return tuple(sizes)


__all__ = [
    "BackendBuildContext",
    "BackendDefinition",
    "backend_capabilities",
    "backend_names",
    "browser_choice_batch_sizes",
    "browser_compatible_backends",
    "browser_supports_noul_choice",
    "create_registered_backend",
    "get_backend_definition",
]
