"""Backend registry metadata, lookup, and construction tests."""

from __future__ import annotations

import sys

import pytest

from bruv.application import ConfigurationError
from bruv.backends.interface import DecisionBackend
from bruv.backends.needle import NeedleAdapter, NeedleSelection
from bruv.backends.registry import (
    BackendBuildContext,
    BackendDefinition,
    _BackendRegistry,
    backend_capabilities,
    backend_names,
    create_registered_backend,
    get_backend_definition,
)
from bruv.config import AppConfig, load_config
from bruv.contracts.spec import spec
from bruv.domain.validation import BackendCapabilities
from bruv.onboarding.credentials import Credentials


def test_registry_names_and_capabilities() -> None:
    assert backend_names == ("typesafe", "simple-jev", "needle")
    assert tuple(spec()["backends"]) == backend_names
    assert backend_capabilities("needle").backend == "needle"
    assert backend_capabilities("simple-jev").calibrated is False
    assert get_backend_definition("typesafe").needs_credentials is True


def test_registry_rejects_unknown_backend_without_paid_request() -> None:
    with pytest.raises(ConfigurationError) as caught:
        get_backend_definition("missing")
    assert caught.value.paid_request is False
    assert caught.value.message == "Unknown backend 'missing'."


def test_registry_rejects_duplicate_names() -> None:
    capabilities = BackendCapabilities(
        backend="duplicate",
        question_types=frozenset(),
        calibrated=False,
        allows_json_state=True,
    )

    def build(context: BackendBuildContext) -> DecisionBackend:
        raise AssertionError(context)

    definition = BackendDefinition(
        name="duplicate",
        capabilities=capabilities,
        build=build,
        setup_description="test",
    )
    with pytest.raises(RuntimeError, match="Duplicate backend registry name"):
        _BackendRegistry((definition, definition))


def test_registry_rejects_name_capabilities_mismatch() -> None:
    definition = BackendDefinition(
        name="definition-name",
        capabilities=BackendCapabilities(
            backend="capabilities-name",
            question_types=frozenset(),
            calibrated=False,
            allows_json_state=True,
        ),
        build=lambda context: (_ for _ in ()).throw(AssertionError(context)),
        setup_description="test",
    )
    with pytest.raises(RuntimeError, match="name must match capabilities backend"):
        _BackendRegistry((definition,))


def test_needle_construction_uses_generic_selected_dependency_override() -> None:
    class FakeRuntime:
        def classify(
            self, *, text: str, schema: dict[str, object], description: str
        ) -> NeedleSelection:
            raise AssertionError((text, schema, description))

    runtime = FakeRuntime()
    backend = create_registered_backend(
        BackendBuildContext(
            config=AppConfig(backend="needle"),
            credentials=Credentials(),
            selected_dependency_factory=lambda config, credentials: runtime,
        )
    )
    assert isinstance(backend, NeedleAdapter)
    assert backend._runtime is runtime


def test_listing_config_and_spec_do_not_import_optional_needle_package(
    monkeypatch, tmp_path
) -> None:
    sys.modules.pop("needle", None)
    imported: list[str] = []
    original_import = __import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "needle" or name.startswith("needle."):
            imported.append(name)
            raise AssertionError("optional needle package imported")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded_import)
    assert "needle" in backend_names
    assert (
        load_config(backend_override="needle", env={}, path=tmp_path / "missing.toml").backend
        == "needle"
    )
    assert "needle" in spec()["backends"]
    assert imported == []
