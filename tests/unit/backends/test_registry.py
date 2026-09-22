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
    assert backend_names == ("typesafe", "simple-jev", "needle", "rlcd-modernbert", "openrouter")
    assert tuple(spec()["backends"]) == backend_names
    assert backend_capabilities("needle").backend == "needle"
    assert backend_capabilities("simple-jev").calibrated is False
    assert get_backend_definition("typesafe").needs_credentials is True


def test_needle_registry_metadata_drives_doctor() -> None:
    definition = get_backend_definition("needle")
    assert definition.runs_local is True
    assert definition.optional_module == "needle"
    for name in ("typesafe", "simple-jev"):
        hosted = get_backend_definition(name)
        assert hosted.runs_local is False
        assert hosted.optional_module is None


def test_doctor_local_skips_come_from_registry_metadata(monkeypatch) -> None:
    """Doctor consults BackendDefinition.runs_local, not a backend-name list."""
    from bruv.onboarding import doctor as doctor_module

    def fake_definition(name: str):
        definition = get_backend_definition(name)
        if name == "simple-jev":
            return BackendDefinition(
                name=definition.name,
                capabilities=definition.capabilities,
                build=definition.build,
                setup_description=definition.setup_description,
                install_hint=definition.install_hint,
                needs_credentials=definition.needs_credentials,
                runs_local=True,
            )
        return definition

    monkeypatch.setattr(doctor_module, "get_backend_definition", fake_definition)
    results = doctor_module.run_doctor(
        config=AppConfig(backend="simple-jev"),  # type: ignore[arg-type]
        credentials=Credentials(),
        network_probe=lambda url: False,
    )
    for name in ("endpoint", "reachability"):
        item = next(r for r in results if r.name == name)
        assert item.ok is True
        assert "local runtime" in item.message


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


def test_needle_builder_passes_configured_model_without_running_inference() -> None:
    class FakeRuntime:
        def classify(self, **_kwargs: object) -> NeedleSelection:
            raise AssertionError("inference must not run")

    with pytest.raises(ConfigurationError, match="base Needle 3 model"):
        create_registered_backend(
            BackendBuildContext(
                config=AppConfig(backend="needle", needle_model="unsupported"),
                credentials=Credentials(),
                selected_dependency_factory=lambda config, credentials: FakeRuntime(),
            )
        )


def test_rlcd_registry_metadata_matches_adapter_capabilities() -> None:
    from bruv.backends.rlcd_modernbert import RlcdModernBertAdapter

    definition = get_backend_definition("rlcd-modernbert")
    assert definition.capabilities == RlcdModernBertAdapter.capabilities
    assert definition.needs_credentials is False
    assert definition.runs_local is True
    assert definition.optional_module is None
    assert definition.runtime_packages == (
        "onnxruntime",
        "tokenizers",
        "numpy",
        "huggingface_hub",
    )
    assert (
        definition.install_hint
        == "Install RLCD support with: pip install 'bruv[rlcd-modernbert] @ git+https://github.com/hichem300/bruv.git'"
    )


def test_config_rlcd_defaults_match_pinned_artifacts() -> None:
    from bruv.backends.rlcd_artifacts import REPO_ID, REVISION

    config = AppConfig()
    assert config.rlcd_model == REPO_ID
    assert config.rlcd_revision == REVISION


def test_rlcd_builder_forwards_config_pins(monkeypatch) -> None:
    import bruv.backends.rlcd_modernbert as rlcd_module

    captured: dict[str, str] = {}

    def fake_build_adapter(*, model: str, revision: str) -> object:
        captured["model"] = model
        captured["revision"] = revision
        return object()

    monkeypatch.setattr(rlcd_module, "build_adapter", fake_build_adapter)
    create_registered_backend(
        BackendBuildContext(
            config=AppConfig(
                backend="rlcd-modernbert",
                rlcd_model="custom/other",
                rlcd_revision="customrev",
            ),
            credentials=Credentials(),
        )
    )
    assert captured == {"model": "custom/other", "revision": "customrev"}


def test_build_adapter_rejects_unpinned_selection_before_artifacts(monkeypatch) -> None:
    import bruv.backends.rlcd_modernbert as rlcd_module

    def fail_artifacts() -> dict[str, object]:
        raise AssertionError("ensure_artifacts must not run for an unpinned selection")

    monkeypatch.setattr(rlcd_module, "ensure_artifacts", fail_artifacts)
    with pytest.raises(ConfigurationError, match="pinned ModernBERT model revision"):
        rlcd_module.build_adapter(model="wrong/other")
    with pytest.raises(ConfigurationError, match="pinned ModernBERT model revision"):
        rlcd_module.build_adapter(revision="wrongrev")


def test_listing_config_spec_and_dry_run_do_not_import_rlcd_modules(monkeypatch, tmp_path) -> None:
    for name in list(sys.modules):
        if name.startswith("bruv.backends.rlcd"):
            sys.modules.pop(name)
    forbidden = {
        "bruv.backends.rlcd_modernbert",
        "bruv.backends.rlcd_artifacts",
        "bruv.backends.rlcd_calibration",
        "onnxruntime",
        "tokenizers",
        "numpy",
        "huggingface_hub",
    }
    imported: list[str] = []
    original_import = __import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        root = name.split(".")[0]
        if name in forbidden or root in {"onnxruntime", "tokenizers", "numpy", "huggingface_hub"}:
            imported.append(name)
            raise AssertionError(f"optional/RLCD module imported: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded_import)
    from bruv.contracts.spec import spec as spec_fn

    assert "rlcd-modernbert" in backend_names
    assert (
        load_config(
            backend_override="rlcd-modernbert", env={}, path=tmp_path / "missing.toml"
        ).backend
        == "rlcd-modernbert"
    )
    assert "rlcd-modernbert" in spec_fn()["backends"]

    from typer.testing import CliRunner

    from bruv.cli import app

    runner = CliRunner()
    dry = runner.invoke(
        app,
        ["noul", "--backend", "rlcd-modernbert", "--dry-run", "--state", "ctx", "ok?"],
    )
    assert dry.exit_code == 0
    assert imported == []


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
