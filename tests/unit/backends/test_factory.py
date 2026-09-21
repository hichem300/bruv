"""Backend factory selection and composition tests."""

from __future__ import annotations

import httpx
import pytest

from bruv.application import ConfigurationError
from bruv.backends.factory import backend_capabilities, create_backend
from bruv.backends.needle import NeedleAdapter, NeedleSelection
from bruv.backends.simple_jev import SimpleJevAdapter
from bruv.backends.typesafe import TypeSafeJevAdapter
from bruv.config import AppConfig
from bruv.onboarding.credentials import Credentials


def _simple_config() -> AppConfig:
    return AppConfig(backend="simple-jev")


def _typesafe_config() -> AppConfig:
    return AppConfig(backend="typesafe")


def test_creates_simple_jev_adapter_with_injected_client() -> None:
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})))
    backend = create_backend(_simple_config(), Credentials(), client_factory=lambda c, k: client)
    assert isinstance(backend, SimpleJevAdapter)


def test_creates_needle_adapter_with_injected_runtime() -> None:
    class FakeRuntime:
        def classify(self, **_kwargs: object) -> NeedleSelection:
            raise AssertionError("inference must not run during construction")

    runtime = FakeRuntime()
    backend = create_backend(
        AppConfig(backend="needle"),
        Credentials(),
        client_factory=lambda config, credentials: runtime,
    )
    assert isinstance(backend, NeedleAdapter)
    assert backend._runtime is runtime


def test_needle_capabilities_do_not_construct_runtime(monkeypatch) -> None:
    def fail_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "needle" or name.startswith("needle."):
            raise AssertionError("optional needle package imported")
        return original_import(name, *args, **kwargs)

    original_import = __import__
    monkeypatch.setattr("builtins.__import__", fail_import)
    capabilities = backend_capabilities(AppConfig(backend="needle"))
    assert capabilities.backend == "needle"


def test_creates_typesafe_adapter_with_injected_client() -> None:
    class FakeClient:
        pass

    fake = FakeClient()
    backend = create_backend(
        _typesafe_config(),
        Credentials(typesafe_api_key="key"),
        client_factory=lambda c, k: fake,
    )
    assert isinstance(backend, TypeSafeJevAdapter)


def test_default_typesafe_missing_sdk_raises(tmp_path, monkeypatch) -> None:
    # Ensure typesafe_sdk import fails.
    monkeypatch.syspath_prepend(str(tmp_path))
    import sys

    sys.modules.pop("typesafe_sdk", None)
    with pytest.raises(ConfigurationError):
        create_backend(_typesafe_config(), Credentials(typesafe_api_key="key"))


def test_default_typesafe_missing_credential_raises(tmp_path, monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(tmp_path))
    import sys

    sys.modules.pop("typesafe_sdk", None)
    with pytest.raises(ConfigurationError):
        create_backend(_typesafe_config(), Credentials())
