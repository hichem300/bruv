"""Credential precedence and protected persistence tests."""

from __future__ import annotations

import os
import stat
import sys

import pytest

from bruv.application import ConfigurationError
from bruv.onboarding.credentials import load_credentials, save_credentials


def test_env_credential_takes_precedence(tmp_path) -> None:
    cred_file = tmp_path / "credentials"
    cred_file.write_text('TYPESAFE_API_KEY="from-file"\n', encoding="utf-8")
    os.chmod(cred_file, 0o600)
    creds = load_credentials(env={"TYPESAFE_API_KEY": "from-env"}, path=cred_file)
    assert creds.typesafe_api_key == "from-env"


def test_file_credential_used_when_env_absent(tmp_path) -> None:
    cred_file = tmp_path / "credentials"
    cred_file.write_text('TYPESAFE_API_KEY="from-file"\n', encoding="utf-8")
    os.chmod(cred_file, 0o600)
    creds = load_credentials(env={}, path=cred_file)
    assert creds.typesafe_api_key == "from-file"


def test_missing_credential_returns_none(tmp_path) -> None:
    creds = load_credentials(env={}, path=tmp_path / "missing")
    assert creds.typesafe_api_key is None
    assert creds.has_typesafe is False


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission check")
def test_insecure_file_permissions_rejected(tmp_path) -> None:
    cred_file = tmp_path / "credentials"
    cred_file.write_text('TYPESAFE_API_KEY="key"\n', encoding="utf-8")
    os.chmod(cred_file, 0o644)
    with pytest.raises(ConfigurationError):
        load_credentials(env={}, path=cred_file)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission write")
def test_save_creates_mode_0600(tmp_path) -> None:
    cred_file = tmp_path / "credentials"
    save_credentials("a-secret-key", path=cred_file)
    mode = cred_file.stat().st_mode
    assert (mode & 0o077) == 0
    assert (mode & stat.S_IRUSR) and (mode & stat.S_IWUSR)
    creds = load_credentials(env={}, path=cred_file)
    assert creds.typesafe_api_key == "a-secret-key"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission write")
def test_save_refuses_empty_key(tmp_path) -> None:
    with pytest.raises(ConfigurationError):
        save_credentials("   ", path=tmp_path / "credentials")


def test_credentials_never_accept_flags() -> None:
    # load_credentials exposes no flag-style parameter; this is a structural
    # guard against adding one.
    import inspect

    params = inspect.signature(load_credentials).parameters
    assert "flag" not in params
    assert "api_key" not in params
