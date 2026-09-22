"""Protected credential storage for the TypeSafe backend.

No credential is ever accepted through a CLI flag. Credentials resolve in this
order: ``TYPESAFE_API_KEY`` environment variable first, then a user-owned
credential file. On POSIX the credential file is mode ``0600``. On Windows the
file ACL is restricted to the current user; if secure permissions cannot be
applied, persistence is refused and the environment variable is recommended.
"""

from __future__ import annotations

import os
import stat
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import platformdirs

from bruv.application import ConfigurationError

CREDENTIAL_ENV = "TYPESAFE_API_KEY"
OPENROUTER_API_KEY = "OPENROUTER_API_KEY"


@dataclass(frozen=True, slots=True)
class Credentials:
    """Resolved credentials. ``None`` means no credential is available."""

    typesafe_api_key: str | None = None
    openrouter_api_key: str | None = None

    @property
    def has_typesafe(self) -> bool:
        return bool(self.typesafe_api_key)

    @property
    def has_openrouter(self) -> bool:
        return bool(self.openrouter_api_key)


def credentials_dir() -> Path:
    return Path(platformdirs.user_config_dir("bruv", appauthor=False))


def credentials_path() -> Path:
    return credentials_dir() / "credentials"


def load_credentials(
    *,
    env: Mapping[str, str] | None = None,
    path: Path | None = None,
) -> Credentials:
    """Resolve credentials without ever printing their values."""
    environment = env if env is not None else os.environ
    env_typesafe = environment.get(CREDENTIAL_ENV)
    env_openrouter = environment.get(OPENROUTER_API_KEY)
    if env_typesafe and env_typesafe.strip() and env_openrouter and env_openrouter.strip():
        return Credentials(typesafe_api_key=env_typesafe, openrouter_api_key=env_openrouter)

    credential_file = path if path is not None else credentials_path()
    if not credential_file.is_file():
        return Credentials(
            typesafe_api_key=env_typesafe if env_typesafe and env_typesafe.strip() else None,
            openrouter_api_key=env_openrouter if env_openrouter and env_openrouter.strip() else None,
        )

    if _insecure_permissions(credential_file):
        raise ConfigurationError(
            message="Credential file permissions are too open.",
            paid_request=False,
            action=f"Run `chmod 600 {credential_file}` or move the key to {CREDENTIAL_ENV}.",
        )

    try:
        text = credential_file.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigurationError(
            message="Could not read credential file.",
            paid_request=False,
            action=f"Move the key to the {CREDENTIAL_ENV} environment variable.",
        ) from exc

    keys = _parse_credential_text(text)
    return Credentials(
        typesafe_api_key=(env_typesafe if env_typesafe and env_typesafe.strip() else None)
        or keys["TYPESAFE_API_KEY"] or None,
        openrouter_api_key=(env_openrouter if env_openrouter and env_openrouter.strip() else None)
        or keys["OPENROUTER_API_KEY"] or None,
    )


def _parse_credential_text(text: str) -> dict[str, str | None]:
    """Parse a minimal key=value credential file with one or more keys."""
    parsed: dict[str, str | None] = {"TYPESAFE_API_KEY": None, "OPENROUTER_API_KEY": None}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for name in ("TYPESAFE_API_KEY", "OPENROUTER_API_KEY"):
            if line.startswith(f"{name}="):
                value = line.split("=", 1)[1].strip().strip('"').strip("'")
                parsed[name] = value if value else None
    return parsed


def _insecure_permissions(path: Path) -> bool:
    if sys.platform == "win32":
        return False  # ACL checks are applied at save time; reads trust them.
    mode = path.stat().st_mode
    return bool(mode & (stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH))


def save_credentials(key: str, *, path: Path | None = None) -> Path:
    """Persist the TypeSafe credential safely or refuse with ConfigurationError."""
    return save_named_credential("TYPESAFE_API_KEY", key, path=path)


def save_openrouter_credentials(key: str, *, path: Path | None = None) -> Path:
    """Persist the OpenRouter credential safely or refuse with ConfigurationError."""
    return save_named_credential("OPENROUTER_API_KEY", key, path=path)


def save_named_credential(name: str, key: str, *, path: Path | None = None) -> Path:
    """Persist one named credential while preserving other existing keys."""
    if not key or not key.strip():
        raise ConfigurationError(
            message="Cannot save an empty credential.",
            paid_request=False,
            action="Provide a non-empty API key.",
        )
    if name not in ("TYPESAFE_API_KEY", "OPENROUTER_API_KEY"):
        raise ConfigurationError(
            message=f"Unknown credential name {name!r}.",
            paid_request=False,
            action="Use TYPESAFE_API_KEY or OPENROUTER_API_KEY.",
        )
    credential_file = path if path is not None else credentials_path()
    credential_file.parent.mkdir(parents=True, exist_ok=True)

    existing: dict[str, str | None] = {"TYPESAFE_API_KEY": None, "OPENROUTER_API_KEY": None}
    if credential_file.is_file() and not _insecure_permissions(credential_file):
        try:
            existing.update(_parse_credential_text(credential_file.read_text(encoding="utf-8")))
        except OSError:
            pass
    existing[name] = key

    payload = "".join(f'{k}="{v}"\n' for k, v in existing.items() if v)
    if sys.platform == "win32":
        _restrict_windows_acl(credential_file)
    _write_restricted(credential_file, payload)
    return credential_file


def _write_restricted(path: Path, payload: str) -> None:
    if sys.platform != "win32":
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
        finally:
            pass
        if (path.stat().st_mode & 0o077) != 0:
            os.chmod(path, 0o600)
    else:
        path.write_text(payload, encoding="utf-8")
        _restrict_windows_acl(path)


def _restrict_windows_acl(path: Path) -> None:
    """Restrict a credential file to the current user on Windows.

    If the restriction cannot be applied, refuse persistence and recommend the
    environment variable instead.
    """
    try:
        import subprocess

        result = subprocess.run(  # noqa: S603,S607
            ["icacls", str(path), "/inheritance:r", "/grant:r", f"{os.getlogin()}:F"],
            check=False,
            capture_output=True,
        )
        if result.returncode != 0:
            raise ConfigurationError(
                message="Could not restrict credential file permissions on Windows.",
                paid_request=False,
                action=f"Use the {CREDENTIAL_ENV} environment variable instead.",
            )
    except (OSError, FileNotFoundError) as exc:
        raise ConfigurationError(
            message="Could not restrict credential file permissions on Windows.",
            paid_request=False,
            action=f"Use the {CREDENTIAL_ENV} environment variable instead.",
        ) from exc


__all__ = [
    "CREDENTIAL_ENV",
    "OPENROUTER_API_KEY",
    "Credentials",
    "credentials_dir",
    "credentials_path",
    "load_credentials",
    "save_credentials",
    "save_openrouter_credentials",
]
