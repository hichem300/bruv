"""Pinned RLCD ModernBERT artifact management.

Downloads the exact pinned revision of the RLCD artifacts, verifies every
file's size and SHA-256 before returning it, and never executes, rewrites,
or falls back to an unpinned revision or global snapshot.

Importing this module never requires ``huggingface_hub`` or ``requests``;
optional dependencies are imported lazily inside the helpers. Core ``bruv``
stays installable without the optional ``rlcd-modernbert`` extra.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from bruv.application import BackendUnavailableError, ConfigurationError

REPO_ID = "heman10x/rlcd-modernbert-151m"
REVISION = "8af2496eb63c7fa66d7d234e1f62629380030eb4"
INSTALL_ACTION = "Install RLCD support with: pip install 'bruv[rlcd-modernbert]'"

OFFLINE_ACTION = (
    "Run bruv once while online so the pinned RLCD artifacts are cached, then retry offline."
)

_CHUNK_SIZE = 1024 * 1024

STATUS_VERIFIED = "verified"
STATUS_MISSING = "missing"
STATUS_HASH_MISMATCH = "hash-mismatch"
STATUS_DEPENDENCY_MISSING = "dependency-missing"

_OFFLINE_ENV_VALUES = frozenset({"1", "true", "yes", "on"})


@dataclass(frozen=True, slots=True)
class RequiredArtifact:
    """One pinned RLCD artifact with its exact expected size and SHA-256."""

    name: str
    size: int
    sha256: str


REQUIRED_ARTIFACTS: tuple[RequiredArtifact, ...] = (
    RequiredArtifact(
        name="model.onnx",
        size=606323181,
        sha256="4ae01f822538b000fa0e55859d4b3e6b40871d860149397e8784428b2a42ee5e",
    ),
    RequiredArtifact(
        name="tokenizer.json",
        size=3583596,
        sha256="8bb449eb0c037aae44115b65905bb339b8f3f74eb37067c19127feb3c0755723",
    ),
    RequiredArtifact(
        name="tokenizer_config.json",
        size=380,
        sha256="fb54f027372062b2ca52282efb04d178a8b57167a00cd8f4e816515823a2c016",
    ),
    RequiredArtifact(
        name="calibrator.json",
        size=1259,
        sha256="af2a876993148efa0726b6ccf710fe2303897d20c0ce8c7c9036eb50f64d23de",
    ),
)


def _sha256(path: Path) -> str:
    """Return the hex SHA-256 of ``path``, reading in fixed-size chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _offline_requested() -> bool:
    value = os.environ.get("HF_HUB_OFFLINE", "").strip().lower()
    return value in _OFFLINE_ENV_VALUES


def _dependency_error() -> ConfigurationError:
    return ConfigurationError(
        message="RLCD artifact management requires the optional rlcd-modernbert extra.",
        paid_request=False,
        action=INSTALL_ACTION,
    )


def _unavailable(message: str, *, action: str) -> BackendUnavailableError:
    return BackendUnavailableError(message=message, paid_request=False, action=action)


def _verify(artifact: RequiredArtifact, path: Path) -> Path:
    """Verify one downloaded file's type, size, and hash; return it on success."""
    if not path.is_file():
        raise _unavailable(
            f"RLCD artifact {artifact.name} is not a regular file on disk.",
            action=INSTALL_ACTION,
        )
    actual_size = path.stat().st_size
    if actual_size != artifact.size:
        raise _unavailable(
            f"RLCD artifact {artifact.name} has unexpected size "
            f"(expected {artifact.size}, found {actual_size}).",
            action=INSTALL_ACTION,
        )
    actual_sha256 = _sha256(path)
    if actual_sha256 != artifact.sha256:
        raise _unavailable(
            f"RLCD artifact {artifact.name} failed checksum verification "
            f"(expected {artifact.sha256}, found {actual_sha256}).",
            action=INSTALL_ACTION,
        )
    return path


def ensure_artifacts(repo_id: str = REPO_ID, revision: str = REVISION) -> dict[str, Path]:
    """Download and verify every pinned RLCD artifact at the exact revision.

    Returns a map of artifact name to verified local ``Path``. Each file is
    downloaded once via ``hf_hub_download`` with the exact pinned revision;
    there is never a fallback to another revision or a global snapshot.
    No file is returned before its size and SHA-256 verify, so nothing
    unverified is ever handed to the caller for execution.
    """
    try:
        from huggingface_hub import hf_hub_download  # type: ignore[import-not-found]
        from huggingface_hub.errors import (  # type: ignore[import-not-found]
            EntryNotFoundError,
            OfflineModeIsEnabled,
            RepositoryNotFoundError,
            RevisionNotFoundError,
        )
    except ImportError:
        raise _dependency_error() from None

    offline = _offline_requested()
    expected_failures: tuple[type[BaseException], ...] = (
        EntryNotFoundError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
        OfflineModeIsEnabled,
        OSError,
    )

    paths: dict[str, Path] = {}
    for artifact in REQUIRED_ARTIFACTS:
        try:
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=artifact.name,
                revision=revision,
                local_files_only=offline,
            )
        except expected_failures:
            if offline:
                raise _unavailable(
                    f"RLCD artifacts are unavailable offline: {artifact.name} is not in "
                    "the local Hugging Face cache.",
                    action=OFFLINE_ACTION,
                ) from None
            raise _unavailable(
                f"RLCD artifact {artifact.name} could not be downloaded from Hugging Face. "
                "Check network connectivity and repository availability.",
                action=INSTALL_ACTION,
            ) from None
        try:
            paths[artifact.name] = _verify(artifact, Path(downloaded))
        except OSError:
            raise _unavailable(
                f"RLCD artifact {artifact.name} could not be read for verification on disk.",
                action=INSTALL_ACTION,
            ) from None
    return paths


def cached_artifact_status(repo_id: str = REPO_ID, revision: str = REVISION) -> dict[str, str]:
    """Report each required RLCD artifact's local cache status without network.

    Doctor-only helper: never downloads anything and never runs inference.
    Each required artifact name maps to exactly ``verified``, ``missing``,
    ``hash-mismatch``, or ``dependency-missing``. ``verified`` requires both
    the exact size and the exact SHA-256. Cache-lookup or filesystem failures
    map safely to ``missing`` instead of crashing the doctor.
    """
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return {artifact.name: STATUS_DEPENDENCY_MISSING for artifact in REQUIRED_ARTIFACTS}

    statuses: dict[str, str] = {}
    for artifact in REQUIRED_ARTIFACTS:
        status = STATUS_MISSING
        try:
            found = try_to_load_from_cache(
                repo_id=repo_id,
                filename=artifact.name,
                revision=revision,
            )
            # ``found`` is a filename str, a ``None``, or a not-found sentinel.
            if isinstance(found, (str, Path)):
                path = Path(found)
                if path.is_file():
                    actual_size = path.stat().st_size
                    if actual_size == artifact.size and _sha256(path) == artifact.sha256:
                        status = STATUS_VERIFIED
                    else:
                        status = STATUS_HASH_MISMATCH
        except (OSError, ValueError):
            status = STATUS_MISSING
        statuses[artifact.name] = status
    return statuses


__all__ = [
    "INSTALL_ACTION",
    "REPO_ID",
    "REVISION",
    "RequiredArtifact",
    "REQUIRED_ARTIFACTS",
    "cached_artifact_status",
    "ensure_artifacts",
]
