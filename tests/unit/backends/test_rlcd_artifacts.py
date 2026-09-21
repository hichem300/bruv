"""Unit tests for pinned RLCD artifact management with fake huggingface_hub.

No network and no real 606MB model file: the artifact table is monkeypatched
to small temp-file definitions, and huggingface_hub is replaced by a fake
module that records every call.
"""

from __future__ import annotations

import hashlib
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from bruv.application import BackendUnavailableError, ConfigurationError
from bruv.backends import rlcd_artifacts
from bruv.backends.rlcd_artifacts import (
    INSTALL_ACTION,
    OFFLINE_ACTION,
    REPO_ID,
    REQUIRED_ARTIFACTS,
    REVISION,
    STATUS_DEPENDENCY_MISSING,
    STATUS_HASH_MISMATCH,
    STATUS_MISSING,
    STATUS_VERIFIED,
    RequiredArtifact,
    cached_artifact_status,
    ensure_artifacts,
    verify_artifact,
)

ARTIFACT_NAMES = ("model.onnx", "tokenizer.json", "tokenizer_config.json", "calibrator.json")

# Pinned upstream values, asserted independently of the module constants.
PINNED: tuple[tuple[str, int, str], ...] = (
    (
        "model.onnx",
        606323181,
        "4ae01f822538b000fa0e55859d4b3e6b40871d860149397e8784428b2a42ee5e",
    ),
    (
        "tokenizer.json",
        3583596,
        "8bb449eb0c037aae44115b65905bb339b8f3f74eb37067c19127feb3c0755723",
    ),
    (
        "tokenizer_config.json",
        380,
        "fb54f027372062b2ca52282efb04d178a8b57167a00cd8f4e816515823a2c016",
    ),
    (
        "calibrator.json",
        1259,
        "af2a876993148efa0726b6ccf710fe2303897d20c0ce8c7c9036eb50f64d23de",
    ),
)

_DOWNLOAD_CALLS: list[dict[str, Any]] = []
_CACHE_CALLS: list[dict[str, Any]] = []


@pytest.fixture(autouse=True)
def _isolated_call_recorders() -> Any:
    """Give every test empty recorder lists, before and after."""
    _DOWNLOAD_CALLS.clear()
    _CACHE_CALLS.clear()
    yield
    _DOWNLOAD_CALLS.clear()
    _CACHE_CALLS.clear()


def _small_artifacts(tmp_path: Path) -> tuple[tuple[RequiredArtifact, ...], dict[str, Path]]:
    """Small stand-in artifact definitions backed by real temp files."""
    contents = {name: f"content-of-{name}".encode() for name in ARTIFACT_NAMES}
    artifacts = tuple(
        RequiredArtifact(
            name=name,
            size=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
        )
        for name, data in contents.items()
    )
    files: dict[str, Path] = {}
    for name, data in contents.items():
        path = tmp_path / name
        path.write_bytes(data)
        files[name] = path
    return artifacts, files


class _FakeHubErrors:
    class EntryNotFoundError(Exception): ...

    class OfflineModeIsEnabled(Exception): ...

    class RepositoryNotFoundError(Exception): ...

    class RevisionNotFoundError(Exception): ...


def _install_fake_hub(
    monkeypatch: pytest.MonkeyPatch,
    *,
    files: dict[str, Path] | None = None,
    error: Exception | None = None,
) -> None:
    hub = types.ModuleType("huggingface_hub")
    errors_mod = types.ModuleType("huggingface_hub.errors")
    for name in (
        "EntryNotFoundError",
        "OfflineModeIsEnabled",
        "RepositoryNotFoundError",
        "RevisionNotFoundError",
    ):
        setattr(errors_mod, name, getattr(_FakeHubErrors, name))

    def hf_hub_download(repo_id: str, filename: str, revision: str, local_files_only: bool) -> Path:
        _DOWNLOAD_CALLS.append(
            {
                "repo_id": repo_id,
                "filename": filename,
                "revision": revision,
                "local_files_only": local_files_only,
            }
        )
        if error is not None:
            raise error
        return files[filename]

    hub.hf_hub_download = hf_hub_download  # type: ignore[attr-defined]
    hub.errors = errors_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors_mod)


def _install_cache_only_hub(monkeypatch: pytest.MonkeyPatch, result: Any) -> None:
    hub = types.ModuleType("huggingface_hub")

    def try_to_load_from_cache(repo_id: str, filename: str, revision: str) -> Any:
        _CACHE_CALLS.append({"repo_id": repo_id, "filename": filename, "revision": revision})
        return result

    hub.try_to_load_from_cache = try_to_load_from_cache  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)


def _install_cache_callables(monkeypatch: pytest.MonkeyPatch, loader: Any) -> None:
    hub = types.ModuleType("huggingface_hub")
    hub.try_to_load_from_cache = loader  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)


# ---------------------------------------------------------------------------
# Pinned artifact table
# ---------------------------------------------------------------------------


def test_required_artifacts_match_pinned_upstream_values() -> None:
    assert tuple(artifact.name for artifact in REQUIRED_ARTIFACTS) == ARTIFACT_NAMES
    for artifact, (name, size, sha256) in zip(REQUIRED_ARTIFACTS, PINNED, strict=True):
        assert artifact.name == name
        assert artifact.size == size
        assert artifact.sha256 == sha256


def test_repo_and_revision_pins_are_exact() -> None:
    assert REPO_ID == "heman10x/rlcd-modernbert-151m"
    assert REVISION == "8af2496eb63c7fa66d7d234e1f62629380030eb4"


# ---------------------------------------------------------------------------
# ensure_artifacts: exact calls, verification, offline, errors
# ---------------------------------------------------------------------------


def test_ensure_artifacts_downloads_each_file_at_exact_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts, files = _small_artifacts(tmp_path)
    monkeypatch.setattr(rlcd_artifacts, "REQUIRED_ARTIFACTS", artifacts)
    _install_fake_hub(monkeypatch, files=files)

    paths = ensure_artifacts()

    assert set(paths) == set(ARTIFACT_NAMES)
    for name, path in paths.items():
        assert path == files[name]
        assert path.is_file()
    assert len(_DOWNLOAD_CALLS) == len(ARTIFACT_NAMES)
    for call in _DOWNLOAD_CALLS:
        assert call["repo_id"] == REPO_ID
        assert call["revision"] == REVISION
        assert call["local_files_only"] is False
        assert call["filename"] in files


def test_ensure_artifacts_rejects_corrupt_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts, files = _small_artifacts(tmp_path)
    # Same length as the original content so the size check passes and the
    # checksum check is what fires.
    files["model.onnx"].write_bytes(b"X" * files["model.onnx"].stat().st_size)
    monkeypatch.setattr(rlcd_artifacts, "REQUIRED_ARTIFACTS", artifacts)
    _install_fake_hub(monkeypatch, files=files)

    with pytest.raises(BackendUnavailableError) as exc_info:
        ensure_artifacts()
    error = exc_info.value
    assert error.paid_request is False
    assert error.action == INSTALL_ACTION
    assert "failed checksum verification" in error.message
    assert str(tmp_path) not in error.message


def test_ensure_artifacts_rejects_wrong_size_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts, files = _small_artifacts(tmp_path)
    files["tokenizer.json"].write_bytes(b"short")
    monkeypatch.setattr(rlcd_artifacts, "REQUIRED_ARTIFACTS", artifacts)
    _install_fake_hub(monkeypatch, files=files)

    with pytest.raises(BackendUnavailableError) as exc_info:
        ensure_artifacts()
    assert "unexpected size" in exc_info.value.message


def test_ensure_artifacts_rejects_directory_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts, files = _small_artifacts(tmp_path)
    files["tokenizer_config.json"].unlink()
    directory = tmp_path / "tokenizer_config.json"
    directory.mkdir()
    monkeypatch.setattr(rlcd_artifacts, "REQUIRED_ARTIFACTS", artifacts)
    _install_fake_hub(
        monkeypatch,
        files={**{n: tmp_path / n for n in ARTIFACT_NAMES}, "tokenizer_config.json": directory},
    )

    with pytest.raises(BackendUnavailableError) as exc_info:
        ensure_artifacts()
    assert "not a regular file" in exc_info.value.message


def test_ensure_artifacts_offline_env_values_include_uppercase_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts, files = _small_artifacts(tmp_path)
    monkeypatch.setattr(rlcd_artifacts, "REQUIRED_ARTIFACTS", artifacts)
    _install_fake_hub(monkeypatch, files=files)

    for value in ("1", "true", "yes", "on", "ON", "True"):
        _DOWNLOAD_CALLS.clear()
        monkeypatch.setenv("HF_HUB_OFFLINE", value)
        ensure_artifacts()
        assert all(call["local_files_only"] is True for call in _DOWNLOAD_CALLS), value

    _DOWNLOAD_CALLS.clear()
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    ensure_artifacts()
    assert all(call["local_files_only"] is False for call in _DOWNLOAD_CALLS)

    monkeypatch.delenv("HF_HUB_OFFLINE")
    _DOWNLOAD_CALLS.clear()
    ensure_artifacts()
    assert all(call["local_files_only"] is False for call in _DOWNLOAD_CALLS)


@pytest.mark.parametrize(
    ("error", "offline"),
    [
        (_FakeHubErrors.EntryNotFoundError("gone"), False),
        (_FakeHubErrors.RepositoryNotFoundError("no repo"), False),
        (_FakeHubErrors.RevisionNotFoundError("no rev"), False),
        (_FakeHubErrors.OfflineModeIsEnabled("offline"), True),
        (OSError("disk"), False),
        (OSError("disk"), True),
    ],
)
def test_ensure_artifacts_download_errors_fail_unpaid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    offline: bool,
) -> None:
    artifacts, files = _small_artifacts(tmp_path)
    monkeypatch.setattr(rlcd_artifacts, "REQUIRED_ARTIFACTS", artifacts)
    _install_fake_hub(monkeypatch, files=files, error=error)
    if offline:
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    else:
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)

    with pytest.raises(BackendUnavailableError) as exc_info:
        ensure_artifacts()
    error_obj = exc_info.value
    assert error_obj.paid_request is False
    if offline:
        assert error_obj.action == OFFLINE_ACTION
        assert "unavailable offline" in error_obj.message
    else:
        assert error_obj.action == INSTALL_ACTION
        assert "could not be downloaded" in error_obj.message
    assert str(tmp_path) not in error_obj.message


def test_ensure_artifacts_read_race_failure_is_unpaid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts, files = _small_artifacts(tmp_path)
    monkeypatch.setattr(rlcd_artifacts, "REQUIRED_ARTIFACTS", artifacts)
    _install_fake_hub(monkeypatch, files=files)

    def raise_oserror(path: Path) -> str:
        raise OSError("file vanished during hashing")

    monkeypatch.setattr(rlcd_artifacts, "_sha256", raise_oserror)

    with pytest.raises(BackendUnavailableError) as exc_info:
        ensure_artifacts()
    assert "could not be read for verification" in exc_info.value.message
    assert exc_info.value.action == INSTALL_ACTION


def test_missing_huggingface_hub_dependency_fails_with_install_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", None)

    with pytest.raises(ConfigurationError) as exc_info:
        ensure_artifacts()
    error = exc_info.value
    assert error.paid_request is False
    assert error.action == INSTALL_ACTION
    assert "rlcd-modernbert" in error.message


# ---------------------------------------------------------------------------
# verify_artifact: consumption-boundary reverification
# ---------------------------------------------------------------------------


def test_verify_artifact_accepts_intact_file_for_every_pinned_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts, files = _small_artifacts(tmp_path)
    monkeypatch.setattr(rlcd_artifacts, "REQUIRED_ARTIFACTS", artifacts)

    for name, path in files.items():
        assert verify_artifact(path, name) == path
        assert path.is_file()


def test_verify_artifact_rejects_tampered_file_with_checksum_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts, files = _small_artifacts(tmp_path)
    files["model.onnx"].write_bytes(b"X" * files["model.onnx"].stat().st_size)
    monkeypatch.setattr(rlcd_artifacts, "REQUIRED_ARTIFACTS", artifacts)

    with pytest.raises(BackendUnavailableError) as exc_info:
        verify_artifact(files["model.onnx"], "model.onnx")
    error = exc_info.value
    assert error.paid_request is False
    assert error.action == INSTALL_ACTION
    assert "failed checksum verification" in error.message
    assert str(tmp_path) not in error.message


def test_verify_artifact_rejects_wrong_size_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts, files = _small_artifacts(tmp_path)
    files["tokenizer.json"].write_bytes(b"short")
    monkeypatch.setattr(rlcd_artifacts, "REQUIRED_ARTIFACTS", artifacts)

    with pytest.raises(BackendUnavailableError) as exc_info:
        verify_artifact(files["tokenizer.json"], "tokenizer.json")
    assert "unexpected size" in exc_info.value.message
    assert exc_info.value.paid_request is False


def test_verify_artifact_rejects_directory_and_missing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts, files = _small_artifacts(tmp_path)
    directory = tmp_path / "dir"
    directory.mkdir()
    monkeypatch.setattr(rlcd_artifacts, "REQUIRED_ARTIFACTS", artifacts)

    with pytest.raises(BackendUnavailableError) as exc_info:
        verify_artifact(directory, "calibrator.json")
    assert "not a regular file" in exc_info.value.message

    with pytest.raises(BackendUnavailableError) as exc_info:
        verify_artifact(tmp_path / "missing.json", "calibrator.json")
    assert "not a regular file" in exc_info.value.message
    assert str(tmp_path) not in exc_info.value.message


def test_verify_artifact_unknown_name_fails_sanitized_and_unpaid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts, files = _small_artifacts(tmp_path)
    monkeypatch.setattr(rlcd_artifacts, "REQUIRED_ARTIFACTS", artifacts)

    with pytest.raises(BackendUnavailableError) as exc_info:
        verify_artifact(files["model.onnx"], "../escape.onnx")
    error = exc_info.value
    assert error.paid_request is False
    assert error.action == INSTALL_ACTION
    assert "not a pinned artifact name" in error.message
    assert str(tmp_path) not in error.message


def test_verify_artifact_read_race_failure_is_unpaid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts, files = _small_artifacts(tmp_path)
    monkeypatch.setattr(rlcd_artifacts, "REQUIRED_ARTIFACTS", artifacts)

    def raise_oserror(path: Path) -> str:
        raise OSError("file vanished during hashing")

    monkeypatch.setattr(rlcd_artifacts, "_sha256", raise_oserror)

    with pytest.raises(BackendUnavailableError) as exc_info:
        verify_artifact(files["calibrator.json"], "calibrator.json")
    assert "could not be read for verification on disk" in exc_info.value.message
    assert exc_info.value.action == INSTALL_ACTION
    assert exc_info.value.paid_request is False


# ---------------------------------------------------------------------------
# cached_artifact_status: zero-network cache checks
# ---------------------------------------------------------------------------


def test_cached_status_verified_requires_exact_size_and_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts, files = _small_artifacts(tmp_path)
    monkeypatch.setattr(rlcd_artifacts, "REQUIRED_ARTIFACTS", artifacts)

    # Every file intact: all verified.
    _install_cache_callables(monkeypatch, lambda repo_id, filename, revision: str(files[filename]))
    statuses = cached_artifact_status()
    assert all(status == STATUS_VERIFIED for status in statuses.values())

    # Corrupt one file: same name, different bytes, wrong hash.
    files["model.onnx"].write_bytes(b"tampered-bytes-with-different-hash")
    statuses = cached_artifact_status()
    assert statuses["model.onnx"] == STATUS_HASH_MISMATCH
    assert statuses["tokenizer.json"] == STATUS_VERIFIED

    # Restore hash but break the size.
    files["model.onnx"].write_bytes(b"x")
    statuses = cached_artifact_status()
    assert statuses["model.onnx"] == STATUS_HASH_MISMATCH

    # Cache points at a nonexistent path: missing, not a crash.
    _install_cache_callables(
        monkeypatch, lambda repo_id, filename, revision: str(tmp_path / "missing.bin")
    )
    statuses = cached_artifact_status()
    assert all(status == STATUS_MISSING for status in statuses.values())


def test_cached_status_none_and_sentinel_results_map_to_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts, _files = _small_artifacts(tmp_path)
    monkeypatch.setattr(rlcd_artifacts, "REQUIRED_ARTIFACTS", artifacts)

    class _CachedNoExist:
        pass

    for result in (None, _CachedNoExist(), 42):
        _install_cache_only_hub(monkeypatch, result)
        statuses = cached_artifact_status()
        assert set(statuses) == set(ARTIFACT_NAMES)
        assert all(status == STATUS_MISSING for status in statuses.values())


def test_cached_status_path_object_is_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts, files = _small_artifacts(tmp_path)
    monkeypatch.setattr(rlcd_artifacts, "REQUIRED_ARTIFACTS", artifacts)
    _install_cache_only_hub(monkeypatch, files["calibrator.json"])

    statuses = cached_artifact_status()
    assert statuses["calibrator.json"] == STATUS_VERIFIED


def test_cached_status_cache_lookup_failure_maps_to_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts, _files = _small_artifacts(tmp_path)
    monkeypatch.setattr(rlcd_artifacts, "REQUIRED_ARTIFACTS", artifacts)
    _install_cache_only_hub(monkeypatch, OSError("cache exploded"))

    statuses = cached_artifact_status()
    assert all(status == STATUS_MISSING for status in statuses.values())


def test_cached_status_passes_exact_repo_and_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts, _files = _small_artifacts(tmp_path)
    monkeypatch.setattr(rlcd_artifacts, "REQUIRED_ARTIFACTS", artifacts)
    _install_cache_only_hub(monkeypatch, None)

    cached_artifact_status()

    assert len(_CACHE_CALLS) == len(ARTIFACT_NAMES)
    for call in _CACHE_CALLS:
        assert call["repo_id"] == REPO_ID
        assert call["revision"] == REVISION


def test_cached_status_never_touches_download_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts, _files = _small_artifacts(tmp_path)
    monkeypatch.setattr(rlcd_artifacts, "REQUIRED_ARTIFACTS", artifacts)
    _install_cache_only_hub(monkeypatch, None)

    cached_artifact_status()

    # The cache-only fake module defines no hf_hub_download at all, and no
    # download call was recorded anywhere in this run.
    assert _DOWNLOAD_CALLS == []


def test_cached_status_missing_dependency_reports_every_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)

    statuses = cached_artifact_status()
    assert set(statuses) == set(ARTIFACT_NAMES)
    assert all(status == STATUS_DEPENDENCY_MISSING for status in statuses.values())


def test_status_constants_are_pinned() -> None:
    assert STATUS_VERIFIED == "verified"
    assert STATUS_MISSING == "missing"
    assert STATUS_HASH_MISMATCH == "hash-mismatch"
    assert STATUS_DEPENDENCY_MISSING == "dependency-missing"
