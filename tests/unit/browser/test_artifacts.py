"""Artifact safety tests: path validation, defaults, transcript privacy."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bruv.browser.artifacts import (
    PRIVACY_WARNING,
    ArtifactOptions,
    ArtifactStore,
    TranscriptWriter,
    privacy_warning,
    safe_origin,
    validate_session_id,
)


def test_invalid_session_ids_rejected(tmp_path: Path) -> None:
    for bad in ("../evil", "a/b", "a\\b", ".hidden", "-lead", "", "a" * 65, "a..b"):
        with pytest.raises(ValueError):
            validate_session_id(bad)
        with pytest.raises(ValueError):
            ArtifactStore(bad, artifact_dir=tmp_path)


def test_disabled_options_skip_filename_validation() -> None:
    options = ArtifactOptions(trace_file_name="../evil.zip", video_file_name="a/b.webm")
    assert options.trace is False and options.video is False


def test_enabled_trace_video_traversal_rejected() -> None:
    with pytest.raises(ValueError, match="trace_file_name"):
        ArtifactOptions(trace=True, trace_file_name="../evil.zip")
    with pytest.raises(ValueError, match="video_file_name"):
        ArtifactOptions(video=True, video_file_name="sub/dir.webm")
    with pytest.raises(ValueError, match="video_file_name"):
        ArtifactOptions(video=True, video_file_name="..webm")


def test_trace_and_video_off_by_default(tmp_path: Path) -> None:
    store = ArtifactStore("s1", artifact_dir=tmp_path)
    assert store.options.trace is False
    assert store.options.video is False
    artifacts = store.prepare()
    assert artifacts.trace_path is None
    assert artifacts.video_path is None


def test_transcript_records_only_safe_metadata(tmp_path: Path) -> None:
    writer = TranscriptWriter(tmp_path / "s1" / TRANSCRIPT_NAME)
    writer.record(
        step=1,
        kind="navigate",
        outcome="ok",
        candidate_id="c0",
        origin="https://example.com",
    )
    writer.record(step=2, kind="click", outcome="failed", error_kind="StaleActionError")
    lines = (tmp_path / "s1" / TRANSCRIPT_NAME).read_text().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["kind"] == "navigate" and first["outcome"] == "ok"
    assert first["origin"] == "https://example.com"
    assert "candidate_id" in first
    assert set(json.loads(lines[1])) <= {"timestamp", "step", "kind", "outcome", "error"}
    for line in lines:
        assert "typed text" not in line
        assert "https://example.com/path?q=1" not in line


TRANSCRIPT_NAME = "transcript.jsonl"


def test_safe_origin_strips_path_and_query() -> None:
    assert safe_origin("https://Example.com/path/page?token=1#frag") == "https://example.com"
    assert safe_origin("http://other.org/x") == "http://other.org"
    assert safe_origin("not a url") == ""


def test_privacy_warning_exists() -> None:
    assert privacy_warning() == PRIVACY_WARNING
    assert "private" in PRIVACY_WARNING.lower()


def test_session_paths_under_requested_root(tmp_path: Path) -> None:
    store = ArtifactStore(
        "sess-01",
        artifact_dir=tmp_path,
        options=ArtifactOptions(trace=True, video=True),
    )
    assert store.root == tmp_path / "sess-01"
    artifacts = store.prepare()
    for path in (
        artifacts.screenshot_path,
        artifacts.transcript_path,
        artifacts.trace_path,
        artifacts.video_path,
    ):
        assert path is not None and Path(path).is_relative_to(tmp_path)
