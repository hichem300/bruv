"""Snapshot tests for spec and schema fixtures."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from bruv.cli import app

runner = CliRunner()
FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "cli"


def _snapshot(args: list[str], name: str) -> None:
    result = runner.invoke(app, args)
    assert result.exit_code == 0
    expected = json.loads((FIXTURE_DIR / name).read_text())
    assert json.loads(result.stdout) == expected


def test_spec_matches_snapshot() -> None:
    _snapshot(["spec", "--output", "json"], "spec.json")


def test_request_schema_matches_snapshot() -> None:
    _snapshot(["schema", "request"], "request.schema.json")


def test_output_schema_matches_snapshot() -> None:
    _snapshot(["schema", "output"], "output.schema.json")


def test_error_schema_matches_snapshot() -> None:
    _snapshot(["schema", "error"], "error.schema.json")
