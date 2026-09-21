"""Integration tests for the offline `bruv validate` command."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from bruv.cli import app

runner = CliRunner()


def _eval_payload() -> dict:
    return {
        "state": "please route this ticket",
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "Which team handles this?",
                "criteria": {"sales": None, "billing": None},
            }
        },
    }


def test_validate_valid_request_exits_zero(tmp_path: Path) -> None:
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(_eval_payload()), encoding="utf-8")
    result = runner.invoke(app, ["validate", "-f", str(path)])
    assert result.exit_code == 0
    assert "valid" in result.stdout


def test_validate_invalid_request_exits_two(tmp_path: Path) -> None:
    payload = _eval_payload()
    payload["state"] = 5  # Simple Jev rejects top-level number
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = runner.invoke(app, ["validate", "-f", str(path), "--backend", "simple-jev"])
    assert result.exit_code == 2
    assert "invalid" in result.stdout


def test_validate_missing_eval_file_exits_two() -> None:
    result = runner.invoke(app, ["validate"])
    assert result.exit_code == 2


def test_validate_unknown_backend_exits_two_without_traceback(tmp_path: Path) -> None:
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(_eval_payload()), encoding="utf-8")
    result = runner.invoke(app, ["validate", "-f", str(path), "--backend", "bogus"])
    assert result.exit_code == 2
    assert result.exception is not None
    output = result.stdout + result.stderr
    assert "configuration_error" in output
    assert "Traceback" not in output


def test_validate_json_output(tmp_path: Path) -> None:
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(_eval_payload()), encoding="utf-8")
    result = runner.invoke(app, ["validate", "-f", str(path), "--output", "json"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "valid"


def test_validate_makes_no_backend_call(tmp_path: Path) -> None:
    # If validate ever calls a backend, _NeverCallBackend raises AssertionError,
    # which surfaces as a nonzero exit and an exception in output.
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(_eval_payload()), encoding="utf-8")
    result = runner.invoke(app, ["validate", "-f", str(path)])
    assert result.exit_code == 0
    assert "AssertionError" not in result.stdout
