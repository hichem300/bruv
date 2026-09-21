"""Credential-free machine contract tests for spec and schema."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from bruv.cli import app

runner = CliRunner()


def test_spec_json_runs_without_credentials() -> None:
    result = runner.invoke(app, ["spec", "--output", "json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["version"]
    assert any(c["name"] == "validate" for c in payload["commands"])
    assert "paid_request_semantics" in payload


def test_schema_request_has_id_and_version() -> None:
    result = runner.invoke(app, ["schema", "request"])
    assert result.exit_code == 0
    schema = json.loads(result.stdout)
    assert schema["$id"]
    assert schema["version"] == "0.1.0"


def test_schema_output_has_id_and_version() -> None:
    result = runner.invoke(app, ["schema", "output"])
    assert result.exit_code == 0
    schema = json.loads(result.stdout)
    assert schema["$id"]
    assert schema["version"] == "0.1.0"


def test_schema_error_has_id_and_version() -> None:
    result = runner.invoke(app, ["schema", "error"])
    assert result.exit_code == 0
    schema = json.loads(result.stdout)
    assert schema["$id"]
    assert schema["version"] == "0.1.0"


def test_schema_unknown_name_exits_two() -> None:
    result = runner.invoke(app, ["schema", "bogus"])
    assert result.exit_code == 2


def test_spec_lists_all_primary_commands() -> None:
    result = runner.invoke(app, ["spec", "--output", "json"])
    payload = json.loads(result.stdout)
    names = {c["name"] for c in payload["commands"]}
    assert {"noul", "choice", "score", "eval", "validate", "spec", "schema"} <= names


def test_spec_backends_include_needle() -> None:
    result = runner.invoke(app, ["spec", "--output", "json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert "needle" in payload["backends"]


def test_output_schema_advertises_needle_and_relaxed_confidence_only_fields() -> None:
    result = runner.invoke(app, ["schema", "output"])
    assert result.exit_code == 0
    schema = json.loads(result.stdout)
    assert schema["properties"]["backend"]["type"] == "string"

    noul = schema["$defs"]["NoulAnswer"]
    assert "value" in noul["properties"]
    assert "confidence" in noul["properties"]
    assert "required" not in noul  # value/confidence are an optional pairing

    for name in ("ChoiceAnswer", "ScoreAnswer"):
        answer = schema["$defs"][name]
        assert "probabilities" in answer["properties"]
        assert "probabilities" not in answer.get("required", [])
