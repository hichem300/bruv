"""Pure request builder tests."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from bruv.command_builders import (
    MAX_STATE_BYTES,
    build_single_question_request,
    load_eval_request,
    load_state_file,
    parse_levels,
    parse_options,
    read_state_stdin,
)
from bruv.domain.questions import ChoiceQuestion, ScoreQuestion


def _eval_payload() -> dict:
    return {
        "state": "help",
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "Route?",
                "criteria": {"sales": None, "billing": None},
            }
        },
    }


def test_build_single_question_request_maps_fields() -> None:
    question = ChoiceQuestion(instructions="Route?", criteria={"sales": None, "billing": None})
    request = build_single_question_request(
        question_id="route", question=question, state="help", model="m"
    )
    assert request.state == "help"
    assert request.model == "m"
    assert set(request.questions) == {"route"}


def test_load_eval_request_json(tmp_path: Path) -> None:
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(_eval_payload()), encoding="utf-8")
    request = load_eval_request(path, None)
    assert set(request.questions) == {"route"}


def test_load_eval_request_yaml(tmp_path: Path) -> None:
    path = tmp_path / "eval.yaml"
    path.write_text(
        "state: help\nquestions:\n  route:\n    type: choice\n"
        "    instructions: Route?\n    criteria:\n      sales: null\n      billing: null\n",
        encoding="utf-8",
    )
    request = load_eval_request(path, None)
    assert set(request.questions) == {"route"}


def test_load_eval_request_state_override(tmp_path: Path) -> None:
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(_eval_payload()), encoding="utf-8")
    request = load_eval_request(path, "override")
    assert request.state == "override"


def test_load_eval_request_missing_state(tmp_path: Path) -> None:
    path = tmp_path / "eval.json"
    payload = _eval_payload()
    del payload["state"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="state"):
        load_eval_request(path, None)


def test_load_eval_request_missing_questions(tmp_path: Path) -> None:
    path = tmp_path / "eval.json"
    path.write_text(json.dumps({"state": "x"}), encoding="utf-8")
    with pytest.raises(ValueError, match="questions"):
        load_eval_request(path, None)


def test_load_state_file_json(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"k": 1}), encoding="utf-8")
    assert load_state_file(path) == {"k": 1}


def test_load_state_file_yaml(tmp_path: Path) -> None:
    path = tmp_path / "state.yaml"
    path.write_text("k: 1\n", encoding="utf-8")
    assert load_state_file(path) == {"k": 1}


def test_load_state_file_rejects_oversized(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_bytes(b'"' + b"x" * (MAX_STATE_BYTES + 1) + b'"')
    with pytest.raises(ValueError, match="ceiling"):
        load_state_file(path)


def test_read_state_stdin_json() -> None:
    stream = io.StringIO('{"k": 2}')
    assert read_state_stdin(stream) == {"k": 2}


def test_read_state_stdin_yaml_fallback() -> None:
    stream = io.StringIO("k: 3\n")
    assert read_state_stdin(stream) == {"k": 3}


def test_read_state_stdin_empty_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        read_state_stdin(io.StringIO(""))


def test_read_state_stdin_oversized_rejected() -> None:
    stream = io.StringIO('"' + "x" * (MAX_STATE_BYTES + 1) + '"')
    with pytest.raises(ValueError, match="ceiling"):
        read_state_stdin(stream)


def test_parse_options_with_descriptions() -> None:
    criteria = parse_options(["sales=Sales team", "billing"])
    assert criteria == {"sales": "Sales team", "billing": None}


def test_parse_levels_preserves_order() -> None:
    levels = parse_levels(["low", "medium", "high"])
    assert levels == ["low", "medium", "high"]


def test_score_question_builds_from_levels() -> None:
    levels = parse_levels(["low", "high"])
    question = ScoreQuestion(instructions="Rate?", criteria=levels)
    assert question.criteria == ["low", "high"]
