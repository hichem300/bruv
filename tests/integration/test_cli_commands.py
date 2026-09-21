"""Integration tests for primary CLI commands."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from bruv.cli import app
from bruv.domain.results import ChoiceAnswer, DecisionResult, NoulAnswer

runner = CliRunner()


def _fake_result() -> DecisionResult:
    return DecisionResult(
        backend="simple-jev",
        model="Qwen/Qwen3.5-0.8B",
        calibrated=False,
        answers={
            "q1": ChoiceAnswer(
                type="choice",
                choice="billing",
                confidence=0.9,
                probabilities={"billing": 0.9, "other": 0.1},
            )
        },
    )


class _FakeBackend:
    capabilities = type(
        "C",
        (),
        {
            "backend": "simple-jev",
            "question_types": frozenset({"noul", "choice", "score"}),
            "calibrated": False,
            "allows_json_state": True,
        },
    )()

    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, request: object) -> DecisionResult:
        self.calls += 1
        return _fake_result()


@pytest.fixture
def fake_backend(monkeypatch):
    backend = _FakeBackend()

    def _create_backend(config, credentials, *, client_factory=None):
        return backend

    monkeypatch.setattr("bruv.cli.create_backend", _create_backend)
    return backend


def test_choice_command_human_output(fake_backend) -> None:
    result = runner.invoke(
        app,
        ["choice", "Route?", "--option", "sales", "--option", "billing", "--state", "help"],
    )
    assert result.exit_code == 0
    assert "billing" in result.stdout
    assert "uncalibrated" in result.stdout
    assert fake_backend.calls == 1


def test_choice_command_json_output(fake_backend) -> None:
    result = runner.invoke(
        app,
        [
            "choice",
            "Route?",
            "--option",
            "sales",
            "--option",
            "billing",
            "--state",
            "help",
            "--output",
            "json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["backend"] == "simple-jev"
    assert payload["calibrated"] is False


def test_dry_run_makes_no_backend_call(fake_backend) -> None:
    result = runner.invoke(
        app,
        [
            "choice",
            "Route?",
            "--option",
            "sales",
            "--option",
            "billing",
            "--state",
            "help",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert fake_backend.calls == 0
    payload = json.loads(result.stdout)
    assert payload["dry_run"] is True


def test_needle_dry_run_stays_lazy(monkeypatch) -> None:
    sys.modules.pop("needle", None)
    original_import = __import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "needle" or name.startswith("needle."):
            raise AssertionError("optional needle package imported")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded_import)
    result = runner.invoke(
        app,
        ["noul", "Decide?", "--state", "context", "--backend", "needle", "--dry-run"],
    )
    assert result.exit_code == 0
    assert json.loads(result.stdout)["backend"] == "needle"


def test_needle_json_output_serializes_confidence_only(monkeypatch) -> None:
    needle_result = DecisionResult(
        backend="needle",
        model="Cactus-Compute/needle3",
        calibrated=True,
        answers={"q1": NoulAnswer(value=True, confidence=0.9)},
        provider_metadata={"confidence_scope": "whole_response", "probabilities_available": False},
    )

    class _NeedleBackend:
        capabilities = type(
            "C",
            (),
            {
                "backend": "needle",
                "question_types": frozenset({"noul", "choice", "score"}),
                "calibrated": True,
                "allows_json_state": True,
            },
        )()

        def evaluate(self, request: object) -> DecisionResult:
            return needle_result

    def _create_needle_backend(config, credentials, *, client_factory=None):
        return _NeedleBackend()

    monkeypatch.setattr("bruv.cli.create_backend", _create_needle_backend)
    result = runner.invoke(
        app,
        ["noul", "Refund?", "--state", "ctx", "--backend", "needle", "--output", "json"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["backend"] == "needle"
    assert payload["model"] == "Cactus-Compute/needle3"
    assert payload["calibrated"] is True
    assert payload["answers"]["q1"] == {"type": "noul", "value": True, "confidence": 0.9}
    assert "probabilities" not in payload["answers"]["q1"]
    assert payload["provider_metadata"] == {
        "confidence_scope": "whole_response",
        "probabilities_available": False,
    }


def test_missing_state_exits_two(fake_backend) -> None:
    result = runner.invoke(app, ["choice", "Route?", "--option", "sales", "--option", "billing"])
    assert result.exit_code == 2


def test_noul_command_runs(fake_backend) -> None:
    result = runner.invoke(app, ["noul", "Yes?", "--state", "ctx"])
    assert result.exit_code == 0


def test_score_command_runs(fake_backend) -> None:
    result = runner.invoke(
        app, ["score", "Rate?", "--level", "low", "--level", "high", "--state", "ctx"]
    )
    assert result.exit_code == 0


def test_eval_command_runs(tmp_path: Path, fake_backend) -> None:
    path = tmp_path / "eval.json"
    path.write_text(
        json.dumps(
            {
                "state": "help",
                "questions": {
                    "q1": {
                        "type": "choice",
                        "instructions": "Route?",
                        "criteria": {"sales": None, "billing": None},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    result = runner.invoke(app, ["eval", str(path)])
    assert result.exit_code == 0


def test_field_extraction_outputs_value(fake_backend) -> None:
    result = runner.invoke(
        app,
        [
            "choice",
            "Route?",
            "--option",
            "sales",
            "--option",
            "billing",
            "--state",
            "help",
            "--field",
            "answers.q1.choice",
        ],
    )
    assert result.exit_code == 0
    assert result.stdout.strip() == "billing"


def test_fail_under_gate_fails(fake_backend, monkeypatch) -> None:
    # Use a score result below threshold.
    from bruv.domain.results import ScoreAnswer

    def evaluate(self, request: object) -> DecisionResult:
        return DecisionResult(
            backend="simple-jev",
            model="Qwen/Qwen3.5-0.8B",
            calibrated=False,
            answers={
                "q1": ScoreAnswer(
                    type="score",
                    score=0.2,
                    confidence=0.5,
                    legend={"0": "low", "1": "high"},
                    probabilities={"0": 0.5, "1": 0.5},
                )
            },
        )

    monkeypatch.setattr(_FakeBackend, "evaluate", evaluate)
    result = runner.invoke(
        app,
        [
            "score",
            "Rate?",
            "--level",
            "low",
            "--level",
            "high",
            "--state",
            "ctx",
            "--fail-under",
            "0.5",
            "--field",
            "answers.q1.score",
        ],
    )
    assert result.exit_code == 10


def test_abstain_band_exits_eleven(fake_backend, monkeypatch) -> None:
    from bruv.domain.results import ScoreAnswer

    def evaluate(self, request: object) -> DecisionResult:
        return DecisionResult(
            backend="simple-jev",
            model="Qwen/Qwen3.5-0.8B",
            calibrated=False,
            answers={
                "q1": ScoreAnswer(
                    type="score",
                    score=0.5,
                    confidence=0.5,
                    legend={"0": "low", "1": "high"},
                    probabilities={"0": 0.5, "1": 0.5},
                )
            },
        )

    monkeypatch.setattr(_FakeBackend, "evaluate", evaluate)
    result = runner.invoke(
        app,
        [
            "score",
            "Rate?",
            "--level",
            "low",
            "--level",
            "high",
            "--state",
            "ctx",
            "--abstain-band",
            "0.4:0.6",
            "--field",
            "answers.q1.score",
        ],
    )
    assert result.exit_code == 11
