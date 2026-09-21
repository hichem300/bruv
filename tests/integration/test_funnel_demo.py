"""Integration tests for the funnel audit demo."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from bruv.cli import app
from bruv.domain.validation import BackendCapabilities

runner = CliRunner()


def test_default_demo_is_dry_run_zero_calls(tmp_path: Path, monkeypatch) -> None:
    calls = {"count": 0}

    def _create_backend(config, credentials, *, client_factory=None):
        calls["count"] += 1
        raise AssertionError("dry run must not create a backend")

    monkeypatch.setattr("bruv.cli.create_backend", _create_backend)
    result = runner.invoke(app, ["demo", "funnel-audit", "--backend", "simple-jev"])
    assert result.exit_code == 0
    assert calls["count"] == 0
    payload = json.loads(result.stdout)
    assert payload["dry_run"] is True


def test_unknown_demo_exits_two() -> None:
    result = runner.invoke(app, ["demo", "bogus"])
    assert result.exit_code == 2


def test_copy_to_copies_without_executing(tmp_path: Path) -> None:
    dest = tmp_path / "copy"
    result = runner.invoke(app, ["demo", "funnel-audit", "--copy-to", str(dest)])
    assert result.exit_code == 0
    assert (dest / "audit.yaml").is_file()
    assert (dest / "sample-funnel.md").is_file()


def test_execute_runs_backend(tmp_path: Path, monkeypatch) -> None:
    from bruv.domain.results import ChoiceAnswer, DecisionResult

    def _create_backend(config, credentials, *, client_factory=None):
        class B:
            capabilities = BackendCapabilities(
                backend="simple-jev",
                question_types=frozenset({"noul", "choice", "score"}),
                calibrated=False,
                allows_json_state=True,
            )

            def evaluate(self, request: object) -> DecisionResult:
                return DecisionResult(
                    backend="simple-jev",
                    model="Qwen/Qwen3.5-0.8B",
                    calibrated=False,
                    answers={
                        "offer_understandable": ChoiceAnswer(
                            type="choice",
                            choice="yes",
                            confidence=0.9,
                            probabilities={"yes": 0.9, "no": 0.1},
                        )
                    },
                )

        return B()

    monkeypatch.setattr("bruv.cli.create_backend", _create_backend)
    result = runner.invoke(
        app, ["demo", "funnel-audit", "--backend", "simple-jev", "--execute", "--output", "json"]
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["calibrated"] is False


def test_demo_request_is_valid() -> None:
    from bruv.demos.funnel_audit import load_demo_request

    request = load_demo_request()
    assert set(request.questions) == {
        "offer_understandable",
        "cta_visible_specific",
        "proof_supports_claim",
        "obvious_friction",
        "highest_priority_leak",
        "severity",
    }
