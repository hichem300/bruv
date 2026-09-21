"""RLCD CLI contract tests: laziness, fake-backend rendering, gates, config pins.

No real model, download, or network is ever touched. Optional RLCD runtime
packages (numpy, onnxruntime, tokenizers, huggingface_hub) are never imported
in any path exercised here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from bruv.backends.registry import backend_capabilities
from bruv.backends.rlcd_artifacts import REPO_ID, REVISION
from bruv.backends.rlcd_calibration import CALIBRATOR_SCOPE, UPSTREAM_ABSTAIN_SENTINEL
from bruv.cli import app
from bruv.config import AppConfig
from bruv.domain.requests import DecisionRequest
from bruv.domain.results import (
    ABSTAIN_ANSWER_ID,
    AbstainAnswer,
    ChoiceAnswer,
    DecisionResult,
    NoulAnswer,
    ScoreAnswer,
)

runner = CliRunner()

RLCD_BACKEND = "rlcd-modernbert"
OPTIONAL_PACKAGES = ("numpy", "onnxruntime", "tokenizers", "huggingface_hub")


@pytest.fixture
def rlcd_lazy_import_guard(monkeypatch: pytest.MonkeyPatch):
    """Fail loudly if any optional RLCD runtime package gets imported."""
    return _install_lazy_import_guard(
        monkeypatch, OPTIONAL_PACKAGES
    )


def _install_lazy_import_guard(
    monkeypatch: pytest.MonkeyPatch, packages: tuple[str, ...]
):
    original_import = __import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        for package in packages:
            if name == package or name.startswith(f"{package}."):
                raise AssertionError(f"optional package imported: {package}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded_import)
    return None


class _FakeRlcdBackend:
    capabilities = backend_capabilities(RLCD_BACKEND)

    def __init__(self, result: DecisionResult) -> None:
        self._result = result
        self.requests: list[DecisionRequest] = []

    def evaluate(self, request: DecisionRequest) -> DecisionResult:
        self.requests.append(request)
        return self._result


def _install_fake_backend(
    monkeypatch: pytest.MonkeyPatch, result: DecisionResult
) -> _FakeRlcdBackend:
    backend = _FakeRlcdBackend(result)

    def _create_backend(config: object, credentials: object, *, client_factory: object = None):
        return backend

    monkeypatch.setattr("bruv.cli.create_backend", _create_backend)
    return backend


def _rlcd_provider_metadata(
    *,
    k: int,
    full_distribution: dict[str, float],
    expected_value: float | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "revision": REVISION,
        "calibration_scope": CALIBRATOR_SCOPE,
        "temperature_path": f"per_k:{k}",
        "probability_base": "conditional_on_sufficient_evidence",
        "full_distribution": full_distribution,
    }
    if expected_value is not None:
        payload["expected_value_over_substantive_mass"] = expected_value
    return {"q1": {"rlcd": payload}}


def _run_json(args: list[str]) -> tuple[int, dict[str, Any]]:
    result = runner.invoke(app, args)
    assert result.exception is None, repr(result.exception)
    return result.exit_code, json.loads(result.stdout)


# --- Laziness: no optional import, no runtime construction, no download -------


def test_rlcd_dry_run_never_imports_optional_packages(rlcd_lazy_import_guard) -> None:
    code, payload = _run_json(
        ["noul", "Refund?", "--state", "ctx", "--backend", RLCD_BACKEND, "--dry-run"]
    )
    assert code == 0
    assert payload["backend"] == RLCD_BACKEND
    assert payload["dry_run"] is True


def test_rlcd_spec_help_and_doctor_stay_lazy(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    # Doctor's rlcd_cache check deliberately probes huggingface_hub availability;
    # everything else must stay free of optional imports.
    _install_lazy_import_guard(monkeypatch, ("numpy", "onnxruntime", "tokenizers"))
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0

    spec = runner.invoke(app, ["spec", "--output", "json"])
    assert spec.exit_code == 0
    assert RLCD_BACKEND in json.loads(spec.stdout)["backends"]

    monkeypatch.setattr(
        "bruv.onboarding.doctor.load_config", lambda **_: AppConfig(backend=RLCD_BACKEND)
    )
    doctor = runner.invoke(app, ["doctor"])
    assert isinstance(doctor.exception, SystemExit)
    assert doctor.exit_code in (0, 1)
    assert "rlcd_cache" in doctor.stdout


# --- Fake RLCD backend through the real CLI/evaluation/rendering path --------


def test_rlcd_noul_probability_json(
    rlcd_lazy_import_guard, monkeypatch: pytest.MonkeyPatch
) -> None:
    full = {"true": 0.60, "false": 0.25, ABSTAIN_ANSWER_ID: 0.15}
    result = DecisionResult(
        backend=RLCD_BACKEND,
        model=REPO_ID,
        calibrated=True,
        answers={"q1": NoulAnswer(noul=0.60 / 0.85)},
        provider_metadata=_rlcd_provider_metadata(k=3, full_distribution=full),
    )
    backend = _install_fake_backend(monkeypatch, result)
    code, payload = _run_json(
        ["noul", "Refund?", "--state", "ctx", "--backend", RLCD_BACKEND, "--output", "json"]
    )
    assert code == 0
    assert backend.requests[0].model is None
    assert payload["backend"] == RLCD_BACKEND
    assert payload["model"] == REPO_ID
    assert payload["calibrated"] is True
    assert payload["answers"]["q1"] == {"type": "noul", "noul": pytest.approx(0.60 / 0.85)}
    assert "probabilities" not in payload["answers"]["q1"]
    rlcd = payload["provider_metadata"]["q1"]["rlcd"]
    assert rlcd["revision"] == REVISION
    assert rlcd["temperature_path"] == "per_k:3"
    assert rlcd["probability_base"] == "conditional_on_sufficient_evidence"
    assert set(rlcd["full_distribution"]) == {"true", "false", ABSTAIN_ANSWER_ID}
    assert UPSTREAM_ABSTAIN_SENTINEL not in json.dumps(payload)


def test_rlcd_choice_conditional_probabilities_json(
    rlcd_lazy_import_guard, monkeypatch: pytest.MonkeyPatch
) -> None:
    full = {"a": 0.20, "b": 0.70, ABSTAIN_ANSWER_ID: 0.10}
    result = DecisionResult(
        backend=RLCD_BACKEND,
        model=REPO_ID,
        calibrated=True,
        answers={
            "q1": ChoiceAnswer(
                choice="b", confidence=0.7, probabilities={"a": 0.3, "b": 0.7}
            )
        },
        provider_metadata=_rlcd_provider_metadata(k=3, full_distribution=full),
    )
    _install_fake_backend(monkeypatch, result)
    code, payload = _run_json(
        [
            "choice",
            "Route?",
            "--option",
            "a",
            "--option",
            "b",
            "--state",
            "ctx",
            "--backend",
            RLCD_BACKEND,
            "--output",
            "json",
        ]
    )
    assert code == 0
    answer = payload["answers"]["q1"]
    assert answer == {
        "type": "choice",
        "choice": "b",
        "confidence": 0.7,
        "probabilities": {"a": 0.3, "b": 0.7},
    }
    assert ABSTAIN_ANSWER_ID not in answer["probabilities"]
    assert answer["probabilities"] != full
    assert payload["provider_metadata"]["q1"]["rlcd"]["full_distribution"] == full


def test_rlcd_score_exact_index_and_legend(
    rlcd_lazy_import_guard, monkeypatch: pytest.MonkeyPatch
) -> None:
    full = {"0": 0.5, "1": 0.3, "2": 0.1, ABSTAIN_ANSWER_ID: 0.1}
    result = DecisionResult(
        backend=RLCD_BACKEND,
        model=REPO_ID,
        calibrated=True,
        answers={
            "q1": ScoreAnswer(
                score=2.0,
                confidence=0.5,
                legend={"0": "low", "1": "mid", "2": "high"},
                probabilities={"0": 0.5, "1": 0.3, "2": 0.2},
            )
        },
        provider_metadata=_rlcd_provider_metadata(
            k=4, full_distribution=full, expected_value=1.1
        ),
    )
    _install_fake_backend(monkeypatch, result)
    code, payload = _run_json(
        [
            "score",
            "Rate?",
            "--level",
            "low",
            "--level",
            "mid",
            "--level",
            "high",
            "--state",
            "ctx",
            "--backend",
            RLCD_BACKEND,
            "--output",
            "json",
        ]
    )
    assert code == 0
    answer = payload["answers"]["q1"]
    assert answer["score"] == 2.0
    assert answer["legend"] == {"0": "low", "1": "mid", "2": "high"}
    assert list(answer["probabilities"]) == ["0", "1", "2"]
    rlcd = payload["provider_metadata"]["q1"]["rlcd"]
    assert rlcd["temperature_path"] == "per_k:4"
    assert rlcd["expected_value_over_substantive_mass"] == 1.1


def test_rlcd_abstain_json_full_distribution(
    rlcd_lazy_import_guard, monkeypatch: pytest.MonkeyPatch
) -> None:
    full = {"0": 0.3, "1": 0.3, "2": 0.05, ABSTAIN_ANSWER_ID: 0.35}
    result = DecisionResult(
        backend=RLCD_BACKEND,
        model=REPO_ID,
        calibrated=True,
        answers={
            "q1": AbstainAnswer(
                reason="insufficient_evidence",
                source_question_type="score",
                confidence=0.35,
                probabilities=full,
                legend={
                    **{"0": "low", "1": "mid", "2": "high"},
                    ABSTAIN_ANSWER_ID: "insufficient evidence",
                },
            )
        },
        provider_metadata=_rlcd_provider_metadata(k=4, full_distribution=full),
    )
    _install_fake_backend(monkeypatch, result)
    code, payload = _run_json(
        [
            "score",
            "Rate?",
            "--level",
            "low",
            "--level",
            "mid",
            "--level",
            "high",
            "--state",
            "ctx",
            "--backend",
            RLCD_BACKEND,
            "--output",
            "json",
        ]
    )
    assert code == 0
    answer = payload["answers"]["q1"]
    assert answer["type"] == "abstain"
    assert answer["reason"] == "insufficient_evidence"
    assert answer["source_question_type"] == "score"
    assert answer["confidence"] == answer["probabilities"][ABSTAIN_ANSWER_ID]
    assert sum(answer["probabilities"].values()) == pytest.approx(1.0)
    assert set(answer["legend"]) == set(answer["probabilities"])
    assert payload["provider_metadata"]["q1"]["rlcd"]["full_distribution"] == full


# --- Gates over fake RLCD results --------------------------------------------


def test_rlcd_abstained_gate_exits_eleven(
    rlcd_lazy_import_guard, monkeypatch: pytest.MonkeyPatch
) -> None:
    full = {"0": 0.3, "1": 0.3, "2": 0.05, ABSTAIN_ANSWER_ID: 0.35}
    result = DecisionResult(
        backend=RLCD_BACKEND,
        model=REPO_ID,
        calibrated=True,
        answers={
            "q1": AbstainAnswer(
                reason="insufficient_evidence",
                source_question_type="score",
                confidence=0.35,
                probabilities=full,
            )
        },
    )
    _install_fake_backend(monkeypatch, result)
    outcome = runner.invoke(
        app,
        [
            "score",
            "Rate?",
            "--level",
            "low",
            "--level",
            "mid",
            "--level",
            "high",
            "--state",
            "ctx",
            "--backend",
            RLCD_BACKEND,
            "--output",
            "json",
            "--field",
            "answers.q1.confidence",
            "--fail-under",
            "0.9",
        ],
    )
    assert outcome.exit_code == 11


def test_rlcd_mixed_multi_question_keeps_sibling_and_normal_gate(
    rlcd_lazy_import_guard, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    eval_file = tmp_path / "eval.json"
    eval_file.write_text(
        json.dumps(
            {
                "state": "ctx",
                "questions": {
                    "q1": {
                        "type": "score",
                        "instructions": "Rate?",
                        "criteria": ["low", "mid", "high"],
                    },
                    "q2": {
                        "type": "choice",
                        "instructions": "Route?",
                        "criteria": {"a": None, "b": None},
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    result = DecisionResult(
        backend=RLCD_BACKEND,
        model=REPO_ID,
        calibrated=True,
        answers={
            "q1": ScoreAnswer(
                score=2.0,
                confidence=0.5,
                legend={"0": "low", "1": "mid", "2": "high"},
                probabilities={"0": 0.5, "1": 0.3, "2": 0.2},
            ),
            "q2": AbstainAnswer(
                reason="insufficient_evidence",
                source_question_type="choice",
                confidence=0.35,
                probabilities={"a": 0.3, "b": 0.35, ABSTAIN_ANSWER_ID: 0.35},
            ),
        },
    )
    _install_fake_backend(monkeypatch, result)
    outcome = runner.invoke(
        app,
        [
            "eval",
            str(eval_file),
            "--backend",
            RLCD_BACKEND,
            "--output",
            "json",
            "--fail-under",
            "0.5",
        ],
    )
    assert outcome.exit_code == 0  # sibling gate stays normal: no abstain exit
    assert "q1" in outcome.stdout and "q2" in outcome.stdout


# --- Config contract: file > defaults only; pins reject before any work -------


def test_rlcd_model_and_revision_come_from_file_only(
    rlcd_lazy_import_guard, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_file = tmp_path / "bruv.toml"
    config_file.write_text(
        'rlcd_model = "custom/model"\nrlcd_revision = "customrev"\n',
        encoding="utf-8",
    )
    from bruv.config import load_config

    config = load_config(env={}, path=config_file)
    assert config.rlcd_model == "custom/model"
    assert config.rlcd_revision == "customrev"

    default = load_config(env={}, path=tmp_path / "missing.toml")
    assert default.rlcd_model == REPO_ID
    assert default.rlcd_revision == REVISION

    env_config = load_config(
        env={"RLCD_MODEL": "env/model", "RLCD_REVISION": "envrev"},
        path=config_file,
    )
    assert env_config.rlcd_model == "custom/model"
    assert env_config.rlcd_revision == "customrev"


def test_pinned_backend_rejects_unpinned_config_model_before_artifacts(
    rlcd_lazy_import_guard, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from bruv.backends import rlcd_modernbert

    calls: list[str] = []

    def _no_artifacts(*args: object, **kwargs: object) -> dict[str, Path]:
        calls.append("ensure_artifacts")
        raise AssertionError("download attempted")

    monkeypatch.setattr(rlcd_modernbert, "ensure_artifacts", _no_artifacts)
    monkeypatch.setattr(
        "bruv.config.config_path", lambda: tmp_path / "bruv.toml", raising=False
    )
    (tmp_path / "bruv.toml").write_text('rlcd_model = "custom/model"\n', encoding="utf-8")

    outcome = runner.invoke(
        app,
        ["noul", "Refund?", "--state", "ctx", "--backend", RLCD_BACKEND, "--output", "json"],
    )
    assert outcome.exit_code == 2
    assert calls == []
    payload = json.loads(outcome.stdout)
    assert payload["error"]["code"] == "configuration_error"
    assert "pinned ModernBERT model revision" in payload["error"]["message"]


def test_adapter_rejects_unpinned_request_model_before_runtime(
    rlcd_lazy_import_guard,
) -> None:
    from bruv.backends.rlcd_calibration import RlcdCalibrator
    from bruv.backends.rlcd_modernbert import RlcdModernBertAdapter

    calibrator = RlcdCalibrator(
        model_id=REPO_ID,
        temperature=1.0,
        log_temperature=0.0,
        scope=CALIBRATOR_SCOPE,
        per_k={},
        artifact_hash="unused",
    )

    class _ExplodingRuntime:
        def run(self, input_ids: object, attention_mask: object) -> object:
            raise AssertionError("runtime must not run")

    class _ExplodingTokenizer:
        def encode_batch(self, texts: list[str]) -> tuple[object, object]:
            raise AssertionError("tokenizer must not run")

    adapter = RlcdModernBertAdapter(
        runtime=_ExplodingRuntime(),  # type: ignore[arg-type]
        tokenizer=_ExplodingTokenizer(),  # type: ignore[arg-type]
        calibrator=calibrator,
    )
    request = DecisionRequest.model_validate(
        {
            "state": "ctx",
            "model": "custom/model",
            "questions": {"q1": {"type": "noul", "instructions": "Refund?"}},
        }
    )
    from bruv.application import ConfigurationError

    with pytest.raises(ConfigurationError) as excinfo:
        adapter.evaluate(request)
    assert "pinned ModernBERT model revision" in excinfo.value.message
