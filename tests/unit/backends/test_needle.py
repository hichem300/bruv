"""Focused tests for Needle runtime safety and adapter constraints."""

from __future__ import annotations

import math
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest

from bruv.application import (
    ApplicationError,
    BackendUnavailableError,
    ConfigurationError,
    ProviderResponseError,
)
from bruv.backends.needle import CactusNeedleRuntime, NeedleAdapter, NeedleSelection
from bruv.domain.questions import ChoiceQuestion, NoulQuestion, ScoreQuestion
from bruv.domain.requests import DecisionRequest
from bruv.domain.results import ChoiceAnswer, NoulAnswer, ScoreAnswer
from bruv.domain.validation import BackendCapabilities

_BASE_MODEL = "Cactus-Compute/needle3"
_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {"value": {"type": "boolean"}},
    "required": ["value"],
}


def _response(value: bool = True) -> dict[str, Any]:
    return {
        "type": "call",
        "success": True,
        "function_calls": [{"name": "select_answer", "arguments": {"value": value}}],
        "suppressed_calls": [],
        "confidence": 0.8,
    }


def test_runtime_cleanup_does_not_mask_inference_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class Agent:
        def __init__(self) -> None:
            self.reset_calls = 0
            self.close_calls = 0

        def reset(self) -> None:
            self.reset_calls += 1
            if self.reset_calls > 1:
                raise RuntimeError("cleanup reset failed")

        def complete(self, _text: str) -> dict[str, Any]:
            raise RuntimeError("private inference failure")

        def close(self) -> None:
            self.close_calls += 1
            raise RuntimeError("cleanup close failed")

    agent = Agent()
    monkeypatch.setattr(
        "bruv.backends.needle.importlib.import_module",
        lambda _name: SimpleNamespace(Needle=lambda **_kwargs: agent),
    )

    runtime = CactusNeedleRuntime()
    with pytest.raises(BackendUnavailableError, match="inference could not complete"):
        runtime.classify(text="private", schema=_SCHEMA, description="private")

    assert agent.reset_calls == 2
    assert agent.close_calls == 1
    assert not hasattr(runtime, "_agents")


def test_runtime_serializes_native_access_across_instances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counter_lock = threading.Lock()
    active = 0
    max_active = 0

    class Agent:
        def reset(self) -> None:
            pass

        def complete(self, _text: str) -> dict[str, Any]:
            nonlocal active, max_active
            with counter_lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.03)
            with counter_lock:
                active -= 1
            return _response()

        def close(self) -> None:
            pass

    module = SimpleNamespace(Needle=lambda **_kwargs: Agent())
    monkeypatch.setattr("bruv.backends.needle.importlib.import_module", lambda _name: module)
    runtimes = [CactusNeedleRuntime(), CactusNeedleRuntime()]
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def classify(runtime: CactusNeedleRuntime) -> None:
        try:
            barrier.wait()
            runtime.classify(text="x", schema=_SCHEMA, description="x")
        except BaseException as exc:  # pragma: no cover - assertion reports thread failures
            errors.append(exc)

    threads = [threading.Thread(target=classify, args=(runtime,)) for runtime in runtimes]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert max_active == 1


def test_adapter_rejects_non_base_models() -> None:
    runtime = SimpleNamespace(classify=lambda **_kwargs: NeedleSelection(True, 0.8))
    with pytest.raises(ConfigurationError) as configured:
        NeedleAdapter(runtime, model="custom")
    assert configured.value.paid_request is False

    adapter = NeedleAdapter(runtime)
    request = DecisionRequest(
        state="x",
        model="custom",
        questions={"q": NoulQuestion(instructions="Decide")},
    )
    with pytest.raises(ConfigurationError) as requested:
        adapter.evaluate(request)
    assert requested.value.paid_request is False


def test_response_rejects_contradictory_envelopes() -> None:
    simultaneous = _response()
    simultaneous["suppressed_calls"] = [{"name": "select_answer", "arguments": {"value": False}}]
    wrong_type = _response()
    wrong_type["type"] = "respond"

    for response in (simultaneous, wrong_type):
        with pytest.raises(ProviderResponseError) as caught:
            CactusNeedleRuntime._parse_response(response)
        assert caught.value.paid_request is False


def test_response_accepts_documented_empty_refusal() -> None:
    selection = CactusNeedleRuntime._parse_response(
        {
            "type": "respond",
            "success": True,
            "function_calls": [],
            "suppressed_calls": [],
            "confidence": None,
        }
    )
    assert selection == NeedleSelection(value=None, confidence=None)


def _adapter(selection: NeedleSelection) -> tuple[NeedleAdapter, list[dict[str, object]]]:
    calls: list[dict[str, object]] = []

    def classify(**kwargs: object) -> NeedleSelection:
        calls.append(kwargs)
        return selection

    return NeedleAdapter(SimpleNamespace(classify=classify)), calls


def test_noul_contract_maps_boolean_schema_and_selection_answer() -> None:
    selection = NeedleSelection(value=True, confidence=0.9)
    adapter, calls = _adapter(selection)
    result = adapter.evaluate(
        DecisionRequest(
            state="ctx",
            questions={
                "q": NoulQuestion(
                    instructions="Decide refund",
                    criteria={"true": "mentions refund", "false": "no refund"},
                )
            },
        )
    )

    call = calls[0]
    schema = call["schema"]
    assert isinstance(schema, dict)
    assert schema["properties"] == {
        "value": {
            "type": "boolean",
            "description": "Boolean decision selected from the supplied state.",
        }
    }
    assert schema["required"] == ["value"]
    assert schema["additionalProperties"] is False
    description = call["description"]
    assert isinstance(description, str)
    assert "Decide refund" in description
    assert 'Return one boolean decision' in description
    assert '"false":"no refund"' in description

    answer = result.answers["q"]
    assert isinstance(answer, NoulAnswer)
    assert answer.value is True
    assert answer.confidence == pytest.approx(0.9)
    assert answer.noul is None


def test_choice_schema_preserves_criteria_order() -> None:
    criteria = {"alpha": 1, "beta": 2, "gamma": 3}
    adapter, calls = _adapter(NeedleSelection(value="beta", confidence=0.7))
    result = adapter.evaluate(
        DecisionRequest(
            state="ctx",
            questions={"q": ChoiceQuestion(instructions="Pick", criteria=dict(criteria))},
        )
    )

    schema = calls[0]["schema"]
    assert isinstance(schema, dict)
    assert schema["properties"]["value"]["enum"] == list(criteria)
    answer = result.answers["q"]
    assert isinstance(answer, ChoiceAnswer)
    assert answer.choice == "beta"
    assert answer.confidence == pytest.approx(0.7)
    assert answer.probabilities is None


def test_choice_label_outside_schema_rejected() -> None:
    adapter, _ = _adapter(NeedleSelection(value="delta", confidence=0.7))
    request = DecisionRequest(
        state="ctx",
        questions={"q": ChoiceQuestion(instructions="Pick", criteria={"alpha": 1, "beta": 2})},
    )
    with pytest.raises(ProviderResponseError, match="did not match the question schema"):
        adapter.evaluate(request)


def test_score_selection_maps_index_to_score_and_legend() -> None:
    levels = ["poor", "fair", "good"]
    adapter, calls = _adapter(NeedleSelection(value="1", confidence=0.6))
    result = adapter.evaluate(
        DecisionRequest(
            state="ctx",
            questions={"q": ScoreQuestion(instructions="Rate", criteria=list(levels))},
        )
    )

    schema = calls[0]["schema"]
    assert isinstance(schema, dict)
    assert schema["properties"]["value"]["enum"] == ["0", "1", "2"]
    description = calls[0]["description"]
    assert isinstance(description, str)
    assert '"0":"poor"' in description

    answer = result.answers["q"]
    assert isinstance(answer, ScoreAnswer)
    assert answer.score == pytest.approx(1.0)
    assert answer.legend == {"0": "poor", "1": "fair", "2": "good"}


@pytest.mark.parametrize(
    ("question", "value"),
    [
        (NoulQuestion(instructions="Decide"), "yes"),
        (ChoiceQuestion(instructions="Pick", criteria={"a": 1, "b": 2}), True),
        (ScoreQuestion(instructions="Rate", criteria=["low", "high"]), 0),
        (ScoreQuestion(instructions="Rate", criteria=["low", "high"]), "9"),
    ],
)
def test_selection_value_of_wrong_type_or_outside_schema_rejected(
    question: NoulQuestion | ChoiceQuestion | ScoreQuestion, value: object
) -> None:
    adapter, _ = _adapter(NeedleSelection(value=value, confidence=0.8))
    with pytest.raises(ProviderResponseError, match="did not match the question schema"):
        adapter.evaluate(DecisionRequest(state="ctx", questions={"q": question}))


@pytest.mark.parametrize(
    "confidence",
    [-0.1, 1.5, math.inf, math.nan, "0.9", True, None],
)
def test_invalid_confidence_rejected(confidence: object) -> None:
    adapter, _ = _adapter(NeedleSelection(value=True, confidence=confidence))
    with pytest.raises(ProviderResponseError, match="invalid selection confidence"):
        adapter.evaluate(
            DecisionRequest(state="ctx", questions={"q": NoulQuestion(instructions="Decide")})
        )


@pytest.mark.parametrize(
    "selection,match",
    [
        (NeedleSelection(value=None, confidence=0.5), "declined to answer"),
        (NeedleSelection(value=None, confidence=0.5, suppressed=True), "withheld the answer"),
    ],
)
def test_empty_and_suppressed_selections_rejected(selection: NeedleSelection, match: str) -> None:
    adapter, _ = _adapter(selection)
    with pytest.raises(ProviderResponseError, match=match):
        adapter.evaluate(
            DecisionRequest(state="ctx", questions={"q": NoulQuestion(instructions="Decide")})
        )


def _call(arguments: object) -> dict[str, object]:
    return {
        "success": True,
        "type": "call",
        "function_calls": [{"name": "select_answer", "arguments": arguments}],
        "suppressed_calls": [],
    }


def _malformed_cases() -> list[tuple[dict[str, object], str]]:
    return [
        (
            {"type": "call", "function_calls": [], "suppressed_calls": []},
            "omitted its success status",
        ),
        ({"success": True, "type": "tool_call"}, "invalid turn type"),
        (
            {"success": True, "type": "call", "function_calls": None, "suppressed_calls": []},
            "omitted its call lists",
        ),
        (
            {"success": True, "type": "call", "function_calls": [], "suppressed_calls": "x"},
            "omitted its call lists",
        ),
        (
            {
                "success": True,
                "type": "respond",
                "function_calls": [],
                "suppressed_calls": [{"name": "select_answer"}],
            },
            "turn type contradicted",
        ),
        (
            {"success": True, "type": "call", "function_calls": [{}, {}], "suppressed_calls": []},
            "unexpected number of selections",
        ),
        (
            {"success": True, "type": "call", "function_calls": ["call"], "suppressed_calls": []},
            "unexpected selection call",
        ),
        (
            {
                "success": True,
                "type": "call",
                "function_calls": [{"name": "other_tool"}],
                "suppressed_calls": [],
            },
            "unexpected selection call",
        ),
        (
            {
                "success": True,
                "type": "call",
                "function_calls": [{"name": "select_answer"}],
                "suppressed_calls": [],
            },
            "arguments were malformed",
        ),
        (_call({}), "arguments were malformed"),
        (_call({"value": 1, "extra": 2}), "arguments were malformed"),
    ]


@pytest.mark.parametrize(
    "response,match",
    _malformed_cases(),
)
def test_malformed_runtime_responses_rejected(response: dict[str, object], match: str) -> None:
    with pytest.raises(ProviderResponseError, match=match):
        CactusNeedleRuntime._parse_response(response)


def test_response_success_false_maps_to_backend_unavailable() -> None:
    with pytest.raises(BackendUnavailableError, match="runtime failure"):
        CactusNeedleRuntime._parse_response({"success": False, "type": "call"})


def test_result_metadata_reports_no_probabilities_and_no_usage() -> None:
    adapter, _ = _adapter(NeedleSelection(value=True, confidence=0.8))
    result = adapter.evaluate(
        DecisionRequest(state="ctx", questions={"q": NoulQuestion(instructions="Decide")})
    )

    assert result.backend == "needle"
    assert result.model == _BASE_MODEL
    assert result.calibrated is True
    assert result.usage is None
    assert result.provider_metadata == {
        "confidence_scope": "whole_response",
        "probabilities_available": False,
    }


def test_dict_state_serialized_deterministically() -> None:
    adapter, calls = _adapter(NeedleSelection(value=True, confidence=0.8))
    adapter.evaluate(
        DecisionRequest(
            state={"b": 1, "a": [2, {"z": 3, "y": 4}]},
            questions={"q": NoulQuestion(instructions="Decide")},
        )
    )
    assert calls[0]["text"] == '{"a":[2,{"y":4,"z":3}],"b":1}'


def test_unsupported_request_model_rejected_before_runtime() -> None:
    adapter, calls = _adapter(NeedleSelection(value=True, confidence=0.8))
    request = DecisionRequest(
        state="ctx", model="custom", questions={"q": NoulQuestion(instructions="Decide")}
    )
    with pytest.raises(ConfigurationError, match="base Needle 3 model"):
        adapter.evaluate(request)
    assert calls == []


def test_application_errors_pass_through_unexpected_ones_are_wrapped() -> None:
    provider_error = ProviderResponseError(
        message="bad selection", paid_request=False, action="retry"
    )

    raising = SimpleNamespace(classify=lambda **_kwargs: (_ for _ in ()).throw(provider_error))
    with pytest.raises(ApplicationError) as passed:
        NeedleAdapter(raising).evaluate(
            DecisionRequest(state="ctx", questions={"q": NoulQuestion(instructions="Decide")})
        )
    assert passed.value is provider_error

    def boom(**_kwargs: object) -> NeedleSelection:
        raise ValueError("boom")

    unexpected = SimpleNamespace(classify=boom)
    with pytest.raises(BackendUnavailableError, match="failed before returning a selection"):
        NeedleAdapter(unexpected).evaluate(
            DecisionRequest(state="ctx", questions={"q": NoulQuestion(instructions="Decide")})
        )


def test_runtime_returning_wrong_selection_type_rejected() -> None:
    adapter = NeedleAdapter(SimpleNamespace(classify=lambda **_kwargs: object()))
    with pytest.raises(ProviderResponseError, match="invalid selection"):
        adapter.evaluate(
            DecisionRequest(state="ctx", questions={"q": NoulQuestion(instructions="Decide")})
        )


def test_telemetry_disabled_before_optional_import(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NEEDLE_TELEMETRY", raising=False)
    seen: dict[str, str | None] = {}

    def fake_import(name: str) -> SimpleNamespace:
        seen["env"] = __import__("os").environ.get("NEEDLE_TELEMETRY")
        seen["name"] = name
        return SimpleNamespace()

    monkeypatch.setattr("bruv.backends.needle.importlib.import_module", fake_import)
    CactusNeedleRuntime()

    assert seen == {"name": "needle", "env": "0"}


def test_optional_needle_module_not_imported_on_module_load() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import bruv.backends.needle;"
            " print('needle' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.stdout.strip() == "False"


def test_backend_capabilities_accept_registry_names_but_reject_blank() -> None:
    capabilities = BackendCapabilities(
        backend="future-provider",
        question_types=frozenset({"noul"}),
        calibrated=False,
        allows_json_state=False,
    )
    assert capabilities.backend == "future-provider"
    with pytest.raises(ValueError, match="nonblank"):
        BackendCapabilities(
            backend=" ",
            question_types=frozenset(),
            calibrated=False,
            allows_json_state=False,
        )
