"""Focused tests for Needle runtime safety and adapter constraints."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest

from bruv.application import BackendUnavailableError, ConfigurationError, ProviderResponseError
from bruv.backends.needle import CactusNeedleRuntime, NeedleAdapter, NeedleSelection
from bruv.domain.questions import NoulQuestion
from bruv.domain.requests import DecisionRequest
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
