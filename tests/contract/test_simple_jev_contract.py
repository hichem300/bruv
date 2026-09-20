"""Simple Jev backend contract: canonical mapping and calibration."""

from __future__ import annotations

import httpx
import pytest

from bruv.backends.simple_jev import SimpleJevAdapter
from bruv.domain.results import ChoiceAnswer, NoulAnswer, ScoreAnswer
from tests.contract._shared import mixed_request, mixed_response
from tests.contract.backend_contract import assert_backend_contract


def _client(handler: httpx.MockTransport) -> httpx.Client:
    return httpx.Client(transport=handler, follow_redirects=False, verify=True)


def test_simple_jev_satisfies_backend_contract() -> None:
    handler = httpx.MockTransport(lambda request: httpx.Response(200, json=mixed_response()))
    backend = SimpleJevAdapter(
        base_url="http://127.0.0.1:8000",
        model="Qwen/Qwen3.5-0.8B",
        client=_client(handler),
    )
    result = assert_backend_contract(backend, mixed_request())
    assert result.calibrated is False
    assert result.backend == "simple-jev"
    assert isinstance(result.answers["route"], ChoiceAnswer)
    assert isinstance(result.answers["urgency"], ScoreAnswer)
    assert isinstance(result.answers["asks_for_refund"], NoulAnswer)


def test_simple_jev_always_reports_uncalibrated() -> None:
    response = mixed_response()
    response["calibrated"] = True  # provider must not influence canonical marker
    handler = httpx.MockTransport(lambda request: httpx.Response(200, json=response))
    backend = SimpleJevAdapter(
        base_url="http://127.0.0.1:8000",
        model="Qwen/Qwen3.5-0.8B",
        client=_client(handler),
    )
    result = backend.evaluate(mixed_request())
    assert result.calibrated is False


def test_simple_jev_posts_classifier_payload_with_correct_headers() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["method"] = request.method
        captured["content_type"] = request.headers.get("content-type")
        captured["body"] = request.read().decode()
        return httpx.Response(200, json=mixed_response())

    backend = SimpleJevAdapter(
        base_url="http://127.0.0.1:8000/",
        model="Qwen/Qwen3.5-0.8B",
        client=_client(httpx.MockTransport(handler)),
    )
    backend.evaluate(mixed_request())
    assert captured["method"] == "POST"
    assert captured["url"] == "http://127.0.0.1:8000/v1/classifier"
    assert captured["content_type"] == "application/json"


def test_simple_jev_redirects_not_followed() -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(302, headers={"Location": "http://127.0.0.1:8000/v1/classifier"})

    backend = SimpleJevAdapter(
        base_url="http://127.0.0.1:8000",
        model="Qwen/Qwen3.5-0.8B",
        client=_client(httpx.MockTransport(handler)),
    )
    with pytest.raises(Exception):  # noqa: B017
        backend.evaluate(mixed_request())
    assert calls["count"] == 1


def test_simple_jev_timeout_is_backend_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out")

    from bruv.application import BackendUnavailableError

    backend = SimpleJevAdapter(
        base_url="http://127.0.0.1:8000",
        model="Qwen/Qwen3.5-0.8B",
        client=_client(httpx.MockTransport(handler)),
    )
    with pytest.raises(BackendUnavailableError) as exc_info:
        backend.evaluate(mixed_request())
    assert exc_info.value.paid_request is False


def test_simple_jev_malformed_response_is_paid_provider_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json")

    from bruv.application import ProviderResponseError

    backend = SimpleJevAdapter(
        base_url="http://127.0.0.1:8000",
        model="Qwen/Qwen3.5-0.8B",
        client=_client(httpx.MockTransport(handler)),
    )
    with pytest.raises(ProviderResponseError) as exc_info:
        backend.evaluate(mixed_request())
    assert exc_info.value.paid_request is True


def test_simple_jev_errors_do_not_leak_request_body_values() -> None:
    secret = "bruv_test_secret_DO_NOT_PRINT"  # noqa: S105

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=secret.encode())

    backend = SimpleJevAdapter(
        base_url="http://127.0.0.1:8000",
        model="Qwen/Qwen3.5-0.8B",
        client=_client(httpx.MockTransport(handler)),
    )
    try:
        backend.evaluate(mixed_request())
    except Exception as exc:  # noqa: BLE001
        assert secret not in str(exc)
    else:
        pytest.fail("expected an error")
