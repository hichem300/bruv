"""Integration tests for CLI error handling and paid-request language."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from bruv.application import AuthenticationError, ProviderResponseError
from bruv.cli import app
from bruv.domain.results import DecisionResult
from bruv.domain.validation import BackendCapabilities

runner = CliRunner()


class _ErrorBackend:
    capabilities = BackendCapabilities(
        backend="typesafe",
        question_types=frozenset({"noul", "choice", "score"}),
        calibrated=True,
        allows_json_state=True,
    )

    def __init__(self, error: Exception) -> None:
        self._error = error

    def evaluate(self, request: object) -> DecisionResult:
        raise self._error


@pytest.fixture
def error_backend(monkeypatch):
    holder: dict[str, Exception] = {}

    def _create_backend(config, credentials, *, client_factory=None):
        return _ErrorBackend(holder["error"])

    monkeypatch.setattr("bruv.cli.create_backend", _create_backend)
    return holder


def test_authentication_error_exit_three_json(error_backend) -> None:
    error_backend["error"] = AuthenticationError(
        message="no key", paid_request=False, action="set key"
    )
    result = runner.invoke(
        app,
        ["noul", "Yes?", "--state", "ctx", "--output", "json"],
    )
    assert result.exit_code == 3
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["error"]["paid_request"] is False


def test_provider_response_error_exit_five_json(error_backend) -> None:
    error_backend["error"] = ProviderResponseError(message="bad", paid_request=True, action="retry")
    result = runner.invoke(
        app,
        ["noul", "Yes?", "--state", "ctx", "--output", "json"],
    )
    assert result.exit_code == 5
    payload = json.loads(result.stdout)
    assert payload["error"]["paid_request"] is True


def test_missing_non_interactive_input_exits_two() -> None:
    result = runner.invoke(app, ["noul", "Yes?"])
    assert result.exit_code == 2


def test_invalid_choice_exits_two() -> None:
    result = runner.invoke(app, ["choice", "Route?", "--option", "only-one", "--state", "ctx"])
    assert result.exit_code == 2


def test_human_error_paid_request_language(error_backend) -> None:
    error_backend["error"] = ProviderResponseError(message="bad", paid_request=True, action="retry")
    result = runner.invoke(app, ["noul", "Yes?", "--state", "ctx"])
    assert result.exit_code == 5
    assert "provider_response_error" in result.output
