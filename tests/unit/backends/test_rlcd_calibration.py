"""Unit tests for RLCD per-K calibration: loader, temperature lookup, softmax.

NumPy is imported lazily by the module under test and is installed via the
declared ``rlcd-modernbert`` extra, so every calibration test runs for real.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pytest

from bruv.application import ProviderResponseError
from bruv.backends.rlcd_calibration import (
    CALIBRATOR_SCOPE,
    CANONICAL_ABSTAIN_ID,
    UPSTREAM_ABSTAIN_SENTINEL,
    RlcdCalibrator,
    calibrated_distribution,
    load_calibrator,
    temperature_for,
)

CORRUPT_ACTION = "Re-download artifacts or reinstall 'bruv[rlcd-modernbert]'."

SUPPORTED_K = (2, 3, 4, 5, 6, 7, 9, 11, 17, 25)


def _valid_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "format_version": "rlcd-calibrator-v1",
        "model_id": "heman10x/rlcd-modernbert-151m",
        "temperature": 2.5,
        "log_temperature": 0.75,
        "scope": CALIBRATOR_SCOPE,
        "per_k": {str(k): 0.5 + 0.25 * k for k in SUPPORTED_K},
        "artifact_hash": "abc123",
    }
    payload.update(overrides)
    return payload


def _write(payload: Any, tmp_path: Path, name: str = "calibrator.json") -> Path:
    path = tmp_path / name
    if isinstance(payload, (bytes, bytearray)):
        path.write_bytes(payload)
    else:
        path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _calibrator_from(payload: Any, tmp_path: Path) -> RlcdCalibrator:
    return load_calibrator(_write(payload, tmp_path))


def _assert_unpaid(exc: pytest.ExceptionInfo[ProviderResponseError]) -> None:
    error = exc.value
    assert isinstance(error, ProviderResponseError)
    assert error.paid_request is False


# ---------------------------------------------------------------------------
# Loader: happy path
# ---------------------------------------------------------------------------


def test_load_calibrator_parses_every_field(tmp_path: Path) -> None:
    calibrator = _calibrator_from(_valid_payload(), tmp_path)

    assert calibrator.model_id == "heman10x/rlcd-modernbert-151m"
    assert calibrator.temperature == 2.5
    assert calibrator.log_temperature == 0.75
    assert calibrator.scope == CALIBRATOR_SCOPE
    assert calibrator.artifact_hash == "abc123"
    assert calibrator.per_k == {k: 0.5 + 0.25 * k for k in SUPPORTED_K}
    assert all(isinstance(key, int) for key in calibrator.per_k)


def test_loader_errors_are_sanitized_and_unpaid(tmp_path: Path) -> None:
    missing = tmp_path / "nope.json"
    with pytest.raises(ProviderResponseError) as missing_error:
        load_calibrator(missing)
    _assert_unpaid(missing_error)
    assert missing_error.value.action == CORRUPT_ACTION
    assert str(missing) not in missing_error.value.message

    corrupt = _write(b"{not json", tmp_path)
    with pytest.raises(ProviderResponseError) as corrupt_error:
        load_calibrator(corrupt)
    _assert_unpaid(corrupt_error)
    assert str(corrupt) not in corrupt_error.value.message

    bad_utf8 = _write(b'{"a": "\xff\xfe"}', tmp_path)
    with pytest.raises(ProviderResponseError) as utf8_error:
        load_calibrator(bad_utf8)
    _assert_unpaid(utf8_error)


def test_loader_rejects_non_object_root(tmp_path: Path) -> None:
    with pytest.raises(ProviderResponseError) as exc_info:
        _calibrator_from([1, 2, 3], tmp_path)
    _assert_unpaid(exc_info)
    assert "not a JSON object" in exc_info.value.message


def test_loader_rejects_wrong_format_version(tmp_path: Path) -> None:
    with pytest.raises(ProviderResponseError) as exc_info:
        _calibrator_from(_valid_payload(format_version="rlcd-calibrator-v2"), tmp_path)
    _assert_unpaid(exc_info)
    assert "schema is unsupported" in exc_info.value.message


def test_loader_names_first_missing_required_field(tmp_path: Path) -> None:
    for field in ("model_id", "temperature", "log_temperature", "scope", "per_k", "artifact_hash"):
        payload = _valid_payload()
        del payload[field]
        with pytest.raises(ProviderResponseError) as exc_info:
            _calibrator_from(payload, tmp_path)
        _assert_unpaid(exc_info)
        assert field in exc_info.value.message


def test_loader_rejects_wrong_scope(tmp_path: Path) -> None:
    with pytest.raises(ProviderResponseError) as exc_info:
        _calibrator_from(_valid_payload(scope="some_other_scope"), tmp_path)
    _assert_unpaid(exc_info)
    assert "not supported" in exc_info.value.message


def test_loader_rejects_blank_and_nontext_text_fields(tmp_path: Path) -> None:
    for field in ("model_id", "scope", "artifact_hash"):
        with pytest.raises(ProviderResponseError):
            _calibrator_from(_valid_payload(**{field: "   "}), tmp_path)
        with pytest.raises(ProviderResponseError):
            _calibrator_from(_valid_payload(**{field: 42}), tmp_path)


def test_loader_rejects_malformed_global_temperatures(tmp_path: Path) -> None:
    for value in (0.0, -1.0, "hot", None, True, float("nan")):
        with pytest.raises(ProviderResponseError) as exc_info:
            _calibrator_from(_valid_payload(temperature=value), tmp_path)
        _assert_unpaid(exc_info)
        assert "temperature" in exc_info.value.message

    for value in (float("inf"), "cold", None, True):
        with pytest.raises(ProviderResponseError):
            _calibrator_from(_valid_payload(log_temperature=value), tmp_path)


def test_loader_rejects_huge_int_temperature_sanitized(tmp_path: Path) -> None:
    with pytest.raises(ProviderResponseError) as exc_info:
        _calibrator_from(_valid_payload(temperature=10**400), tmp_path)
    _assert_unpaid(exc_info)
    # The error is the stable sanitized message: it names only the field and
    # never echoes any part of the rejected numeric literal.
    assert (
        exc_info.value.message
        == "RLCD calibrator field 'temperature' must be a finite positive number."
    )
    assert str(10**400)[:50] not in exc_info.value.message


def test_loader_rejects_malformed_per_k(tmp_path: Path) -> None:
    with pytest.raises(ProviderResponseError) as list_error:
        _calibrator_from(_valid_payload(per_k=[1, 2]), tmp_path)
    _assert_unpaid(list_error)
    assert "per_k" in list_error.value.message

    for key in ("03", " 3", "3.0", "+3", "three", ""):
        with pytest.raises(ProviderResponseError) as key_error:
            _calibrator_from(_valid_payload(per_k={key: 1.5}), tmp_path)
        _assert_unpaid(key_error)
        assert "integer string" in key_error.value.message

    for value in (0.0, -2.0, "hot", None, True, float("nan"), float("inf")):
        with pytest.raises(ProviderResponseError) as value_error:
            _calibrator_from(_valid_payload(per_k={"3": value}), tmp_path)
        _assert_unpaid(value_error)
        assert "per_k" in value_error.value.message


def test_loader_rejects_huge_int_per_k_value(tmp_path: Path) -> None:
    with pytest.raises(ProviderResponseError) as exc_info:
        _calibrator_from(_valid_payload(per_k={"3": 10**400}), tmp_path)
    _assert_unpaid(exc_info)
    assert exc_info.value.message == "RLCD calibrator per_k['3'] is not a finite number."
    assert str(10**400)[:50] not in exc_info.value.message


# ---------------------------------------------------------------------------
# temperature_for: per_k only, no global fallback
# ---------------------------------------------------------------------------


def test_temperature_for_returns_exact_per_k_entry() -> None:
    calibrator = RlcdCalibrator(
        model_id="m",
        temperature=999.0,
        log_temperature=888.0,
        scope=CALIBRATOR_SCOPE,
        per_k={3: 1.5},
        artifact_hash="h",
    )
    assert temperature_for(calibrator, 3) == 1.5


@pytest.mark.parametrize("k", SUPPORTED_K)
def test_missing_k_never_falls_back_to_global_temperature(k: int) -> None:
    calibrator = RlcdCalibrator(
        model_id="m",
        temperature=999.0,
        log_temperature=888.0,
        scope=CALIBRATOR_SCOPE,
        per_k={other: 1.0 for other in SUPPORTED_K if other != k},
        artifact_hash="h",
    )
    with pytest.raises(ProviderResponseError) as exc_info:
        temperature_for(calibrator, k)
    _assert_unpaid(exc_info)
    assert f"K={k}" in exc_info.value.message
    assert "no temperature" in exc_info.value.message


def test_empty_per_k_fails_for_every_supported_k() -> None:
    calibrator = RlcdCalibrator(
        model_id="m",
        temperature=1.0,
        log_temperature=1.0,
        scope=CALIBRATOR_SCOPE,
        per_k={},
        artifact_hash="h",
    )
    for k in SUPPORTED_K:
        with pytest.raises(ProviderResponseError):
            temperature_for(calibrator, k)


# ---------------------------------------------------------------------------
# calibrated_distribution: softmax math for every supported K
# ---------------------------------------------------------------------------


def _reference_softmax(logits: list[float], temperature: float) -> list[float]:
    scaled = [value / temperature for value in logits]
    peak = max(scaled)
    exps = [math.exp(value - peak) for value in scaled]
    total = math.fsum(exps)
    return [exp / total for exp in exps]


def _full_calibrator() -> RlcdCalibrator:
    return RlcdCalibrator(
        model_id="m",
        temperature=999.0,
        log_temperature=888.0,
        scope=CALIBRATOR_SCOPE,
        per_k={k: 0.5 + 0.25 * k for k in SUPPORTED_K},
        artifact_hash="h",
    )


def _logits(width: int = 30) -> list[float]:
    return [math.sin(index) * 2.0 for index in range(width)]


@pytest.mark.parametrize("k", SUPPORTED_K)
def test_softmax_matches_reference_for_every_supported_k(k: int) -> None:
    calibrator = _full_calibrator()
    row = _logits()
    probabilities, winner = calibrated_distribution(row, calibrator, k)

    expected = _reference_softmax(row[:k], calibrator.per_k[k])
    assert len(probabilities) == k
    assert winner == row[:k].index(max(row[:k]))
    for got, want in zip(probabilities, expected, strict=True):
        assert got == pytest.approx(want, rel=1e-12, abs=1e-15)
    # Exact second-pass renormalization tolerance.
    assert math.fsum(probabilities) == pytest.approx(1.0, abs=1e-12)
    assert all(p > 0.0 for p in probabilities)
    assert max(probabilities) == probabilities[winner]


@pytest.mark.parametrize("k", SUPPORTED_K)
def test_softmax_is_shift_invariant(k: int) -> None:
    calibrator = _full_calibrator()
    row = _logits()
    base, base_winner = calibrated_distribution(row, calibrator, k)
    shifted, shifted_winner = calibrated_distribution(
        [value + 1000.0 for value in row], calibrator, k
    )

    assert shifted_winner == base_winner
    for got, want in zip(shifted, base, strict=True):
        assert got == pytest.approx(want, rel=1e-12, abs=1e-15)


def test_temperature_changes_spread_but_keeps_normalization() -> None:
    row = _logits()[:3]
    hot = RlcdCalibrator("m", 1.0, 1.0, CALIBRATOR_SCOPE, {3: 10.0}, "h")
    cold = RlcdCalibrator("m", 1.0, 1.0, CALIBRATOR_SCOPE, {3: 0.1}, "h")

    hot_probs, _ = calibrated_distribution(row, hot, 3)
    cold_probs, _ = calibrated_distribution(row, cold, 3)

    assert math.fsum(hot_probs) == pytest.approx(1.0, abs=1e-12)
    assert math.fsum(cold_probs) == pytest.approx(1.0, abs=1e-12)
    assert max(cold_probs) > max(hot_probs)


def test_calibration_failures_are_unpaid_and_sanitized() -> None:
    calibrator = _full_calibrator()

    with pytest.raises(ProviderResponseError) as low_k:
        calibrated_distribution(_logits(), calibrator, 0)
    _assert_unpaid(low_k)

    with pytest.raises(ProviderResponseError) as short_row:
        calibrated_distribution([0.1, 0.2], calibrator, 3)
    _assert_unpaid(short_row)
    assert "insufficient width" in short_row.value.message

    with pytest.raises(ProviderResponseError) as nonfinite:
        calibrated_distribution([float("nan"), 0.2, 0.3], calibrator, 3)
    _assert_unpaid(nonfinite)

    with pytest.raises(ProviderResponseError) as overflow:
        # Tiny temperature drives row/temperature past float64 range.
        tiny = RlcdCalibrator("m", 1.0, 1.0, CALIBRATOR_SCOPE, {3: 1e-3}, "h")
        calibrated_distribution([1e308, -1e308, 0.0], tiny, 3)
    _assert_unpaid(overflow)
    assert "could not be calibrated" in overflow.value.message


def test_malformed_row_type_fails_unpaid() -> None:
    with pytest.raises(ProviderResponseError) as exc_info:
        calibrated_distribution(["a", "b", "c"], _full_calibrator(), 3)  # type: ignore[list-item]
    _assert_unpaid(exc_info)
    assert "malformed" in exc_info.value.message


def test_canonical_constants_are_pinned() -> None:
    assert UPSTREAM_ABSTAIN_SENTINEL == "__insufficient_evidence__"
    assert CANONICAL_ABSTAIN_ID == "__abstain__"
    assert CALIBRATOR_SCOPE == "open_domain_calibrated_v1"


def test_sha256_of_written_calibrator_is_verifiable(tmp_path: Path) -> None:
    # Guard that the loader reads exactly the bytes we wrote on disk.
    path = _write(_valid_payload(), tmp_path)
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    _calibrator_from(_valid_payload(), tmp_path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
