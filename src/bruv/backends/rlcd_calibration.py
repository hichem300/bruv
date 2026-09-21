"""RLCD per-K stratified calibration.

``per_k`` keys only; no global fallback to ``temperature`` or
``log_temperature``. Numpy is imported lazily inside the calibration
helper so importing this module never requires the optional
``rlcd-modernbert`` extra.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bruv.application import ProviderResponseError

UPSTREAM_ABSTAIN_SENTINEL = "__insufficient_evidence__"
CANONICAL_ABSTAIN_ID = "__abstain__"
CALIBRATOR_SCOPE = "open_domain_calibrated_v1"

_FORMAT_VERSION = "rlcd-calibrator-v1"
_CORRUPT_ACTION = "Re-download artifacts or reinstall 'bruv[rlcd-modernbert]'."
_LOGITS_ACTION = "Re-run the evaluation or check the RLCD model output."

_REQUIRED_FIELDS = (
    "model_id",
    "temperature",
    "log_temperature",
    "scope",
    "per_k",
    "artifact_hash",
)


@dataclass(frozen=True, slots=True)
class RlcdCalibrator:
    """Loaded RLCD calibrator artifact.

    ``per_k`` maps a choice count to its calibrated temperature. The global
    ``temperature``/``log_temperature`` pair is schema metadata only and is
    never used as a sampling temperature.
    """

    model_id: str
    temperature: float
    log_temperature: float
    scope: str
    per_k: dict[int, float]
    artifact_hash: str


def _calibrator_error(message: str) -> ProviderResponseError:
    return ProviderResponseError(
        message=message,
        paid_request=False,
        action=_CORRUPT_ACTION,
    )


def _logits_error(message: str) -> ProviderResponseError:
    return ProviderResponseError(
        message=message,
        paid_request=False,
        action=_LOGITS_ACTION,
    )


def _finite_number(value: Any) -> float | None:
    """Return ``value`` as a finite float, or ``None`` if it is not one.

    Booleans are never numbers here even though ``bool`` subclasses ``int``.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    if not math.isfinite(number):
        return None
    return number


def _nonempty_text(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value


def load_calibrator(path: Path) -> RlcdCalibrator:
    """Load and validate an RLCD calibrator artifact JSON file.

    Fails closed with sanitized errors: no file path, prompt, or response
    content ever enters an error message.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise _calibrator_error("RLCD calibrator artifact is missing or corrupt.") from None
    if not isinstance(raw, dict):
        raise _calibrator_error("RLCD calibrator artifact is not a JSON object.")

    if raw.get("format_version") != _FORMAT_VERSION:
        raise _calibrator_error("RLCD calibrator schema is unsupported.")

    missing = [field for field in _REQUIRED_FIELDS if field not in raw]
    if missing:
        raise _calibrator_error(f"RLCD calibrator is missing required field {missing[0]!r}.")

    model_id = _nonempty_text(raw["model_id"])
    if model_id is None:
        raise _calibrator_error("RLCD calibrator field 'model_id' is malformed.")

    artifact_hash = _nonempty_text(raw["artifact_hash"])
    if artifact_hash is None:
        raise _calibrator_error("RLCD calibrator field 'artifact_hash' is malformed.")

    scope = _nonempty_text(raw["scope"])
    if scope is None:
        raise _calibrator_error("RLCD calibrator field 'scope' is malformed.")
    if scope != CALIBRATOR_SCOPE:
        raise _calibrator_error(f"RLCD calibrator scope {scope!r} is not supported.")

    temperature = _finite_number(raw["temperature"])
    if temperature is None or temperature <= 0.0:
        raise _calibrator_error(
            "RLCD calibrator field 'temperature' must be a finite positive number."
        )

    log_temperature = _finite_number(raw["log_temperature"])
    if log_temperature is None:
        raise _calibrator_error("RLCD calibrator field 'log_temperature' must be a finite number.")

    per_k_raw = raw["per_k"]
    if not isinstance(per_k_raw, dict):
        raise _calibrator_error("RLCD calibrator field 'per_k' is malformed.")
    per_k: dict[int, float] = {}
    for key, value in per_k_raw.items():
        if not isinstance(key, str):
            raise _calibrator_error(f"RLCD calibrator per_k key {key!r} is not a string.")
        try:
            choice_count = int(key)
        except ValueError:
            raise _calibrator_error(
                f"RLCD calibrator per_k key {key!r} is not an integer string."
            ) from None
        if str(choice_count) != key:
            raise _calibrator_error(f"RLCD calibrator per_k key {key!r} is not an integer string.")
        scaled_temperature = _finite_number(value)
        if scaled_temperature is None:
            raise _calibrator_error(f"RLCD calibrator per_k[{key!r}] is not a finite number.")
        if scaled_temperature <= 0.0:
            raise _calibrator_error(f"RLCD calibrator per_k[{key!r}] must be positive.")
        per_k[choice_count] = scaled_temperature

    return RlcdCalibrator(
        model_id=model_id,
        temperature=temperature,
        log_temperature=log_temperature,
        scope=scope,
        per_k=per_k,
        artifact_hash=artifact_hash,
    )


def temperature_for(calibrator: RlcdCalibrator, k: int) -> float:
    """Return the calibrated temperature for exactly ``k`` choices.

    Only explicit ``per_k[k]`` entries count. There is no fallback to the
    global ``temperature`` or ``log_temperature``.
    """
    temperature = calibrator.per_k.get(k)
    if temperature is None:
        raise ProviderResponseError(
            message=f"RLCD calibrator has no temperature for K={k}.",
            paid_request=False,
            action="Use a question whose choice count has a calibrated per_k entry.",
        )
    return temperature


def calibrated_distribution(
    logits_row: Sequence[float],
    calibrator: RlcdCalibrator,
    k: int,
) -> tuple[list[float], int]:
    """Calibrate the first ``k`` logits into a probability distribution.

    Returns ``(probabilities, winner_index)``. Probabilities are float64,
    numerically stable, and explicitly renormalized a second time by dividing
    by their sum. No smoothing, clamping, or rescaling is applied.
    """
    if k < 1:
        raise _logits_error("RLCD choice count must be at least 1.")
    temperature = temperature_for(calibrator, k)

    # Local import: numpy belongs to the optional rlcd-modernbert extra only.
    import numpy as np  # type: ignore[import-not-found]

    try:
        row = np.asarray(logits_row, dtype=np.float64)
    except (TypeError, ValueError):
        raise _logits_error("RLCD logits row is malformed.") from None
    if row.ndim != 1 or row.size < k:
        raise _logits_error("RLCD logits row has insufficient width.")
    row = row[:k]
    if not np.all(np.isfinite(row)):
        raise _logits_error("RLCD logits row contains non-finite values.")

    with np.errstate(over="ignore", invalid="ignore"):
        scaled = row / temperature
    if not np.all(np.isfinite(scaled)):
        raise _logits_error("RLCD logits row could not be calibrated.")
    winner = int(np.argmax(scaled))
    shifted = scaled - scaled[winner]
    exps = np.exp(shifted)
    total = float(exps.sum())
    if total <= 0.0:
        raise _logits_error("RLCD logits row could not be calibrated.")
    probs = exps / total
    probs = probs / probs.sum()

    probabilities = [float(p) for p in probs]
    return probabilities, winner


__all__ = [
    "CANONICAL_ABSTAIN_ID",
    "CALIBRATOR_SCOPE",
    "UPSTREAM_ABSTAIN_SENTINEL",
    "RlcdCalibrator",
    "calibrated_distribution",
    "load_calibrator",
    "temperature_for",
]
