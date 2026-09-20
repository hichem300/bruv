"""Deterministic JSON output for machine consumers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel

from bruv.application import ApplicationError


def to_jsonable(value: Any) -> Any:
    """Convert pydantic models and nested data to JSON-safe structures."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value


def render_json(value: BaseModel | Mapping[str, Any]) -> str:
    """Render one deterministic JSON object terminated by a newline."""
    payload = to_jsonable(value)
    return json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"


def render_success(result: BaseModel) -> str:
    """Render the machine success envelope."""
    return render_json(result)


def render_error(error: ApplicationError) -> str:
    """Render the machine error envelope."""
    envelope = {"ok": False, "error": error.to_dict()}
    return json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n"


__all__ = ["render_error", "render_json", "render_success", "to_jsonable"]
