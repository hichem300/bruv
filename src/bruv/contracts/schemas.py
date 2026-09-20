"""Generated JSON Schemas for machine consumers."""

from __future__ import annotations

from typing import Any

from bruv.domain.requests import DecisionRequest
from bruv.domain.results import DecisionResult

SCHEMA_VERSION = "0.1.0"

_IDS = {
    "request": "https://bruv.dev/schemas/0.1.0/request.schema.json",
    "output": "https://bruv.dev/schemas/0.1.0/output.schema.json",
    "error": "https://bruv.dev/schemas/0.1.0/error.schema.json",
}

_ERROR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "ok": {"type": "boolean", "const": False},
        "error": {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "message": {"type": "string"},
                "paid_request": {"type": "boolean"},
                "action": {"type": "string"},
            },
            "required": ["code", "message", "paid_request", "action"],
        },
    },
    "required": ["ok", "error"],
}


def _with_meta(schema: dict[str, Any], name: str) -> dict[str, Any]:
    schema["$id"] = _IDS[name]
    schema["version"] = SCHEMA_VERSION
    return schema


def request_schema() -> dict[str, Any]:
    return _with_meta(DecisionRequest.model_json_schema(), "request")


def result_schema() -> dict[str, Any]:
    return _with_meta(DecisionResult.model_json_schema(), "output")


def error_schema() -> dict[str, Any]:
    return _with_meta(dict(_ERROR_SCHEMA), "error")


SCHEMAS = {
    "request": request_schema,
    "output": result_schema,
    "error": error_schema,
}

__all__ = ["SCHEMA_VERSION", "SCHEMAS", "error_schema", "request_schema", "result_schema"]
